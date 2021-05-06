import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import os, glob, datetime, io
import torch, torchaudio
import copy
import lmdb
import random
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import pickle

from pytorchtools import EarlyStopping

import seaborn as sns; sns.set()
from metrics_plots import *


class FSD_CRNN(nn.Module):
    '''implementation of baseline CRNN model described in FSD50K paper'''

    def __init__(self):
        super().__init__()

        # three convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(5,5), padding=2),
            nn.BatchNorm2d(128, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,5))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(5,5), padding=2),
            nn.BatchNorm2d(128, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,4))
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(5,5), padding=2),
            nn.BatchNorm2d(128, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )

        # bidirectional GRU layer
        self.bigru = nn.GRU(input_size=256, hidden_size=64, bidirectional=True)

        # fully connected output layer
        self.fc = nn.Linear(128, 200)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # rearrange the data for recurrent layer
        # (2 frequency bands of 128 channels stacked)
        # GRU wants time steps, batch, features (stacked)
        x = x.permute(2, 0, 3, 1).reshape(12, -1, 256)

        # take final state as output of BiGRU
        _, x = self.bigru(x)
        # stack forward and backward outputs
        x = x.permute(1, 0, 2).reshape(-1, 128)

        x = self.fc(x)

        return torch.sigmoid(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FSD_VGG(nn.Module):
    '''implementation of VGG-like network described in FSD50K paper'''

    def __init__(self):
        super().__init__()

        # repeating myself here - better way?
        self.conv_group1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2,2))
        )


        self.conv_group2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2,2))
        )


        self.conv_group3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(128, track_running_stats=False),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2,2))
        )

        self.glob_maxpool = nn.MaxPool2d(kernel_size=(12,12))
        self.glob_avgpool = nn.AvgPool2d(kernel_size=(12,12))

        self.fc_layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.Linear(256, 200)
        )


    def forward(self, x):

        x = self.conv_group1(x)
        x = self.conv_group2(x)
        x = self.conv_group3(x)

        x_max = self.glob_maxpool(x).squeeze()
        x_avg = self.glob_avgpool(x).squeeze()

        # in case 'batch' of one is missing first dimension
        if x_max.ndim < 2:
            x_max = x_max.unsqueeze(0)
            x_avg = x_avg.unsqueeze(0)

        x = torch.cat((x_max, x_avg), 1)

        x = self.fc_layers(x)

        return torch.sigmoid(x)


class FSD50K_MelSpec1s(Dataset):
    '''Dataset object to load FSD50K preprocessed into 1-second clips with
    96-band mel-spectrogram representation'''

    def __init__(self, transforms=None, split='train',
                 uploader_min=0, uploader_name=None,
                 anno_dir='FSD50K.ground_truth/',
                 subdir=None):

        self.transforms = transforms
        self.data_path = 'FSD50K_' + split + '_1sec_segs/'

        if subdir:
            self.data_path = os.path.join(subdir, self.data_path)
            anno_dir = os.path.join(subdir, anno_dir)

        # load labels
        vocab = pd.read_csv(anno_dir + 'vocabulary.csv', header=None)
        self.labels = vocab[1]
        self.mids = vocab[2]
        self.len_vocab = len(vocab)

        self.info = pd.read_csv(anno_dir + split + '.csv')
        if uploader_min: # if user has set a value for minimum clips
            uploaders = (
                self.info.uploader.value_counts() >= uploader_min).to_dict()
            uploaders = [name for name in uploaders.keys() if uploaders[name]]
            # remove all uploaders that do not meet the minimum clip threshold
            self.info = self.info[self.info.uploader.isin(uploaders)]

        # in case user sets single uploader
        self._full_info = self.info
        # list of available uploaders
        self.all_uploaders = self._full_info.uploader.unique()

        # if user has specified an uploader, set it here
        if uploader_name: self.set_uploader(uploader_name)
        else: self._import_data()

    def __len__(self): return self.info.n_segs.sum()

    def set_uploader(self, uploader_name):
        self.info = self._full_info[self._full_info.uploader == uploader_name]
        self.uploader = uploader_name
        self._import_data()

    def _import_data(self):
        # this method of making a file list should work
        # regardless of whether full set or subset is used
        self.file_list = []
        for fname in self.info.fname:
            n_segs = int(self.info[self.info.fname==fname].n_segs)

            for n in range(n_segs):
                filepath = self.data_path + str(fname) + '.' + str(n) + '.pt'
                self.file_list.append(filepath)

        # order of clips in tensor
        self.clip_order = torch.tensor(self.info['fname'].to_numpy())
        # number of segments in tensor
        self.n_segs = torch.tensor(
            self.info['n_segs'].to_numpy()).unsqueeze(1)

        # set up ground truth array
        self.ground_truth = torch.zeros(len(self.info), self.len_vocab)
        for i, clip_number in enumerate(self.clip_order):
            tags = self.info.iloc[i]['mids'].split(',')
            for tag in tags:
                tag_idx = np.where(self.mids == tag)[0][0]
                self.ground_truth[i, tag_idx] = 1

        # other useful properties
        self.class_clip_n = self.ground_truth.sum(0).numpy()
        self.missing_classes = self.labels[self.class_clip_n == 0]

    def __getitem__(self, item):
        # load file
        filepath = self.file_list[item]
        x = torch.load(filepath)

        # find index of original audio clip from filename
        clip_index = int(filepath.split('/')[-1].split('.')[0])
        csv_index = np.where(self.clip_order == clip_index)[0][0]

        y = self.ground_truth[csv_index]

        # probably no transforms (data already processed) but good to include
        if self.transforms is not None:
            x = self.transforms(x)

        return x.unsqueeze(0), y, filepath

    def display_class_contents(self, height=20, width=20):
        '''displays a bar chart showing number of clips per class in dataset'''
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        fig.set_figheight(20)
        fig.set_figwidth(20)

        # need to edit this so it detects order of magnitude
        upper_ylim = round(int(torch.max(self.ground_truth.sum(0))), -3)

        for n, ax in enumerate((ax1, ax2, ax3, ax4)):

            l, u = n*50, n*50 + 50

            ax.set_ylim(0, upper_ylim)
            ax.bar(self.labels[l:u], self.class_clip_n[l:u])

            for tick in ax.get_xticklabels():
                # still not ideal visually
                tick.set_rotation(90)


# transform as per spec in FSD50K paper
# paper states 30ms frames with 10ms overlap, output of shape
# t*f = 101*96 - this works out at 660-sample frames with 220-sample
# overlap, given fs = 22050 Hz
fsd_melspec = torch.nn.Sequential(
    torchaudio.transforms.Resample(orig_freq=44100, new_freq=22050),
    torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_mels=96,
                                         n_fft=660, hop_length=220)
)

def segment_audio(in_dir, out_dir, n_frames=101, n_overlap=50):
    '''segment raw FSD50K audio into 1-second mel-spectrograms and
    save these in PyTorch .pt format'''

    file_list = glob.glob(in_dir + '*')

    if not os.path.exists(out_dir): os.mkdir(out_dir)

    for filepath_in in file_list:
        # this is the bit that is slightly dodgy
        filename_in = filepath_in.split('/')[-1].split('.')[0]

        # load in audio clip and calculate mel spectrogram
        audio_mel = fsd_melspec(torchaudio.load(filepath_in)[0]).squeeze().T

        # if the clip must be extended in order to reach one second
        while len(audio_mel) < n_frames:
            n_extra_frames = n_frames - len(audio_mel)
            audio_mel = torch.cat((audio_mel, audio_mel[:n_extra_frames]), 0)

        # calculate number of segments to be made from clip
        n_segments = int((len(audio_mel)/n_overlap)) - 1

        for i in range(n_segments):
            # figure out correct output filename
            filename_out = filename_in + '.' + str(i) + '.pt'
            filepath_out = out_dir + filename_out

            data = audio_mel[i * n_overlap: i * n_overlap + n_frames]

            # fix missing final frames
            if data.shape != torch.Size([101, 96]):
                data = torch.cat((data, data[-1].unsqueeze(0)))

            # save the 1-second of frames to a file
            torch.save(data, filepath_out)


def add_uploader_info(dataframe, json_info):
    '''add a column to pandas metadata with uploader i.d. for each clip'''
    uploader_list = []
    dataframe = dataframe.loc[:]

    for fname in dataframe.fname:
        uploader = json_info[json_info.index == fname].uploader.to_numpy()[0]
        uploader_list.append(uploader)

    dataframe['uploader'] = uploader_list

    return dataframe


def add_n_segs(dataframe, seg_dir):
    '''add a column to pandas metadata with number of segments for each clip'''
    dataframe = dataframe.loc[:]

    clip_order = dataframe['fname'].to_numpy()

    segs = glob.glob(seg_dir + '*')
    numsegs = np.array([sum('/' + str(clip_number) + '.' in filepath
                                     for filepath in segs)
                                     for clip_number in clip_order])
    dataframe['n_segs'] = numsegs

    return dataframe


def train(n_epochs, optimiser, model, loss_func, device,
          dataloader, val_dataloader=None, val_data=None,
          best_epoch_path='checkpoints/best.pt',
          checkpoints_dir='checkpoints/',
          early_stopping_active=False,
          plateau_catcher_active=False,
          early_stopping_patience=10,
          verbose=True,
          save_all_checkpoints=False):
    '''Train a model for set number of epochs. Will also calculate val
    performance if val data is provided. EarlyStopping active by default
    with a patience of 10 epochs'''

    # could definitely add kwargs for these
    if early_stopping_active:
        early_stopping = EarlyStopping(
            patience=early_stopping_patience, verbose=True, path=best_epoch_path)

    if plateau_catcher_active:
        plateau_catcher = ReduceLROnPlateau(optimiser, mode='max',
            patience=5, factor=0.5, verbose=True)

    loss_history = []
    val_loss_history = []
    # epoch loop
    for epoch in range(1, n_epochs + 1):

        # model in learning mode
        model.train()

        train_loss = 0.0

        # loop through all available batches of data
        for x, y_true, _ in dataloader:

            # send data to GPU if available
            x = x.to(device); y_true = y_true.to(device)

            # get predictions
            y_pred = model(x)

            # calculate losses
            loss = loss_func(y_pred, y_true)

            # clear gradients
            optimiser.zero_grad()

            # backpropagate loss
            loss.backward()

            # update weights
            optimiser.step()

            # sum losses over whole epoch
            train_loss += loss.item()

        loss_history.append(train_loss / len(dataloader))

        # !!! COULD VERY LIKELY SUBSTITUTE THIS FOR EVAL FUNC !!!
        if val_dataloader:
            # validation stage
            val_clip_scores, _ = model_eval(
                model, val_dataloader, val_data, device)
            # calcuate pr-auc
            val_pr_auc = pr_auc(
                val_clip_scores, val_data.ground_truth.to(device))

            # if improvements plateau, reduce lr
            if plateau_catcher_active: plateau_catcher.step(val_pr_auc)

            # early stopping
            # negative metric as expects loss to decrease
            if early_stopping_active:
                early_stopping(-val_pr_auc, model)
                if early_stopping.early_stop:
                    if verbose: print('Early stopping')
                    break
            val_loss_history.append(val_loss / len(val_dataloader))

        if save_all_checkpoints:
            checkpoint_path = checkpoints_dir +'epoch_' + str(epoch) + '.pt'
            torch.save(model.state_dict(), checkpoint_path)

        if val_dataloader:
            log_string = (
                '\n{} Epoch {}, Train loss {}, Val loss {}, Val PR-AUC {}'.format(
                datetime.datetime.now(), epoch,
                train_loss / len(dataloader),
                val_loss / len(val_dataloader),
                val_pr_auc))
        else:
            log_string = (
                '\n{} Epoch {}, Train loss {}'.format(
                datetime.datetime.now(), epoch,
                train_loss / len(dataloader)))

        if verbose: print(log_string, end=" ")

    return model, loss_history, val_loss_history


def model_eval(model, test_dataloader, test_data, device):
    '''evaluate the performance of a trained model - automatically aggregates
    scores for segments across a whole clip'''

    model.eval()
    # initialise output aggregation array
    test_clip_scores = torch.zeros(len(test_data.info), 200).to(device)

    with torch.no_grad():
        for x, y_true, filepath in test_dataloader:

            # send data to GPU if available
            x = x.to(device); y_true = y_true.to(device)

            # get predictions
            y_pred = model(x)

            # save predictions for aggregation across whole clips
            for i, output in enumerate(y_pred):
                clip_number = int(filepath[i].split('/')[-1].split('.')[0])
                clip_index = np.where(test_data.clip_order == clip_number)[0][0]
                test_clip_scores[clip_index] += output

        # divide scores by number of segments per clip
        # (averaging across segments)
        test_clip_scores /= test_data.n_segs.to(device)

    return test_clip_scores.to(device), test_data.ground_truth
    # i.e. y_pred, y_true


def federated_train(model, device, loss_func, clients, rounds,
                    C=0.2, B=64, E=4, lr=5e-04,
                    val_dataloader=None, val_data=None,
                    early_stopping_active=False,
                    best_epoch_path='checkpoints/best.pt',
                    verbose=True):
    '''train a central model using federated learning'''

    if early_stopping_active:
        early_stopping = EarlyStopping(
            patience=20, verbose=True, path=best_epoch_path)

    glob_model = model.to(device)

    n_clients_to_select = round(C * len(clients))

    len_data = sum([len(client) for client in clients])

    val_loss_history = []

    for i in range(rounds):
        if verbose: print('Round ' + str(i+1))

        # randomly select clients for this round
        this_rounds_clients = random.sample(clients, n_clients_to_select)

        # set up dict with dataloaders and models for each client
        this_rounds_data = dict()
        for client in this_rounds_clients:
            this_rounds_data[client.uploader] = [client,
                DataLoader(client, batch_size=B, num_workers=12, shuffle=True),
                    copy.deepcopy(glob_model)]

        # train client models
        for key, item in this_rounds_data.items():

            client, dataloader, model = item
            # optim must get the client model parameters
            optim = torch.optim.Adam(model.parameters(), lr=lr)

            if verbose: print("\nTraining {}'s model".format(key), end='')

            train(n_epochs=E,
                  optimiser=optim,
                  model=model,
                  loss_func=loss_func,
                  device=device,
                  dataloader=dataloader)

        # store old global model state
        prev_glob_model = copy.deepcopy(glob_model)

        # zero out global model parameters
        zero_model_params(glob_model)

        # total weight for previous global model
        untrained_weight = 1

        # add weighted parameters from this round's local models
        for key, item in this_rounds_data.items():

            client, _, client_model = item
            client_weight = len(client) / len_data

            fed_weight_update(glob_model, client_model, client_weight)

            untrained_weight -= client_weight

        # add previous model weights back in
        # (representing the local models not trained this time around)
        # (this behaviour should be alterable after Nilsson)
        fed_weight_update(glob_model, prev_glob_model, untrained_weight)

        if val_data:
            if verbose: print('\nEvaluating global model...')
            y_pred_val, y_true_val = model_eval(
                glob_model, val_dataloader, val_data, device)
            auc_val = pr_auc(y_pred_val, y_true_val)

            val_loss_history.append(auc_val)

            if verbose:
                print('\nGlobal Val PR-AUC: {:.4f}\n'.format(float(auc_val)))

            if early_stopping_active:
                early_stopping(-auc_val, glob_model)
                if early_stopping.early_stop:
                    if verbose: print('Early stopping')
                    return torch.load(best_epoch_path), val_loss_history

    return torch.load(best_epoch_path), np.array(val_loss_history)


def fed_weight_update(model_a, model_b, weight):
    with torch.no_grad():
        for params_a, params_b in zip(model_a.parameters(), model_b.parameters()):
            params_a.data += weight * params_b.data

def zero_model_params(model):
    device = torch.device('cuda') if next(
        model.parameters()).is_cuda else torch.device('cpu')
    with torch.no_grad():
        for param in model.parameters():
            param.data = torch.zeros(param.data.shape).to(device)
