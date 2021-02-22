import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import os, glob, datetime
import torch, torchaudio
import copy
import random
from torch import nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorchtools import EarlyStopping

import seaborn as sns; sns.set()
from metrics_plots import *


class FSD_CRNN(nn.Module):
    '''implementation of baseline CRNN model described in FSD50k paper'''

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
    '''implementation of VGG-like network described in FSD50k paper'''

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

        x = torch.cat((x_max, x_avg), 1)

        x = self.fc_layers(x)

        return torch.sigmoid(x)


class FSD50k_MelSpec1s(Dataset):
    '''Dataset object to load FSD50k preprocessed into 1-second clips with
    96-band mel-spectrogram representation'''

    def __init__(self, transforms=None, split='train',
                 uploader_min=0, uploader_name=None,
                 anno_dir='FSD50K.ground_truth/'):

        self.transforms = transforms

        self.data_path = 'FSD50k_' + split + '_1sec_segs/'
        
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
        clip_index = int(filepath[filepath.find('/') +1:].split('.')[0])
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
                
     
        

# transform as per spec in FSD50k paper
# paper states 30ms frames with 10ms overlap, output of shape
# t*f = 101*96 - this works out at 660-sample frames with 220-sample
# overlap, given fs = 22050 Hz
fsd_melspec = torch.nn.Sequential(
    torchaudio.transforms.Resample(orig_freq=44100, new_freq=22050),
    torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_mels=96,
                                         n_fft=660, hop_length=220)
)

class FSD50k(Dataset):
    '''Dataset object for FSD50K. This will load audio and labels
    from the development set by default, from the eval set if dev is
    set to False. Assumes default FSD50k directory names.'''

    def __init__(self, transforms=fsd_melspec,
                 anno_dir='FSD50K.ground_truth/',
                 dev=True):

        if dev: prefix = 'dev'
        else: prefix = 'eval'

        self.transforms = transforms

        self.audio_path = 'FSD50K.' + prefix + '_audio/'

        # read file list and annotations
        self.info = pd.read_csv(anno_dir + prefix + '.csv')
        self.vocab = pd.read_csv(anno_dir + 'vocabulary.csv', header=None)

        self.labels = vocab[1]
        self.mids = vocab[2]

    def __getitem__(self, item):

        # set up array of zeros for binarised output
        y = torch.zeros((len(vocab)), dtype=torch.bool)

        # get m_ids from metadata as python list
        tags = self.info.iloc[item]['mids'].split(',')

        # binarised indication of tags
        for tag in tags:
            y[np.where(self.mids == tag)[0][0]] = True

        # load associated audio file
        filepath = self.audio_path + str(self.info.iloc[item]['fname']) + '.wav'
        x = torchaudio.load(filepath)[0]

        if self.transforms is not None:
            x = self.transforms(x)

        return x, y

    def __len__(self): return len(self.info)


def segment_audio(file_list, out_dir, n_frames=101, n_overlap=50):

    for filepath_in in file_list:
        # this is the bit that is slightly dodgy
        filename_in = filepath_in.split('/')[2].split('.')[0]

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

            # save the 1-second of frames to a file
            torch.save(audio_mel[i * n_overlap: i * n_overlap + n_frames],
                filepath_out)

# calculate how many extra frames are needed for exact split
# n_extra_frames = n_frames - len(audio_mel) % n_frames

# audio_mel = torch.cat((audio_mel, audio_mel[:n_extra_frames]), 0)

def train(n_epochs, optimiser, model, loss_func, device,
          dataloader, val_dataloader=None, val_data=None,
          best_epoch_path='checkpoints/best.pt',
          checkpoints_dir='checkpoints/',
          early_stopping_active=True,
          plateau_catcher_active=True):

    # could definitely add kwargs for these
    if early_stopping_active:
        early_stopping = EarlyStopping(
            patience=10, verbose=True, path=best_epoch_path)

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
            model.eval()

            val_clip_scores = torch.zeros(len(val_data.info), 200).to(device)
            val_loss = 0.0

            with torch.no_grad():
                for x, y_true, filepath in val_dataloader:

                    # send data to GPU if available
                    x = x.to(device); y_true = y_true.to(device)

                    # get predictions
                    y_pred = model(x)

                    # calculate losses
                    loss = loss_func(y_pred, y_true)

                    val_loss += loss.item()

                    # save predictions for aggregation across whole clips
                    for i, output in enumerate(y_pred):
                        clip_number = int(filepath[i].split('/')[1].split('.')[0])
                        clip_index = np.where(
                            val_data.clip_order == clip_number)[0][0]
                        val_clip_scores[clip_index] += output

                # divide scores by number of segments per clip
                # (averaging across segments)
                val_clip_scores /= val_data.n_segs.to(device)

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
                        print('Early stopping')
                        break
            val_loss_history.append(val_loss / len(val_dataloader))

        checkpoint_path = checkpoints_dir +'epoch_' + str(epoch) + '.pt'
        torch.save(model.state_dict(), checkpoint_path)

        if val_dataloader:
            print('{} Epoch {}, Train loss {}, Val loss {}, Val PR-AUC {}'.format(
                datetime.datetime.now(), epoch,
                train_loss / len(dataloader),
                val_loss / len(val_dataloader),
                val_pr_auc))
        else:
            print('{} Epoch {}, Train loss {}'.format(
                datetime.datetime.now(), epoch,
                train_loss / len(dataloader)))
    return model, loss_history, val_loss_history


def model_eval(model, test_dataloader, test_data, device,
    best_epoch_path='checkpoints/best.pt'):

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
                clip_number = int(filepath[i].split('/')[1].split('.')[0])
                clip_index = np.where(test_data.clip_order == clip_number)[0][0]
                test_clip_scores[clip_index] += output

        # divide scores by number of segments per clip
        # (averaging across segments)
        test_clip_scores /= test_data.n_segs.to(device)

    return test_clip_scores, test_data.ground_truth # (i.e. y_pred, y_true)

# TEST THIS NEXT
def federated_train(model, device, loss_func, clients, rounds, 
                    C=0.2, B=64, E=4, lr=5e-04, lr_decay=0):
# C = 0.2 # fraction of clients to select for training (1/5th)
# B = 64 # local minibatch size
# E = 4 # local epochs - keeping this low - best benchmark was E=4
# lr = 5e-4 # same as FSD benchmark
# lr_decay = 0 # for now

    glob_model = model.to(device)
    
    n_clients_to_select = round(C * len(clients))
    
    len_data = sum([len(client) for client in clients])
    
    for i in range(rounds):
        print('Round ', str(i))
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
            optim = torch.optim.Adam(model.parameters(), lr=lr)
            print("\nTraining {}'s model".format(key))

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
        # add weighted weights (ha) from this round's local models
        print('\nMerging model weights...')
        for key, item in this_rounds_data.items():

            client, _, client_model = item
            client_weight = len(client) / len_data

            fed_weight_update(glob_model, client_model, client_weight)

            untrained_weight -= client_weight

        # add previous model weights back in 
        # (representing the local models not trained this time around)
        # (this behaviour should be alterable after Nilsson)
        fed_weight_update(glob_model, prev_glob_model, untrained_weight)
    
            
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