import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import os, glob
import torch, torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
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
                 anno_dir='FSD50K.ground_truth/'):

        self.transforms = transforms

        data_path = 'FSD50k_' + split + '_1sec_segs'

        self.info = pd.read_csv(anno_dir + split + '.csv')

        self.file_list = glob.glob(data_path + '/*')

        vocab = pd.read_csv(anno_dir + 'vocabulary.csv', header=None)
        self.labels = vocab[1]
        self.mids = vocab[2]
        self.len_vocab = len(vocab)

        # order of clips in tensor
        self.clip_order = torch.tensor(self.info['fname'].to_numpy())
        # number of segments in tensor
        self.n_segs = torch.tensor(
            self.info['n_segs'].to_numpy()).unsqueeze(1)

        # set up ground truth array
        # hopefully pr-auc can do the whole array
        # but can compare this to the output array line-by-line if needs be
        self.ground_truth = torch.zeros(len(self.info), self.len_vocab)
        for i, clip_number in enumerate(self.clip_order):
            tags = self.info.iloc[i]['mids'].split(',')
            for tag in tags:
                tag_idx = np.where(self.mids == tag)[0][0]
                self.ground_truth[i, tag_idx] = 1


    def __len__(self): return len(self.file_list)

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
