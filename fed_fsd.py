import glob

class FSD50k(Dataset):
    '''Dataset object for FSD50K. This will load audio and labels
    from the development set by default, from the eval set if dev is
    set to False. Assumes default FSD50k directory names.'''

    def __init__(self, transforms=fed_melspec,
                 anno_dir='FSD50K.ground_truth/',
                 dev=True):

        if dev: prefix = 'dev'
        else: prefix = 'eval'

        self.transforms = transforms

        self.audio_path = 'FSD50K.' + prefix + '_audio/'

        # read file list and annotations
        self.info = pd.read_csv(anno_dir + prefix + '.csv')
        self.vocab = pd.read_csv(anno_dir + 'vocabulary.csv')

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
        filepath = self.audio_path + str(
            self.info.iloc[item]['fname']) + '.wav'
        x = torchaudio.load(filepath)[0]

        if self.transforms is not None:
            x = self.transforms(x)

        return x, y

    def __len__(self): return len(self.info)

# transform as per spec in FSD50k paper
# paper states 30ms frames with 10ms overlap, output of shape
# t*f = 101*96 - this works out at 660-sample frames with 220-sample
# overlap, given fs = 22050 Hz
fsd_melspec = torch.nn.Sequential(
    torchaudio.transforms.Resample(orig_freq=44100, new_freq=22050),
    torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_mels=96,
                                         n_fft=660, hop_length=220)
)



def segment_audio(dataset_dir, out_dir, n_frames=101, n_overlap=50):

    file_list = glob.glob(dataset_dir + '*')

    for filepath_in in file_list:
        filename_in = filepath_in[filepath_in.find('/')+1:].split('.')[0]

        # load in audio clip and calculate mel spectrogram
        audio_mel = fsd_melspec(torchaudio.load(filepath_in)[0]).squeeze().T

        # if the clip must be extended in order to reach one second
        if len(audio_mel) < n_frames:
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
