class FSD50k(Dataset):
    '''Dataset object for FSD50K. This will load audio and labels
    from the development set by default, from the eval set if dev is
    set to False. Assumes default FSD50k directory names.'''

    def __init__(self, transforms=None,
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
# still need to do the one-second patches as described
# n.b. paper does not specify fft length
fsd_melspec = torch.nn.Sequential(
    torchaudio.transforms.Resample(orig_freq=44100, new_freq=22050),
    torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_mels=96)
)
