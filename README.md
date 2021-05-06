# Federated Learning for Audio in FSD50K

## by Marc C. Green

A system designed to test federated learning (FL) with audio data where clients
are highly imbalanced. Models are trained on the [FSD50K dataset][1], with
clients based on the 'uploader' metadata. This can result in come clients with
thousands of audio clips, with many more with only a single clip. The main
features of the code provided here are:

- Script performing segmentation of FSD50K dataset into 101x96 mel-spectrograms
as specified in the [original FSD50K paper][2].
- Reformatting of metadata to include uploader info and number of created
segments per original clip.
- PyTorch Dataset objects for the FSD50K mel-spectrogram data.
- Functions to train models using FL.
- Script utilising `ray[tune]` to perform a grid search of FL parameters.

#### Requirements:
- Python 3.6 or later
- Python modules:
  - [numpy](http://www.numpy.org/)
  - [scipy](https://www.scipy.org/)
  - [scikit-learn](http://scikit-learn.org/stable/)
  - [pandas](http://pandas.pydata.org/)
  - [matplotlib](https://matplotlib.org/)
  - [seaborn](https://seaborn.pydata.org/)
  - [pytorch](https://pytorch.org/)
  - [torchaudio](https://pytorch.org/audio/stable/index.html)
  - [pytorch_lightning](https://www.pytorchlightning.ai/)
  - [ray](https://docs.ray.io/en/latest/tune/)

Tested using Python 3.6.12 on Ubuntu 20.10

#### Usage:
To recreate the results reported in the upcoming WASPAA paper
_Federated Learning With Highly Imbalanced Audio Data_, simply run
`fsd_segment_setup.py` in the FSD50K home directory, followed by
`raytune_script.py`.

[1]:https://zenodo.org/record/4060432
[2]:https://arxiv.org/abs/2010.00475
