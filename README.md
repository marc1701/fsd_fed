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

#### Dependencies:
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

Tested using Python 3.6.12 on Ubuntu 20.10. From a fresh virtual environment on a CUDA-enabled Linux system, the following installation order should work:


```
conda create --name flfsd50k python=3.6
conda activate flfsd50k
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
conda install pandas
conda install -c anaconda scipy
conda install -c conda-forge scikit-learn
conda install -c conda-forge pytorch-lightning
conda install -c anaconda seaborn
pip install ray
conda install -c conda-forge libsndfile
```

The resulting environment has been frozen into `requirements.txt`. Check the file for a full detail on versions and dependencies.


#### Dataset:


This repository requires the installation of the [FSD50K dataset](https://zenodo.org/record/4060432). The dataset should be downloaded following the official layout. Given a root `<FSD50K_PATH>`, (e.g. `./datasets/fsd50k`), the following steps should achieve that:

```
# mkdir and cd to e.g. ./datasets/fsd50k

# download, merge, uncompress and cleanup dev data
wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z01?download=1
wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z02?download=1
wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z03?download=1
wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z04?download=1
wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z05?download=1
wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.zip?download=1
for i in ./*download=1; do mv -v $i "${i/?download=1/}"; done
zip -s 0 FSD50K.dev_audio.zip --out unsplit.zip
unzip unsplit.zip  # this will uncompress the data into a folder with wav files
rm FSD50K.dev_audio.z*
rm unsplit.zip

# download, merge and uncompress eval data
wget https://zenodo.org/record/4060432/files/FSD50K.eval_audio.z01?download=1
wget https://zenodo.org/record/4060432/files/FSD50K.eval_audio.zip?download=1
for i in ./*download=1; do mv -v $i "${i/?download=1/}"; done
zip -s 0 FSD50K.eval_audio.zip --out unsplit.zip
unzip unsplit.zip  # this will uncompress the data into a folder with wav files
rm FSD50K.eval_audio.z*
rm unsplit.zip

# download and uncompress smaller files
wget https://zenodo.org/record/4060432/files/FSD50K.doc.zip?download=1
wget https://zenodo.org/record/4060432/files/FSD50K.ground_truth.zip?download=1
wget https://zenodo.org/record/4060432/files/FSD50K.metadata.zip?download=1
for i in ./*download=1; do mv -v $i "${i/?download=1/}"; done
for i in FSD50K.*.zip; do unzip $i; done  # this will uncompress the zip files into folders
for i in FSD50K.*.zip; do rm $i; done

# At this point we should be left with the filestructure as in https://zenodo.org/record/4060432
```


#### Usage:
To recreate the results reported in M. C. Green & M. D. Plumbley (2021),
[_Federated Learning With Highly Imbalanced Audio Data_](https://arxiv.org/abs/2105.08550), submitted for
publication, follow these steps (make sure all dependencies and the dataset are installed):

```
git clone https://github.com/marc1701/fsd_fed.git
cd fsd_fed

# Convert raw audio files into mel spectrograms. This may need a few hours and about
# 450GB of your disk
python fsd_create_mels.py -p <FSD50K_PATH>

# Perform a grid search on the FL setup as per paper
python raytune_script.py -p <FSD50K_PATH> -o <DATA_OUTPUT>
```


### Key Functions and Classes:
###  `FSD50K_MelSpec1s`
This is a custom PyTorch dataset object designed to load FSD50K data,
preprocessed into mel-spectrogram patches representing one-second clips.
The class is reliant on the directory structure established by the
`fsd_segment_setup.py` script.

###### Key Parameters:
`split` - can be set to `'train'`, `'val'`, or `'test'` and will return an
object loading the related split of the dataset. Default `'train'`.

`uploader_min` - a mininmum number of clips for a given FSD50K uploader's clips
to be included in the object's data. This is used to generate a set of FL
clients with a given minimum audio clip contribution. Default `0`, which will
load all of the data (no minimum).

`uploader_name` - used to set the class to load data from a specific uploader
at time of init. Default `None`.

`subdir` - used if FSD50K data is contained in a different directory to the
Python instance using this class. Default `None`.

###### Key Methods:
`set_uploader(uploader_name)` - used to set the class to access data from one
specific uploader after it has been instantiated. This method is used by
`federated_train` to make copies of this object referencing different uploaders.


###  `federated_train`
A function that will simulate a federated learning scenario, training a central
model by aggregating parameters from models trained conventionally on simulated
clients.

###### Parameters:
`model` - standard PyTorch model to be trained.

`device` - hardware device performing the training (CPU or GPU).

`loss_func` - standard PyTorch loss function.

`clients` - list of clients contributing parameters to the centralised model.
This should take the form of a list of `FSD50K_MelSpec1s` objects with
`uploader_name` set to FSD50K uploaders.

`rounds` - number of communications rounds to simulate.

`C` - fraction of clients to select in each comms. round.

`B` - batch size to use for client model training.

`E` - number of epochs to run on clients.

`lr` - learning rate

`val_dataloader` and `val_data` - data with which to calculate AUC.
Both need to be specified in order for patch-level predictions to be correctly aggregated over a whole clip.

`best_epoch_path` - file path to save

----

Use of `raytune_script.py` to conduct federated learning experiments is
recommended as `ray[tune]` will take care of saving model parameters at
every round and saving progress reports, as well as parallelizing the grid
search across multiple GPUs, if desired.

[1]:https://zenodo.org/record/4060432
[2]:https://arxiv.org/abs/2010.00475
