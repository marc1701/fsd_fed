#!/usr/env/bin python
# -*- encoding:utf-8 -*-


"""
Once you have downloaded the FSD50K dataset as instructed in the website at
https://zenodo.org/record/4060432 this script creates segments of mel
spectrograms from the wav files and splits them into train, val and test
subsets. The resulting folders with mel spectrograms are created inside the
dataset folder, with following sizes (make sure you have room for them):

FSD50K_train_1sec_segs 260GB
FSD50K_test_1sec_segs 114GB
FSD50K_val_1sec_segs 792KB
"""


from argparse import ArgumentParser
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import os, glob
import seaborn as sns; sns.set()
# custom code
from fed_fsd import *


# users must download FSD50K as instructed in https://zenodo.org/record/4060432
# and provide the path to the dataset folder
parser = ArgumentParser('Create melspectrograms from FSD50K in same folder')
parser.add_argument('-p', '--fsd50k_path', type=str, required=True,
                    help='Path to the official FSD50K installation')
args = parser.parse_args()
#
FSD50K_PATH = args.fsd50k_path
FSD50K_DEV = os.path.join(FSD50K_PATH, 'FSD50K.dev_audio', '')
FSD50K_TEST = os.path.join(FSD50K_PATH, 'FSD50K.eval_audio', '')
FSD50K_GT = os.path.join(FSD50K_PATH, 'FSD50K.ground_truth', '')
FSD50K_METADATA = os.path.join(FSD50K_PATH, 'FSD50K.metadata', '')
#
OUT_DEV = os.path.join(FSD50K_PATH, 'FSD50K_train_1sec_segs', '')
OUT_VAL = os.path.join(FSD50K_PATH, 'FSD50K_val_1sec_segs', '')
OUT_TEST = os.path.join(FSD50K_PATH, 'FSD50K_test_1sec_segs', '')


# segment the audio into t*f = 101 * 96
print('Creating mel spectrograms...')
segment_audio(FSD50K_DEV, OUT_DEV)
segment_audio(FSD50K_TEST, OUT_TEST)

# reformatting of csv info files
dev_info = pd.read_csv(os.path.join(FSD50K_GT, 'dev.csv'))
test_info = pd.read_csv(os.path.join(FSD50K_GT, 'eval.csv'))

# split dev to train and val
train_info = dev_info[dev_info.split == 'train']
val_info = dev_info[dev_info.split == 'val']

# move val files to their own directory
print('Splitting dev data into train/val...')
val_dir = OUT_VAL
if not os.path.exists(val_dir):
    os.mkdir(val_dir)

len_val_info = len(val_info)

for i, fname in enumerate(val_info.fname, 1):
    print(f'[{i}/{len_val_info}] train/val split processing:', fname)
    fname = str(fname)
    segs = glob.glob(f"{os.path.join(OUT_DEV, fname)}.*.pt")
    # segs = glob.glob(OUT_DEV + fname + '*')

    assert segs, f'{OUT_DEV + fname + "*"} should point to existing files!'
    # import pdb; pdb.set_trace()
    # try:
    #     assert segs, f'{OUT_DEV + fname + "*"} should point to existing files!'
    # except:
    #     print("\n\n\nDUPLICATE? CHECK IF SAME BEGIN IN OTHER PRIOR FILE", fname)

    for seg in segs:
        out_path = os.path.join(OUT_VAL, os.path.basename(seg))
        if os.path.exists(out_path):
            print('  already exists in', out_path, "removing from", seg)
            os.remove(seg)
        else:
            os.rename(seg, out_path)
            print('  moved', seg, 'to', out_path)


# load json files with uploader information and add to the info dataframes
print("Preparing CSV files...")
dev_json = pd.read_json(
    os.path.join(FSD50K_METADATA, 'dev_clips_info_FSD50K.json')).T
test_json = pd.read_json(
    os.path.join(FSD50K_METADATA, 'eval_clips_info_FSD50K.json')).T


train_info = add_uploader_info(train_info, dev_json)
val_info = add_uploader_info(val_info, dev_json)
test_info = add_uploader_info(test_info, test_json)

# add number of segments for each clip
train_info = add_n_segs(train_info, OUT_DEV)
val_info = add_n_segs(val_info, OUT_VAL)
test_info = add_n_segs(test_info, OUT_TEST)

# save final reworked csv info files
# train_info.iloc[:,:-1].to_csv(os.path.join(FSD50K_GT, 'train.csv'), index=False)
# val_info.iloc[:,:-1].to_csv(os.path.join(FSD50K_GT, 'val.csv'), index=False)
# test_info.iloc[:,:-1].to_csv(os.path.join(FSD50K_GT, 'test.csv'), index=False)
train_info.to_csv(os.path.join(FSD50K_GT, 'train.csv'), index=False)
val_info.to_csv(os.path.join(FSD50K_GT, 'val.csv'), index=False)
test_info.iloc[:,:-1].to_csv(os.path.join(FSD50K_GT, 'test.csv'), index=False)

print('Saved train.csv, val.csv, test.csv into', FSD50K_GT)
