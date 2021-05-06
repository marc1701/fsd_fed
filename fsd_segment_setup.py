import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import os, glob
import seaborn as sns; sns.set()

# custom code
from fed_fsd import *

# segment the audio into t*f = 101 * 96
segment_audio('FSD50K.dev_audio/', 'FSD50K_train_1sec_segs/')
segment_audio('FSD50K.eval_audio/', 'FSD50K_test_1sec_segs/')

# reformatting of csv info files
dev_info = pd.read_csv('FSD50K.ground_truth/dev.csv')
test_info = pd.read_csv('FSD50K.ground_truth/eval.csv')

# split dev to train and val
train_info = dev_info[dev_info.split == 'train']
val_info = dev_info[dev_info.split == 'val']

# move val files to their own directory
val_dir = 'FSD50K_val_1sec_segs/'
if not os.path.exists(val_dir): os.mkdir(val_dir)
for fname in val_info.fname:

    fname = str(fname)
    segs = glob.glob('FSD50K_train_1sec_segs/' + fname + '*')

    for seg in segs:
        os.rename(seg, 'FSD50K_val_1sec_segs/' + seg.split('/')[1])

# load json files with uploader information and add to the info dataframes
dev_json = pd.read_json('FSD50K.metadata/dev_clips_info_FSD50K.json').T
test_json = pd.read_json('/FSD50K.metadata/eval_clips_info_FSD50K.json').T

train_info = add_uploader_info(train_info, dev_json)
val_info = add_uploader_info(val_info, dev_json)
test_info = add_uploader_info(test_info, test_json)

# add number of segments for each clip
train_info = add_n_segs(train_info, 'FSD50K_train_1sec_segs/')
val_info = add_n_segs(val_info, 'FSD50K_val_1sec_segs/')
test_info = add_n_segs(test_info, 'FSD50K_test_1sec_segs/')

# save final reworked csv info files
train_info.iloc[:,:-1].to_csv('FSD50K.ground_truth/train.csv', index=False)
val_info.iloc[:,:-1].to_csv('FSD50K.ground_truth/val.csv', index=False)
test_info.iloc[:,:,-1].to_csv('FSD50K.ground_truth/test.csv', index=False)
