#matplotlib inline

from os import listdir, makedirs
from os.path import isfile, join, basename, splitext, isfile, exists

import numpy as np
import pandas as pd

from tqdm import tqdm_notebook

import tensorflow as tf
import keras.backend as K

import keras
from keras.models import Sequential, Model
from keras.layers import Dropout, Dense, Flatten, BatchNormalization
from keras.layers import Convolution1D, ZeroPadding1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Concatenate, Average, Maximum, CuDNNLSTM, CuDNNGRU, Bidirectional, TimeDistributed
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.engine.input_layer import Input
from keras.models import load_model

import matplotlib.pyplot as plt
#import seaborn as sns

pd.set_option('precision', 30)
np.set_printoptions(precision = 30)

np.random.seed(7723)
tf.set_random_seed(1090)
###Read the data
train_df = pd.read_csv('../data/train.csv', dtype={'acoustic_data': np.int8, 'time_to_failure': np.float32})

#####Length of training sample
print(train_df.acoustic_data.values.shape)

###Assign train smaples and label
X_train = train_df.acoustic_data.values
y_train = train_df.time_to_failure.values

### Find different segments in the training data 
ends_mask = np.less(y_train[:-1], y_train[1:])
segment_ends = np.nonzero(ends_mask)
train_segments = []
start = 0
for end in segment_ends[0]:
    train_segments.append((start, end))
    start = end
    
print(train_segments)
print(len(train_segments))

