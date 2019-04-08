
# coding: utf-8

# In[ ]:


# Author: Jangho Park

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.display.precision = 15
import os

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy import stats
from sklearn.kernel_ridge import KernelRidge
import warnings
warnings.filterwarnings("ignore")

N_ROWS = 5e6
train = pd.read_csv('../input/train.csv',        
                #nrows= N_ROWS, 
                dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
print('Train input completed')

X_train = train['acoustic_data']
y_train = train['time_to_failure']


# In[ ]:


# del train
# del train

# Cut training data
rows = 150_000

# floor((629145480 / 150000)) = 4194
# 4194 * 150_000 = 629.....

num_seq = int(np.floor(X_train.shape[0] / rows))
X_train_raw = X_train[:int(np.floor(X_train.shape[0] / rows))*rows]
y_train_raw = y_train[:int(np.floor(y_train.shape[0] / rows))*rows]


# In[ ]:


# print(y_train[rows-1::rows])
# Sequence의 갯수 = 4194
# 각 Sequence의 길이 = 150_000
X_train = X_train_raw.values.reshape((-1, rows, 1))
# print(X_train.shape)

# 맨왼쪽=> 말그대로 time
# 중간 => 
y_train = y_train_raw[rows-1::rows].values
# y_train =  np.zeros((4194))

# for i in range(num_seq):
#      y_train[i] = np.average(y_train_raw.values[i * rows:(i+1)*rows:1])

# print(y_train.shape)
print(num_seq)


# In[ ]:


X_train.shape


# In[ ]:


y_train


# In[ ]:


import numpy as np
import pandas as pd
import os
import time
import random
import math
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Bidirectional, TimeDistributed, MaxPooling1D
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint


# Training/ Vaidation Split
X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                y_train,
                                                test_size= 0.2,
                                                random_state= 11)


# In[ ]:


################################################################################
################################################################################
################################################################################
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Flatten
from keras.optimizers import adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import regularizers

model = Sequential()

# model.add(Conv1D(filters=10, kernel_size=50, strides=100, activation='relu', input_shape = (rows, 1)))
# model.add(MaxPooling1D(100,100))
# model.add(Dropout(0.2))
# model.add(Bidirectional(LSTM(50, return_sequences=False)))
# model.add(Dropout(0.2))
# model.add(Bidirectional(LSTM(40, return_sequences=False)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))

# model.compile(optimizer=adam(lr=0.001), loss="mae")
# history = model.fit(X_train, Y_train,
#                     validation_data=(X_val, Y_val),
#                     epochs=30,
#                     verbose=2)

# model.summary()
earlyStopping = EarlyStopping(monitor='val_loss',
                              patience=10,
                              verbose=0,
                              mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5',
                           save_best_only=True,
                           monitor='val_loss',
                           mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=5,
                                   verbose=1,
                                   epsilon=1e-4,
                                   mode='min')
model.compile(loss='mean_absolute_error', optimizer= adam(lr=1e-4), metrics=['mean_absolute_error'])


# In[ ]:


################################################################################
model.fit(X_train, 
        y_train,
        batch_size= 16,
        epochs= 100, 
        validation_data= (X_val, y_val),
        callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
        verbose= 0
         )
###############################################################################
################################################################################


# In[ ]:


y_train_pred = model.predict(X_train)

train_score = mean_absolute_error(y_train.flatten(), y_train_pred)
print("Training Error", train_score)


# In[ ]:


y_pred = model.predict(X_val)

# Plot
plt.figure(figsize=(6, 6))
plt.scatter(y_val, y_pred)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.xlabel('actual', fontsize=12)
plt.ylabel('predicted', fontsize=12)
plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
plt.show()

# training Score
score = mean_absolute_error(y_val.flatten(), y_pred)
print(f'Score: {score:0.3f}')

# Submission
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

X_test = []

for segment in tqdm(submission.index):
        seg = pd.read_csv('../input/test/' + segment + '.csv')
        x = pd.Series(seg['acoustic_data'].values)
        X_test.append(x)

X_test = np.asarray(X_test)
X_test = X_test.reshape((-1, 1))
print(X_test.shape)
X_test = X_test[:int(np.floor(X_test.shape[0] / rows))*rows]
X_test= X_test.reshape((-1, rows, 1))
print(X_test.shape)
submission['time_to_failure'] = model.predict(X_test)
submission.to_csv('submission.csv')

