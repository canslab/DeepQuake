# CNN-RNN Approach

import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout, GRU
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# Fix seeds
from numpy.random import seed
seed(537)
from tensorflow import set_random_seed
set_random_seed(6214)


# Import
print("Loading data...")
start = time.time()
data = pd.read_csv("../data/train.csv", dtype={"acoustic_data": np.float32, "time_to_failure": np.float32})
print("Loaded training data")
print("Executed in", round(time.time()-start), "seconds")

X_train = data['acoustic_data']
y_train = data['time_to_failure']

# Free up memory
del data

# Cut training data
seg_length = 150000
X_train = X_train[:int(np.floor(X_train.shape[0] / seg_length))*seg_length]
X_train= X_train.values.reshape((-1, seg_length, 1))
print(X_train.shape)

y_train = y_train[:int(np.floor(y_train.shape[0] / seg_length))*seg_length]
y_train = y_train[seg_length-1::seg_length].values
print(y_train.shape)

# Training/ Vaidation Split
X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                y_train,
                                                test_size= 0.1,
                                                random_state= 11)

# Define model
cb = [ModelCheckpoint("deep_cnn.hdf5", save_best_only=True, period=3)]

# CNN Encoder
model = Sequential()
model.add(Conv1D(4, 10, strides=10, activation='relu', input_shape=(seg_length, 1))) #150,000 x 1 --> 15,000 x 4
model.add(Conv1D(8, 10, strides=10, activation='relu')) # 15,000 x 4 --> 1500 x 8
model.add(Dropout(0.2))
model.add(Conv1D(30, 10, strides=2, activation='relu')) # 1500 x 8 --> 750 x 30
model.add(MaxPooling1D(2)) # 750 x 30 --> 375 x 30
model.add(Conv1D(60, 5, strides=3, activation='relu')) # 375 x 30 --> 125 x 60
model.add(Dropout(0.1))

#  GRU predictor
model.add(GRU(48))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

# Compile and fit model
model.compile(optimizer=adam(lr=0.0005), loss="mae")

model.fit(X_train,
        y_train,
        batch_size= 32,
        epochs= 30,
        validation_data= (X_val, y_val),
        callbacks=cb,
        verbose= 2
        )

# Load submission file
submission = pd.read_csv('../data/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})

# Load each test data, create the feature matrix, get numeric prediction
for i, seg_id in enumerate(tqdm(submission.index)):
  #  print(i)
    seg = pd.read_csv('../data/test/' + seg_id + '.csv')
    x = seg['acoustic_data'].values
    x = x.reshape((-1, seg_length, 1))
    submission.time_to_failure[i] = model.predict(x)

submission.head()

# Save
submission.to_csv('deep_cnn.csv')
