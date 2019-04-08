# Author Mehmet Emre Sonmez
# @mes2311
import numpy as np
import pandas as pd
import os
import time
import random
import math
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Bidirectional
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint

start = time.time()

# Import data
train = pd.read_csv("../data/balanced/train_samples.csv", dtype={"signal": np.float32}).values
Y_train = pd.read_csv("../data/balanced/train_labels.csv", dtype={"time": np.float32}).values
val = pd.read_csv("../data/balanced/val_samples.csv", dtype={"signal": np.float32}).values
Y_val = pd.read_csv("../data/balanced/val_labels.csv", dtype={"time": np.float32}).values

print("Loaded training data")
print("Executed in", round(time.time()-start), "seconds")

# Separate to discrete samples
seg_length = 150000
num_samples = int(len(train)/len(Y_train))
print(num_samples)
X_train = []
for i in range(num_samples):
    X = train[i*seg_length:(i+1)*seg_length]
    X = (X-5) / 3 # Normalizing
    X_train.append(X)

print(len(X_train[0]))
print(len(X_train))

num_samples = int(len(val)/len(Y_val))
X_val = []
for i in range(num_samples):
    X = val[i*seg_length:(i+1)*seg_length]
    X = (X-5) / 3 # Normalizing
    X_val.append(X)

# Save model periodically
cb = [ModelCheckpoint("bi_lstm.hdf5", save_best_only=True, period=3)]

model = Sequential()

# Encoder CNN
model.add(Conv1D(30, 100, strides=100, activation='relu', input_shape=(seg_length, 1)))
model.add(Conv1D(60, 15, strides=15, activation='relu'))

# Feed representation into bidirectional LSTM
model.add(Bidirectional(LSTM(50, return_sequences=True)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# Compile and fit model
model.summary()
model.compile(optimizer=adam(lr=0.001), loss="mae")
history = model.fit(X_train, Y_train,
                    validation_data=(X_val, Y_val),
                    epochs=30,
                    verbose=2,
                    callbacks=cb)

# Load submission file
submission = pd.read_csv('../data/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})

# Load each test data, create the feature matrix, get numeric prediction
for i, seg_id in enumerate(tqdm(submission.index)):
    seg = pd.read_csv('../data/test/' + seg_id + '.csv')
    x = seg['acoustic_data'].values
    submission.time_to_failure[i] = model.predict(np.expand_dims(create_X(x), 0))

# Save
submission.to_csv('bi_lstm50.csv')
