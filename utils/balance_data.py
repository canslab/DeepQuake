import numpy as np
import pandas as pd
import random
import math
import time
import os
import csv
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt

start = time.time()

# Read training data entirely
data = pd.read_csv("../data/train.csv", dtype={"acoustic_data": np.float32, "time_to_failure": np.float32})
data.rename({"acoustic_data": "signal", "time_to_failure": "time"}, axis="columns", inplace=True)
print("Loaded training data")
print("Executed in", round(time.time()-start), "seconds")

# Show sample
print("Data shape:", data.shape)
print(data.head(5))

# Divide the dataset into segments of length 150,000
# Step_size determines total number of samples produced and overlap between samples
seg_length = 150000
step_size = 50000
num_segments = (data.shape[0]-seg_length) // step_size
print("Creating", num_segments, "segments")

# Count number of samples in each bucket
counts = {}
for i in range(17):
    counts[i] = 0

data_buckets = {}
for i in range(17):
    data_buckets[i] = []

for i in tqdm(range(num_segments)):
    train_X = data.signal.values[i*step_size:i*step_size+seg_length]
    train_Y = data.time.values[i*step_size+seg_length]
    sample = (train_X, train_Y)
    bucket = math.floor(train_Y)
    data_buckets[bucket].append(sample)
    counts[bucket] += 1

# Print class numbers
print(counts)

# Visualize class distribution
plt.bar(range(len(counts)), list(counts.values()), align='center')
plt.xticks(range(len(counts)), list(counts.keys()))
#plt.show()

# Rebalance classes
min_num = counts[13]
new_counts = []
for i in range(17):
    new_num = counts[i]
    if new_num > min_num:
        new_num = int((((counts[i]/min_num)-1)/3 + 1) * min_num) # Custom ratio
    new_counts.append(new_num)

# Split into train and validation
train_set = []
val_set = []
for i in range(17):
    class_data = data_buckets[i]
    random.shuffle(class_data)
    class_data = class_data[0:new_counts[i]]

    split = len(class_data)//10
    val_set += class_data[0:split]
    train_set += class_data[split:]

random.shuffle(train_set)
random.shuffle(val_set)

# Format data and write output
tX, tY = [], []
for sample in train_set:
    tX.append(sample[0])
    tY.append(sample[1])

vX, vY = [], []
for sample in val_set:
    vX.append(sample[0])
    vY.append(sample[1])

tX = np.array(tX).flatten().astype(int)
tY = np.array(tY).flatten()
vX = np.array(vX).flatten().astype(int)
vY = np.array(vY).flatten()

print(tX.shape)
print(vX.shape)

dtx = {"signal": tX}
dty = {"time": tY}
dvx = {"signal": vX}
dvy = {"time": vY}

train_samples = pd.DataFrame(data=dtx)
train_labels = pd.DataFrame(data=dty)
val_samples = pd.DataFrame(data=dvx)
val_labels = pd.DataFrame(data=dvy)

print("Writing new training data")
train_samples.to_csv("train_samples.csv", index_col = False)
train_labels.to_csv("train_labels.csv", index_col = False)

print("Writing new validation data")
val_samples.to_csv("val_samples.csv", index_col = False)
val_samples.to_csv("val_labels.csv", index_col = False)
