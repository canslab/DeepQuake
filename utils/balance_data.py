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
data = pd.read_csv("../data/train.csv",  nrows=10**8, dtype={"acoustic_data": np.float32, "time_to_failure": np.float32})
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

# Write output
train_X = [x[0] for x in train_set]
train_Y = [x[1] for x in train_set]
val_X = [x[0] for x in val_set]
val_Y = [x[1] for x in val_set]

for i in tqdm(range(len(train_set))):
    seg_id += 1
    with open("segment" + str(seg_id) + "X.csv", "w") as outfile:
        writer = csv.writer(outfile, delimiter=',', quotechar='"')
        X = train_set[i][0]
        writer.writerow(X)

"""
train = pd.DataFrame.from_records(train_set, columns=["signal","time"])
val = pd.DataFrame.from_records(val_set,  columns=["signal","time"])
train.to_csv('train.csv')
val.to_csv('val.csv')
"""
