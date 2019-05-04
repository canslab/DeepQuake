#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gc
import os
import time
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')


# In[ ]:


pd.set_option('precision', 30)
np.set_printoptions(precision = 30)

np.random.seed(7723)
tf.set_random_seed(1090)


# In[ ]:


###Read the data
train_df = pd.read_csv('../data/train.csv', dtype={'acoustic_data': np.int8, 'time_to_failure': np.float32})


# In[ ]:


print("Train: rows:{} cols:{}".format(train_df.shape[0], train_df.shape[1]))


# In[ ]:


rows = 150000
segments = int(np.floor(train_df.shape[0] / rows))
print("Number of segments: ", segments)


# In[ ]:


train_X = pd.DataFrame(index=range(segments), dtype=np.float64)
train_y = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])


# In[ ]:


def create_features(seg_id, seg, X):
    xc = pd.Series(seg['acoustic_data'].values)   
    zc = np.fft.fft(xc)
    realFFT = np.real(zc)
    imagFFT = np.imag(zc)
    X.loc[seg_id, 'mean'] = xc.mean()
    X.loc[seg_id, 'std'] = xc.std()
    X.loc[seg_id, 'max'] = xc.max()
    X.loc[seg_id, 'min'] = xc.min()
    X.loc[seg_id, 'sum'] = xc.sum()
    X.loc[seg_id, 'mad'] = xc.mad()
    X.loc[seg_id, 'kurt'] = xc.kurtosis()
    X.loc[seg_id, 'skew'] = xc.skew()
    X.loc[seg_id, 'med'] = xc.median()
    X.loc[seg_id, 'abs_mean'] = np.abs(xc).mean()
    X.loc[seg_id, 'q95'] = np.quantile(xc, 0.95)
    X.loc[seg_id, 'q99'] = np.quantile(xc, 0.99)
    X.loc[seg_id, 'q05'] = np.quantile(xc, 0.05)
    X.loc[seg_id, 'q01'] = np.quantile(xc, 0.01)
    X.loc[seg_id, 'Rmean'] = realFFT.mean()
    X.loc[seg_id, 'Rstd'] = realFFT.std()
    X.loc[seg_id, 'Rmax'] = realFFT.max()
    X.loc[seg_id, 'Rmin'] = realFFT.min()
    X.loc[seg_id, 'Imean'] = imagFFT.mean()
    X.loc[seg_id, 'Istd'] = imagFFT.std()
    X.loc[seg_id, 'Imax'] = imagFFT.max()
    X.loc[seg_id, 'Imin'] = imagFFT.min()
    X.loc[seg_id, 'std_first_50000'] = xc[:50000].std()
    X.loc[seg_id, 'std_last_50000'] = xc[-50000:].std()
    X.loc[seg_id, 'std_first_10000'] = xc[:10000].std()
    X.loc[seg_id, 'std_last_10000'] = xc[-10000:].std()


# In[ ]:


# iterate over all segments
for seg_id in tqdm_notebook(range(segments)):
    seg = train_df.iloc[seg_id*rows:seg_id*rows+rows]
    create_features(seg_id, seg, train_X)
    train_y.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]


# In[ ]:


submission = pd.read_csv('../data/sample_submission.csv', index_col='seg_id')
test_X = pd.DataFrame(columns=train_X.columns, dtype=np.float64, index=submission.index)


# In[ ]:


for seg_id in tqdm_notebook(test_X.index):
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    create_features(seg_id, seg, test_X)


# In[ ]:


print("Train X: {} y: {} Test X: {}".format(train_X.shape, train_y.shape, test_X.shape))


# In[ ]:


train_X.head()


# In[ ]:


test_X.head()


# In[ ]:


###feature exploration
def plot_distplot(feature):
    plt.figure(figsize=(16,6))
    plt.title("Distribution of {} values in the train and test set".format(feature))
    sns.distplot(train_X[feature],color="green", kde=True,bins=120, label='train')
    sns.distplot(test_X[feature],color="blue", kde=True,bins=120, label='test')
    plt.legend()
    plt.show()


# In[ ]:


def plot_distplot_features(features, nlines=4, colors=['green', 'blue'], df1=train_X, df2=test_X):
    i = 0
    plt.figure()
    fig, ax = plt.subplots(nlines,2,figsize=(16,4*nlines))
    for feature in features:
        i += 1
        plt.subplot(nlines,2,i)
        sns.distplot(df1[feature],color=colors[0], kde=True,bins=40, label='train')
        sns.distplot(df2[feature],color=colors[1], kde=True,bins=40, label='test')
    plt.show()


# In[ ]:


features = ['mean', 'std', 'max', 'min', 'sum', 'mad', 'kurt', 'skew']
plot_distplot_features(features)


# In[ ]:


features = ['med','abs_mean', 'q95', 'q99', 'q05', 'q01']
plot_distplot_features(features,3)


# In[ ]:


features = ['Rmean', 'Rstd', 'Rmax','Rmin', 'Imean', 'Istd', 'Imax', 'Imin']
plot_distplot_features(features)


# In[ ]:


features = ['std_first_50000', 'std_last_50000', 'std_first_10000','std_last_10000']
plot_distplot_features(features,2)


# In[ ]:


scaler = StandardScaler()
scaler.fit(pd.concat([train_X, test_X]))
scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)
scaled_test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)


# In[ ]:


features = ['mean', 'std', 'max', 'min', 'sum', 'mad', 'kurt', 'skew']
plot_distplot_features(features, nlines=4, colors=['red', 'magenta'], df1=scaled_train_X, df2=scaled_test_X)


# In[ ]:


features = ['med','abs_mean', 'q95', 'q99', 'q05', 'q01']
plot_distplot_features(features, nlines=3, colors=['red', 'magenta'], df1=scaled_train_X, df2=scaled_test_X)


# In[ ]:


features = ['Rmean', 'Rstd', 'Rmax','Rmin', 'Imean', 'Istd', 'Imax', 'Imin']
plot_distplot_features(features, nlines=4, colors=['red', 'magenta'], df1=scaled_train_X, df2=scaled_test_X)


# In[ ]:


features = ['std_first_50000', 'std_last_50000', 'std_first_10000','std_last_10000']
plot_distplot_features(features, nlines=2, colors=['red', 'magenta'], df1=scaled_train_X, df2=scaled_test_X)

