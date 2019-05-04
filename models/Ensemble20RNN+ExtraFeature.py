import numpy as np 
import pandas as pd
import os
from tqdm import tqdm
from scipy.stats import kurtosis,skew
from scipy.fftpack import fft
#from statsmodels import robust

# Import
float_data = pd.read_csv("./data/train.csv", engine='python', dtype={"acoustic_data": np.float32, "time_to_failure": np.float32}).values
print("Read data")


# Helper function for the data generator. Extracts mean, standard deviation, and quantiles per time step.
# Can easily be extended. Expects a two dimensional array.
def extract_features(z):
    #print(z.shape)
    pthreshold = 80
    nthreshold = -80
    pThresholdCount = []
    nThresholdCount = []
    absmean = []
    #quantile95 = []
    quantile99 = []
    #quantile05 = []
    quantile01 = []
    Rmean = []
    Rstd = []
    Rmax = []
    Rmin = []
    Imean = []
    Istd = []
    Imax = []
    Imin = []
    for i in range(z.shape[0]):
        #print(np.where(z[i] > (pthreshold-5)/6.21))
        zc = fft(z[i])
        realFFT = np.real(zc)
        imagFFT = np.imag(zc)
        Rmean.append(realFFT.mean())
        Rstd.append(realFFT.std())
        #Rmax.append(realFFT.max())
        Rmin.append(realFFT.min())
        Imean.append(imagFFT.mean())
        Istd.append(imagFFT.std())
        Imax.append(imagFFT.max())
        Imin.append(imagFFT.min())
        pThresholdCount.append(np.where(z[i] > (pthreshold-5)/6.21)[0].size)
        nThresholdCount.append(np.where(z[i] < (nthreshold+5)/6.21)[0].size)
        absmean.append(np.abs(z[i]).mean())
        #quantile95.append(np.quantile(z[i], 0.95))
        quantile99.append(np.quantile(z[i], 0.99))
        #quantile05.append(np.quantile(z[i], 0.05))
        quantile01.append(np.quantile(z[i], 0.01))
    pThresholdCount = np.asarray(pThresholdCount)
    nThresholdCount = np.asarray(nThresholdCount)
    absmean = np.asarray(absmean)
    #quantile95 = np.asarray(quantile95)
    quantile99 = np.asarray(quantile99)
    #quantile05 = np.asarray(quantile05)
    quantile01 = np.asarray(quantile01)
    Rmean = np.asarray(Rmean)
    Rstd = np.asarray(Rstd)
    #Rmax = np.asarray(Rmax)
    Rmin = np.asarray(Rmin)
    Imean = np.asarray(Imean)
    Istd = np.asarray(Istd)
    Imax = np.asarray(Imax)
    Imin = np.asarray(Imin)
    return np.c_[#z.mean(axis=1), 
                 z.min(axis=1),
                 z.max(axis=1),
                 z.std(axis=1),
                 pThresholdCount,
                 nThresholdCount,
                 #z.sum(axis=1),
                 #robust.mad(z,axis=1),
                 kurtosis(z,axis=1),
                 skew(z,axis=1),
                 #np.median(z,axis=1),
                 #quantile95,
                 quantile99,
                 #quantile05,
                 quantile01,
                 Rmean,
                 Rstd,
                 #Rmax,
                 Rmin,
                 Imean,
                 Istd,
                 Imax,
                 Imin
                ]

# For a given ending position "last_index", we split the last 150'000 values 
# of "x" into 150 pieces of length 1000 each. So n_steps * step_length should equal 150'000.
# From each piece, a set features are extracted. This results in a feature matrix 
# of dimension (150 time steps x features).  
def create_X(x, last_index=None, n_steps=150, step_length=1000):
    if last_index == None:
        last_index=len(x)
       
    assert last_index - n_steps * step_length >= 0
    
    # Reshaping and approximate standardization with mean 5 and std 6.21.
    #print(x[(last_index - n_steps * step_length):last_index].shape)
    temp = (x[(last_index - n_steps * step_length):last_index].reshape(n_steps, -1) - 5 ) / 6.21
    #print(temp.shape)
    # Extracts features of sequences of full length 1000, of the last 100 values and finally also 
    # of the last 10 observations. 
    return np.c_[extract_features(temp),
                 extract_features(temp[:, -step_length // 10:]),
                 #extract_features(temp[:, -step_length // 100:])
                ]

# Query "create_X" to figure out the number of features
n_features = create_X(float_data[0:150000][:, 0]).shape[1]
print("Our RNN is based on %i features"% n_features)
    
# The generator endlessly selects "batch_size" ending positions of sub-time series. For each ending position,
# the "time_to_failure" serves as target, while the features are created by the function "create_X".
def generator(data, min_index=0, max_index=None, batch_size=16, n_steps=150, step_length=1000):
    if max_index is None:
        max_index = len(data) - 1
     
    while True:
        # Pick indices of ending positions
        rows = np.random.randint(min_index + n_steps * step_length, max_index, size=batch_size)
        
        # Initialize feature matrices and targets
        samples = np.zeros((batch_size, n_steps, n_features))
        targets = np.zeros(batch_size, )
        
        for j, row in enumerate(rows):
            samples[j] = create_X(data[:, 0], last_index=row, n_steps=n_steps, step_length=step_length)
            targets[j] = data[row - 1, 1]
        yield samples, targets
        
batch_size = 32


# Position of second (of 16) earthquake. Used to have a clean split
# between train and validation
second_earthquake = 50085877
float_data[second_earthquake, 1]

# Initialize generators
train_gen = generator(float_data[0:585568143], batch_size=batch_size) # Use this for better score
# train_gen = generator(float_data, batch_size=batch_size, min_index=second_earthquake + 1)
valid_gen = generator(float_data[585568144:629145479], batch_size=batch_size, max_index=second_earthquake)


from numpy.random import seed
from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import Dense, CuDNNGRU
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint,EarlyStopping

ensemble_size = 2
np_seeds = np.random.randint(100000, size=ensemble_size)
tf_seeds = np.random.randint(100000, size=ensemble_size)

models = []

for i in range(ensemble_size):
    # Fix seeds
    seed(np_seeds[i])
    set_random_seed(tf_seeds[i])
    print("Training model", i+1)
    
    model_name = "model" + str(i+1) + ".hdf5"
    cb = [ModelCheckpoint(model_name, save_best_only=True, period=3), 
         EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)]
    
    
    model = Sequential()
    model.add(Dense(15, activation='relu',input_shape=(None, n_features)))
    model.add(CuDNNGRU(10))
    model.add(Dense(7, activation='relu'))
    model.add(Dense(1))
    model.summary()
    
    # Compile and fit model
    model.compile(optimizer=adam(lr=0.001,decay =0.0005,amsgrad=True), loss="mae")
    
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=1000,
                                  epochs=5,
                                  verbose=1,
                                  callbacks=cb,
                                  validation_data=valid_gen,
                                  validation_steps=200)
                                  
    models.append(model)




# Load submission file
submission = pd.read_csv('./data/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})

# Load each test data, create the feature matrix, get numeric prediction
for i, seg_id in enumerate(tqdm(submission.index)):
  #  print(i)
    seg = pd.read_csv('./data/test/' + seg_id + '.csv')
    x = seg['acoustic_data'].values
    sample = create_X(x)
    
    prediction = 0
    for model in models:
        prediction += model.predict(np.expand_dims(sample, 0))
    
    prediction = prediction / ensemble_size
    submission.time_to_failure[i] = prediction

submission.head()

# Save
submission.to_csv('ensemble20.csv')

