import os
import numpy as np

################# HARDCODED INPUTS ######################
CallFolder = '/home/dario/Desktop/Intro-to-ML/Task1b/Raw_Data'

#########################################################
# IMPORT, PREPROCESS AND STORE TRAINING DATA AS NUMPY ARRAY
DataTrain = np.genfromtxt(os.path.join(CallFolder, 'train.csv'), delimiter=',')
DataTrain = np.delete(DataTrain, 0, 0)
DataTrain = np.delete(DataTrain, 0, 1)

X_train = DataTrain[:, 1:]
y_train = DataTrain[:, 0]

#########################################################
# PREPROCESSING DATA
def no_cos(X_data):
    X_data[:, 15:20] = 0.
    return X_data
#########################################################
# GENERATE FULL DATA
row, col = X_train.shape
X_train_full = np.full((row, 21), 0., dtype=np.float16)
X_train_full[:, 0:5] = X_train
X_train_full[:, 5:10] = X_train ** 2
X_train_full[:, 10:15] = np.exp(X_train)
X_train_full[:, 15:20] = np.cos(X_train)
X_train_full[:, 20:21] = 1.

X_train_full = no_cos(X_train_full)
print(X_train_full)
    
#########################################################
# STORE DATA AS NUMPY ARRAY
print('X_train:   ', X_train_full.shape, end=' ||  ')
print('y_train:   ', y_train.shape)
np.save('X_train.npy', X_train_full)
np.save('y_train.npy', y_train)
