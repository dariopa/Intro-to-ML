import os
import numpy as np

################# HARDCODED INPUTS ######################
CallFolder = '/home/dario/Desktop/Intro-to-ML/Project_0/Raw_Data'


#########################################################
# FUNCTIONS FOR PREPROCESSING
def centering(X_data):
    mean_vals = np.mean(X_data, axis=0)
    std_val = np.std(X_data)
    X_data_centered = (X_data - mean_vals)/std_val
    del X_data
    return X_data_centered

#########################################################
# IMPORT, PREPROCESS AND STORE TRAINING DATA AS NUMPY ARRAY
DataTrain = np.genfromtxt(os.path.join(CallFolder, 'train.csv'), delimiter=',')
DataTrain = np.delete(DataTrain, 0, 0)
DataTrain = np.delete(DataTrain, 0, 1)
print(DataTrain.shape)


X_train = centering(DataTrain[:, 1:])
print('X_train: \n', X_train)

y_train = DataTrain[:, 0]
print('y_train: \n', y_train)

np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)

#########################################################
# IMPORT, PREPROCESS AND STORE TEST DATA AS NUMPY ARRAY
X_test = np.genfromtxt(os.path.join(CallFolder, 'test.csv'), delimiter=',')
X_test = np.delete(X_test, 0, 0)
X_test = np.delete(X_test, 0, 1)
print(X_test.shape)

X_test = centering(X_test)
print('X_test: \n', X_test)

np.save('X_test.npy', X_test)

