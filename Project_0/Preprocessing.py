import os
import numpy as np

################# HARDCODED INPUTS ######################
CallFolder = 'Raw_Data'

#########################################################
# IMPORT, PREPROCESS AND STORE TRAINING DATA AS NUMPY ARRAY
DataTrain = np.genfromtxt(os.path.join(CallFolder, 'train.csv'), delimiter=',')
DataTrain = np.delete(DataTrain, 0, 0)
DataTrain = np.delete(DataTrain, 0, 1)
print(DataTrain.shape)


X_train = DataTrain[:, 1:]
print('X_train: \n', X_train)

y_train = DataTrain[:, 0]
print('y_train: \n', y_train)

np.save(CallFolder +'/X_train.npy', X_train)
np.save(CallFolder +'/y_train.npy', y_train)

#########################################################
# IMPORT, PREPROCESS AND STORE TEST DATA AS NUMPY ARRAY
X_test = np.genfromtxt(os.path.join(CallFolder, 'test.csv'), delimiter=',')
X_test = np.delete(X_test, 0, 0)
X_test = np.delete(X_test, 0, 1)
print(X_test.shape)

print('X_test: \n', X_test)

np.save(CallFolder +'/X_test.npy', X_test)

#########################################################
# FUNCTION FOR PREPROCESSING
