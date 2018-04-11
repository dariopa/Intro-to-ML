import os
import numpy as np
import utils_preprocess as PrePro
################# HARDCODED INPUTS ######################
CallFolder = '/home/dario/Desktop/Intro-to-ML/Task2/Raw_Data'

# Preprocess data?
pre = False

#########################################################
# IMPORT, PREPROCESS AND STORE TRAINING DATA AS NUMPY ARRAY
DataTrain = np.genfromtxt(os.path.join(CallFolder, 'train.csv'), delimiter=',')
DataTrain = np.delete(DataTrain, 0, 0)
DataTrain = np.delete(DataTrain, 0, 1)

X_train = DataTrain[:, 1:]
y_train = DataTrain[:, 0]

#########################################################
# IMPORT, PREPROCESS AND STORE TEST DATA AS NUMPY ARRAY
X_test = np.genfromtxt(os.path.join(CallFolder, 'test.csv'), delimiter=',')
X_test = np.delete(X_test, 0, 0)
X_test = np.delete(X_test, 0, 1)

#########################################################
# PREPROCESS DATA
def retain(X_data):
    # X_data = np.delete(X_data, [4,5,6,7,8], 1)
    X_data = np.delete(X_data, [3,4,5,6,15,16], 1)
    return X_data

if pre == True:
    X_train = retain(X_train)
    X_test = retain(X_test)

#########################################################
# STORE DATA AS NUMPY ARRAY
print('X_train:   ', X_train.shape, end=' ||  ')
print('y_train:   ', y_train.shape)
print('X_test:    ', X_test.shape)
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
