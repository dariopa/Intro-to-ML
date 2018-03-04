import os
import numpy as np
import Preprocessing as PrePro
################# HARDCODED INPUTS ######################
CallFolder = '/home/dario/Desktop/Intro-to-ML/Task0/Raw_Data'

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
if pre == True:
    X_train = PrePro.minmax(X_train)
    X_test = PrePro.minmax(X_test)

#########################################################
# STORE DATA AS NUMPY ARRAY
print('X_train:   ', X_train.shape, end=' ||  ')
print('y_train:   ', y_train.shape)
print('X_test:    ', X_test.shape)
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
