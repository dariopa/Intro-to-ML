import os
import numpy as np
import Preprocessing as PrePro
################# HARDCODED INPUTS ######################
CallFolder = '/home/dario/Desktop/Intro-to-ML/Task1b/Raw_Data'

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
# PREPROCESS DATA
X_train = PrePro.GenerateData_all(X_train)

if pre == True:
    X_train = PrePro.minmax(X_train)
    
#########################################################
# STORE DATA AS NUMPY ARRAY
print('X_train:   ', X_train.shape, end=' ||  ')
print('y_train:   ', y_train.shape)
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
