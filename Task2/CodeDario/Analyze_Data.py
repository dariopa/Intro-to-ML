import os
import numpy as np
import matplotlib.pyplot as plt
################# HARDCODED INPUTS ######################
CallFolder = '/home/dario/Desktop/Intro-to-ML/Task2/Raw_Data'

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
# PLOTS
plt.figure(1)
for i in range(len(X_train)):
    if y_train[i] == 0:
        color = 'r'
    elif y_train[i] == 1:
        color = 'g'
    elif y_train[i] == 2:
        color = 'b'
    plt.plot(X_train[i], color)
plt.axis([0, 16, 0, 60])
plt.savefig('data_distribution.jpg')

plt.figure(2)
for i in range(len(X_train)):
    if y_train[i] == 0:
        plt.plot(X_train[i], 'r')
plt.axis([0, 16, 0, 60])
plt.savefig('data_distribution_class_0.jpg')

plt.figure(3)
for i in range(len(X_train)):
    if y_train[i] == 1:
        plt.plot(X_train[i], 'g')
plt.axis([0, 16, 0, 60])
plt.savefig('data_distribution_class_1.jpg')

plt.figure(4)
for i in range(len(X_train)):
    if y_train[i] == 2:
        plt.plot(X_train[i], 'b')
plt.axis([0, 16, 0, 60])
plt.savefig('data_distribution_class_2.jpg')

plt.figure(4)
for i in range(len(X_test)):
    plt.plot(X_test[i], 'b')
plt.axis([0, 16, 0, 60])
plt.savefig('data_distribution_test.jpg')
