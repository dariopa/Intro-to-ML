import os
import numpy as np

class loadfiles: # Class which gets the files and saves them in the right variables
    def __init__(self, path):
        self.path = path
   
    def loadX_train(self):
        DataTrain = np.genfromtxt(os.path.join(self.path, 'train.csv'), delimiter=',')
        DataTrain = np.delete(DataTrain, 0, 0)
        DataTrain = np.delete(DataTrain, 0, 1)
        X_train = DataTrain[:, 1:]
        return X_train

    def loady_train(self):
        DataTrain = np.genfromtxt(os.path.join(self.path, 'train.csv'), delimiter=',')
        DataTrain = np.delete(DataTrain, 0, 0)
        DataTrain = np.delete(DataTrain, 0, 1)
        y_train = DataTrain[:, 0]
        return y_train

    def loadX_test(self):
        X_test = np.genfromtxt(os.path.join(self.path, 'test.csv'), delimiter=',')
        X_test = np.delete(X_test, 0, 0)
        X_test = np.delete(X_test, 0, 1)
        return X_test

class task1btransformation:
    def __init__(self, save_path=None):
        self.save_path = save_path
    
    def transform(self, X):
        shape = X.shape
        print(shape[0])
        X_new = np.empty([shape[0], 21])
        for i in range(0, shape[0]):
            X_new[i,0:5] = X[i,0:5] # Linear 
            X_new[i,5:10] = np.square(X[i,0:5]) # Quadratic
            X_new[i,10:15] = np.exp(X[i,0:5]) # Exponential
            X_new[i,15:20] = np.cos(X[i,0:5]) # Cosine
            X_new[i,20] = 1
        return X_new