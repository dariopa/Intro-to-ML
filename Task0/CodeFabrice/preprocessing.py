import os
import numpy as np

class loadfiles:
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
