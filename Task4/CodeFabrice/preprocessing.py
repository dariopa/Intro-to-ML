import os
import numpy as np
import pandas as pd

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

class loadfiles3: # Class which gets the files and saves them in the right variables
    def __init__(self, path):
        self.path = path
   
    def loadX_train(self):
        DataTrain = pd.read_hdf(os.path.join(self.path, "train.h5"), "train")
        DataTrain = DataTrain.values
        # DataTrain = np.delete(DataTrain, 0, 0)
        # DataTrain = np.delete(DataTrain, 0, 1)
        X_train = DataTrain[:, 1:]
        print(X_train)
        return X_train

    def loady_train(self):
        DataTrain = pd.read_hdf(os.path.join(self.path, "train.h5"), "train")
        DataTrain = DataTrain.values
        # DataTrain = np.delete(DataTrain, 0, 0)
        # DataTrain = np.delete(DataTrain, 0, 1)
        y_train = DataTrain[:, 0]
        return y_train

    def loadX_test(self):
        X_test = pd.read_hdf(os.path.join(self.path, "test.h5"), "test")
        X_test = X_test.values
        # X_test = np.delete(X_test, 0, 0)
        # X_test = np.delete(X_test, 0, 1)        
        return X_test

class loadfiles3: # Class which gets the files and saves them in the right variables
    def __init__(self, path):
        self.path = path
   
    def loadX_train_labeled(self):
        DataTrain = pd.read_hdf(os.path.join(self.path, "train_labeled.h5"), "train")
        DataTrain = DataTrain.values
        # DataTrain = np.delete(DataTrain, 0, 0)
        # DataTrain = np.delete(DataTrain, 0, 1)
        X_train_labeled = DataTrain[:, 1:]
        print(X_train_labeled)
        return X_train_labeled

    def loady_train(self):
        DataTrain = pd.read_hdf(os.path.join(self.path, "train_labeled.h5"), "train")
        DataTrain = DataTrain.values
        # DataTrain = np.delete(DataTrain, 0, 0)
        # DataTrain = np.delete(DataTrain, 0, 1)
        y_train = DataTrain[:, 0]
        return y_train

    def loadX_train_labeled(self):
        DataTrain = pd.read_hdf(os.path.join(self.path, "train_unlabeled.h5"), "train")
        DataTrain = DataTrain.values
        # DataTrain = np.delete(DataTrain, 0, 0)
        # DataTrain = np.delete(DataTrain, 0, 1)
        X_train_unlabeled = DataTrain[:, 1:]
        print(X_train_unlabeled)
        return X_train_unlabeled

    def loadX_test(self):
        X_test = pd.read_hdf(os.path.join(self.path, "test.h5"), "test")
        X_test = X_test.values
        # X_test = np.delete(X_test, 0, 0)
        # X_test = np.delete(X_test, 0, 1)        
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
            X_new[i,0:3] = X[i,0:3]*0 # Linear 
            X_new[i,5:10] = np.square(X[i,0:5]) # Quadratic
            # X_new[i,5:10] = np.square(X[i,0:5])*0 # Quadratic
            X_new[i,10:15] = np.exp(X[i,0:5]) # Exponential
            # X_new[i,10:15] = np.exp(X[i,0:5])*0 # Exponential
            X_new[i,15:20] = np.cos(X[i,0:5]) # Cosine
            X_new[i,15:20] = np.cos(X[i,0:5])*0 # Cosine
            X_new[i,20] = 1
        return X_new

class downsampling: 
    def __init__(self, save_path=None):
        self.save_path = save_path

    def transform(self, X, y):
        shape = X.shape
        NSamples = shape[0]
        NFeatures = shape[1]
        NClasses = 5
        NSamplesPerClass = np.zeros([NClasses,1])
        for i in range(0, NSamples):
            if y[i] == 0:
                NSamplesPerClass[0] += 1
            elif y[i] == 1:
                NSamplesPerClass[1] += 1
            elif y[i] == 2:
                NSamplesPerClass[2] += 1
            elif y[i] == 3:
                NSamplesPerClass[3] += 1
            elif y[i] == 4:
                NSamplesPerClass[4] += 1
        print(NSamplesPerClass)
        NSamplesMin = int(np.amin(NSamplesPerClass))
        print(NSamplesMin)
        NSamplesPerClass = np.zeros([NClasses, 1])
        X_new = np.zeros([NSamplesMin*NClasses, NFeatures])
        y_new = np.zeros([NSamplesMin*NClasses, 1])
        k = 0
        for i in range(0, NSamples):
            if y[i] == 0 and NSamplesPerClass[0] < NSamplesMin:
                X_new[k,:] = X[i]
                y_new[k] = y[i]
                NSamplesPerClass[0] += 1
                k += 1
            elif y[i] == 1 and NSamplesPerClass[1] < NSamplesMin:
                X_new[k,:] = X[i]
                y_new[k] = y[i]
                NSamplesPerClass[1] += 1
                k += 1 
            elif y[i] == 2 and NSamplesPerClass[2] < NSamplesMin:
                X_new[k,:] = X[i]
                y_new[k] = y[i]
                NSamplesPerClass[2] += 1
                k += 1
            elif y[i] == 3 and NSamplesPerClass[3] < NSamplesMin:
                X_new[k,:] = X[i]
                y_new[k] = y[i]
                NSamplesPerClass[3] += 1
                k += 1
            elif y[i] == 4 and NSamplesPerClass[4] < NSamplesMin:
                X_new[k,:] = X[i]
                y_new[k] = y[i]
                NSamplesPerClass[4] += 1
                k += 1
        print(NSamplesPerClass)
        print(X_new.shape)
        print(y_new.shape)
        y_new = np.squeeze(y_new)
        return (X_new, y_new)