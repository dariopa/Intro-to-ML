import os
import numpy as np
import time 
import datetime
import pandas as pd

class savedata: 
    def __init__(self, savepath, datapath):
        self.savepath = savepath # Path where file should be saved
        self.datapath = datapath # Path where to get the test file for the id's

    def saveprediction(self, y_pred):
        X_test = np.genfromtxt(os.path.join(self.datapath, 'test.csv'), delimiter=',')
        X_test = np.delete(X_test, 0, 0)
        X_test_id = X_test[:,[0]]  # Extract id's
        data = np.column_stack((X_test_id, y_pred)) # stack id's and prediction       
        np.savetxt(os.path.join(self.savepath, 'prediction.csv'), data, fmt='%.18e', delimiter=',', newline='\n', header='Id,y', comments='') # add header and save file

class savetask1a: 
    def __init__(self, savepath, datapath):
        self.savepath = savepath # Path where file should be saved
        self.datapath = datapath # Path where to get the test file for the id's

    def saveprediction(self, scores):
        # X_test = np.genfromtxt(os.path.join(self.datapath, 'test.csv'), delimiter=',')
        # X_test = np.delete(X_test, 0, 0)
        # X_test_id = X_test[:,[0]]  # Extract id's
        # data = np.column_stack((X_test_id, y_pred)) # stack id's and prediction      
        data = scores 
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
        filename = st + '_prediction1a.csv'
        print(filename)
        np.savetxt(os.path.join(self.savepath, filename), data, fmt='%5.15f', delimiter=' ', newline='\n', header='', comments='') # save file

class savetask1b: 
    def __init__(self, savepath, datapath):
        self.savepath = savepath # Path where file should be saved
        self.datapath = datapath # Path where to get the test file for the id's

    def saveprediction(self, weights):
        # X_test = np.genfromtxt(os.path.join(self.datapath, 'test.csv'), delimiter=',')
        # X_test = np.delete(X_test, 0, 0)
        # X_test_id = X_test[:,[0]]  # Extract id's
        # data = np.column_stack((X_test_id, y_pred)) # stack id's and prediction      
        data = weights 
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
        filename = st + '_prediction1b.csv'
        print(filename)
        np.savetxt(os.path.join(self.savepath, filename), data, fmt='%5.15f', delimiter=' ', newline='\n', header='', comments='') 
    
class savetask2: 
    def __init__(self, savepath, datapath):
        self.savepath = savepath # Path where file should be saved
        self.datapath = datapath # Path where to get the test file for the id's

    def saveprediction(self, y_pred):
        X_test = np.genfromtxt(os.path.join(self.datapath, 'test.csv'), delimiter=',')
        X_test = np.delete(X_test, 0, 0)
        X_test_id = X_test[:,[0]]  # Extract id's
        data = np.column_stack((X_test_id, y_pred)) # stack id's and prediction      
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
        filename = st + '_prediction2.csv'
        print(filename)
        np.savetxt(os.path.join(self.savepath, filename), data, fmt='%2.15f', delimiter=',', newline='\n', header='Id,y', comments='') # add header and save file

class savetask3: 
    def __init__(self, savepath, datapath):
        self.savepath = savepath # Path where file should be saved
        self.datapath = datapath # Path where to get the test file for the id's

    def saveprediction(self, y_pred):
        X_test = pd.read_hdf(os.path.join(self.datapath, "test.h5"), "test")
        X_test_id = X_test.index.values
        print(X_test_id.shape)
        print(y_pred.shape)
        data = np.column_stack((X_test_id, y_pred)) # stack id's and prediction      
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
        filename = st + '_prediction3.csv'
        print(filename)
        np.savetxt(os.path.join(self.savepath, filename), data, fmt='%2.2f', delimiter=',', newline='\n', header='Id,y', comments='') # add header and save file
        
class savetask4: 
    def __init__(self, savepath, datapath):
        self.savepath = savepath # Path where file should be saved
        self.datapath = datapath # Path where to get the test file for the id's

    def saveprediction(self, y_pred):
        X_test = pd.read_hdf(os.path.join(self.datapath, "test.h5"), "test")
        X_test_id = X_test.index.values
        print(X_test_id.shape)
        print(y_pred.shape)
        data = np.column_stack((X_test_id, y_pred)) # stack id's and prediction      
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
        filename = st + '_prediction4.csv'
        print(filename)
        np.savetxt(os.path.join(self.savepath, filename), data, fmt='%2.2f', delimiter=',', newline='\n', header='Id,y', comments='') # add header and save file
        