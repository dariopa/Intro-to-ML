import os
import numpy as np

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
        np.savetxt(os.path.join(self.savepath, 'prediction1a.csv'), data, fmt='%.18e', delimiter=' ', newline='\n', header='', comments='') # add header and save file
    