import os
import numpy as np

class savedata: 
    def __init__(self, savepath, datapath):
        self.savepath = savepath
        self.datapath = datapath

    def saveprediction(self, y_pred):
        X_test = np.genfromtxt(os.path.join(self.datapath, 'test.csv'), delimiter=',')
        X_test = np.delete(X_test, 0, 0)
        X_test_id = X_test[:,[0]]
        print(X_test_id)
        data = np.column_stack((X_test_id, y_pred))
        print(data)
        header = np.array(['Id', 'y'])
        # prediction = np.append(header,)        
        np.savetxt(os.path.join(self.savepath, 'prediction.csv'), data, fmt='%.18e', delimiter=',', newline='\n', header='Id,y', comments='')
    