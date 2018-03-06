import os
from sklearn.metrics import mean_squared_error

class score: 
    def __init__(self, save_path=None):
        self.save_path = save_path

    def RMSE(self, y, y_pred):
        RMSE = mean_squared_error(y, y_pred)**0.5
        return RMSE