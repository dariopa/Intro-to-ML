import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

class score: 
    def __init__(self, save_path=None):
        self.save_path = save_path

    def RMSE(self, y, y_pred):
        RMSE = mean_squared_error(y, y_pred)**0.5
        return RMSE

    def Accuracy(self, y, y_pred):
        Accuracy = accuracy_score(y, y_pred)
        return Accuracy