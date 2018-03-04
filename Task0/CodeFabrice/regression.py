import os
from sklearn import linear_model

class LinearRegression:
    def __init__(self, save_path=None):
        self.save_path = save_path
    
    def fit(self, X, y):
        self.regr = linear_model.LinearRegression()
        self.regr.fit(X,y)
        return self

    def predict(self, X):
        y_pred = self.regr.predict(X)
        return y_pred
