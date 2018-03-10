import os
from sklearn import linear_model

class LinearRegression: # Simple linear regression
    def __init__(self, save_path=None):
        self.save_path = save_path
    
    def fit(self, X, y):
        self.regr = linear_model.LinearRegression()
        self.regr.fit(X,y)
        return self

    def predict(self, X):
        y_pred = self.regr.predict(X)
        return y_pred

    def getcoeff(self):
        w = self.regr.coef_
        return w

class RidgeRegression:
    def __init__(self, alpha, save_path=None):
        self.save_path = save_path
        self.alpha = alpha
    
    def fit(self, X, y):
        self.regr = linear_model.Ridge(alpha=self.alpha)
        self.regr.fit(X,y)
        return self

    def predict(self, X):
        y_pred = self.regr.predict(X)
        return y_pred
    
    def getcoeff(self):
        w = self.regr.coef_
        return w

class LassoRegression:
    def __init__(self, alpha, save_path=None):
        self.save_path = save_path
        self.alpha = alpha
    
    def fit(self, X, y):
        self.regr = linear_model.Lasso(alpha=self.alpha)
        self.regr.fit(X,y)
        return self

    def predict(self, X):
        y_pred = self.regr.predict(X)
        return y_pred
    
    def getcoeff(self):
        w = self.regr.coef_
        return w