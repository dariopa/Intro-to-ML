import os
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

class SVClassification: # Simple linear regression
    def __init__(self, save_path=None):
        self.save_path = save_path
    
    def fit(self, X, y):
        self.clas = SVC()
        self.clas.fit(X,y)
        return self

    def predict(self, X):
        y_pred = self.clas.predict(X)
        return y_pred

class MultiClassSVC: # Simple linear regression
    def __init__(self, save_path=None):
        self.save_path = save_path
    
    def fit(self, X, y):
        self.clas = LinearSVC(multi_class="crammer_singer")
        self.clas.fit(X,y)
        return self

    def predict(self, X):
        y_pred = self.clas.predict(X)
        return y_pred