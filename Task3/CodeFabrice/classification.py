import os
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

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

class MLPClassification: # Simple linear regression
    def __init__(self, save_path=None):
        self.save_path = save_path
    
    def fit(self, X, y):
        self.clas = MLPClassifier(hidden_layer_sizes=(30, 30, 30), activation='relu', solver='sgd', 
        alpha=2, batch_size=64, learning_rate='adaptive', learning_rate_init=0.0001, 
        max_iter=20000, shuffle=True, random_state=None, tol=0.0001)
        self.clas.fit(X,y)
        return self

    def predict(self, X):
        y_pred = self.clas.predict(X)
        return y_pred