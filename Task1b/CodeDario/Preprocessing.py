from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
import numpy as np

def minmax(X_data):
    scaler = MinMaxScaler(feature_range=(0,1))
    X_data_centered = scaler.fit_transform(X_data)
    return X_data_centered


def GenerateData_all(X_data):
    row, col = X_data.shape
    X_train = np.full((row, 21), 1, dtype=np.float32)
    X_train[:, 0:5] = X_data
    X_train[:, 5:10] = X_data ** 2
    X_train[:, 10:15] = np.exp(X_data)
    X_train[:, 15:20] = np.cos(X_data)
    return X_train

def GenerateData_no_cos_exp(X_data):
    row, col = X_data.shape
    X_train = np.full((row, 10), 1, dtype=np.float32)
    X_train[:, 0:5] = X_data
    X_train[:, 5:10] = X_data ** 2
    return X_train

def GenerateData_no_cos(X_data):
    row, col = X_data.shape
    X_train = np.full((row, 15), 1, dtype=np.float32)
    X_train[:, 0:5] = X_data
    X_train[:, 5:10] = X_data ** 2
    X_train[:, 10:15] = np.exp(X_data)
    return X_train