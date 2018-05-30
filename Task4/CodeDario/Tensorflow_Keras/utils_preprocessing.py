import os
import numpy as np

def centering(X_train, X_test):
    mean_vals = np.mean(X_train, axis=0)
    std_val = np.std(X_train)
    
    X_train = (X_train - mean_vals)/std_val
    X_test = (X_test - mean_vals)/std_val

    return X_train, X_test
