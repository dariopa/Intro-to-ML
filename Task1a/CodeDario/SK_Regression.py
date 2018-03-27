import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from utils_Models import RidgeRegression
from sklearn.model_selection import KFold
from utils_OutputFormat import PrintOutput_1a
from sklearn.metrics import mean_squared_error
from math import sqrt

#########################################################
# HARDCODED PARAM
p_lambda = np.array([[0.1, 1, 10, 100, 1000]])
_, col = p_lambda.shape
avg_RMSE = np.full((col, 1), 0., dtype=np.float32)
#########################################################
# LOAD DATA

X = np.load('X_train.npy')
y = np.load('y_train.npy')
               
#########################################################
# TRAIN DATA

kf = KFold(n_splits=10, random_state=None, shuffle=False)
for i in range(len(avg_RMSE)):
    RMSE_tot = 0.
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_pred = RidgeRegression(p_lambda[0, i], X_train, y_train, X_test)
        
        RMSE = sqrt(mean_squared_error(y_test, y_pred)) / 10
        RMSE_tot += RMSE
    avg_RMSE[i, :] = RMSE_tot
#########################################################
# EVALUATION

PrintOutput_1a(avg_RMSE)