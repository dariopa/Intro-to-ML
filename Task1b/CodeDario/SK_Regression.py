import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from utils_Models import Regression
from utils_OutputFormat import PrintOutput_1b
 
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

(X_train, y_train) = shuffle(X_train, y_train)
               
#########################################################
# TRAIN DATA

# weights = Regression.RidgeRegression(X_train, y_train)
weights = Regression.LassoRegression(X_train, y_train)
# weights = Regression.LinRegression(X_train, y_train)
    
PrintOutput_1b(weights)
print(weights)