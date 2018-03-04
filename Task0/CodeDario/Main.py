import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from Models import Regression
from OutputFormat import PrintOutput

# Decide whether self-evaluation or final submission
#########################################################
final_submission = True

Train_split = 8./10
#########################################################
# LOAD AND SHUFFLE DATA!
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
print('Shape of X_train:', X_train.shape)
print('Shape of y_train:', y_train.shape)
print('Shape of X_test:', X_test.shape)

(X_train, y_train) = shuffle(X_train, y_train)
               
#########################################################
# TRAIN DATA
if final_submission == True:
    # y_test_pred = Regression.PerceptronRegression(X_train, y_train, X_test)
    y_test_pred = Regression.RidgeRegression(X_train, y_train, X_test)
    # y_test_pred = Regression.LinRegression(X_train, y_train, X_test)
    
    # STORE RESULTS IN CSV FILE!
    PrintOutput(y_test_pred, filename='y_test.csv')
    print('\nJob Done!')
else:
    samples = len(X_train)
    X_train_selfeval = X_train[0:int(Train_split * samples), :]
    y_train_selfeval = y_train[0:int(Train_split * samples)]
    X_test_selfeval = X_train[int(Train_split * samples):samples, :]
    y_test_selfeval = y_train[int(Train_split * samples):samples]

    # y_test_pred = Regression.PerceptronRegression(X_train_selfeval, y_train_selfeval, X_test_selfeval)
    y_test_pred = Regression.RidgeRegression(X_train_selfeval, y_train_selfeval, X_test_selfeval)
    # y_test_pred = Regression.LinRegression(X_train_selfeval, y_train_selfeval, X_test_selfeval)
    
    # PREDICT SELF-EVALUATION!
    RMSE = mean_squared_error(y_test_selfeval, y_test_pred)**0.5
    print("\nMean Squared Error:    ", RMSE)
