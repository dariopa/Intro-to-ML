import numpy as np
from sklearn.neural_network import MLPRegressor as MLP
from sklearn.linear_model import Ridge as RDG
from sklearn.linear_model import LinearRegression as LR
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error


# Decide whether self-evaluation or final submission
#########################################################
final_submission = False

Train_split = 8./10
#########################################################

def LinRegression(X_data, y_data, X_test):
    model =LR(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
    model.fit(X_data, y_data)

    y = model.predict(X_test)
    return y

def RidgeRegression(X_data, y_data, X_test):
    model =RDG(alpha=0.00000000001,
               copy_X=True,
               fit_intercept=True,
               max_iter=None,
               solver='auto',
               tol=0.00000000000001)
    model.fit(X_data, y_data)

    y = model.predict(X_test)
    return y

def PerceptronRegression(X_data, y_data, X_test):
    model = MLP(hidden_layer_sizes=(5,),
                activation='relu',
                solver='adam',
                learning_rate_init=0.001,
                tol=0.000001,
                max_iter=20000,
                shuffle=True,
                verbose=0)
    model.fit(X_data, y_data)

    y = model.predict(X_test)
    return y

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
samples = len(X_train)
X_test = np.load("X_test.npy")
print('Shape of X_train:', X_train.shape)
print('Shape of y_train:', y_train.shape)
print('Shape of X_test:', X_test.shape)

(X_train, y_train) = shuffle(X_train, y_train)

if final_submission == True:
    # y_test_pred = PerceptronRegression(X_train, y_train, X_test)
    y_test_pred = RidgeRegression(X_train, y_train, X_test)
    # y_test_pred = LinRegression(X_train, y_train, X_test)
    np.savetxt("y_test.csv", y_test_pred, delimiter=",")

else:
    X_train_selfeval = X_train[0:int(Train_split * samples), :]
    y_train_selfeval = y_train[0:int(Train_split * samples)]
    X_test_selfeval = X_train[int(Train_split * samples):samples, :]
    y_test_selfeval = y_train[int(Train_split * samples):samples]

    # y_test_pred = PerceptronRegression(X_train_selfeval, y_train_selfeval, X_test_selfeval)
    y_test_pred = RidgeRegression(X_train_selfeval, y_train_selfeval, X_test_selfeval)
    # y_test_pred = LinRegression(X_train_selfeval, y_train_selfeval, X_test_selfeval)

    RMSE = mean_squared_error(y_test_selfeval, y_test_pred)**0.5
    print("\nMean Squared Error:    ", RMSE)
