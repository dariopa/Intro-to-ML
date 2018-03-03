import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor as DTR


X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
print('Shape of X_train:', X_train.shape)
print('Shape of y_train:', y_train.shape)
print('Shape of X_test:', X_test.shape)

model = DTR(max_depth=3)
model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)
