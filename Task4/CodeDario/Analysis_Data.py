import os
import numpy as np
import pandas as pd
import random

random_seed = 123
np.random.seed(random_seed)

CallFolder = '../Raw_Data/'

#########################################################
# LOAD DATA!
DataTrain = np.array(pd.read_hdf(CallFolder + "train.h5", "train"))
X_train = DataTrain[:, 1:]
y_train = DataTrain[:, 0]

print('Randomly selected data')
print('X_train:   ', X_train.shape, end=' ||  ')
print('y_train:   ', y_train.shape)

class_zero = 0
class_one = 1
class_two = 2
class_three = 3
class_four = 4

for i in range(len(y_train)):
    if y_train[i] == 0:
        class_zero = 1 + class_zero
    if y_train[i] == 1:
        class_one = 1 + class_one
    if y_train[i] == 2:
        class_two = 1 + class_two
    if y_train[i] == 3:
        class_three = 1 + class_three
    if y_train[i] == 2:
        class_four = 1 + class_four
print('Class imbalance:')
print('Class 0:   ', class_zero)
print('Class 1:   ', class_one)
print('Class 2:   ', class_two)
print('Class 3:   ', class_three)
print('Class 4:   ', class_four)

Labels = [class_zero, class_one, class_two, class_three, class_four]
random_samples = np.min(Labels)
