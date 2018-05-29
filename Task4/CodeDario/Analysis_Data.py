import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

random_seed = 123
np.random.seed(random_seed)

CallFolder = '../Raw_Data/'

#########################################################
# LOAD DATA!
DataTrain = np.array(pd.read_hdf(CallFolder + "train_labeled.h5", "train"))
X_train = DataTrain[:, 1:]
y_train = DataTrain[:, 0]

print('Randomly selected data')
print('X_train:   ', X_train.shape, end=' ||  ')
print('y_train:   ', y_train.shape)

class_zero = 0
class_one = 0
class_two = 0
class_three = 0
class_four = 0
class_five = 0
class_six = 0
class_seven = 0
class_eight = 0
class_nine = 0

for i in range(len(y_train)):
    if y_train[i] == 0:
        class_zero = 1 + class_zero
    if y_train[i] == 1:
        class_one = 1 + class_one
    if y_train[i] == 2:
        class_two = 1 + class_two
    if y_train[i] == 3:
        class_three = 1 + class_three
    if y_train[i] == 4:
        class_four = 1 + class_four
    if y_train[i] == 5:
        class_five = 1 + class_five
    if y_train[i] == 6:
        class_six = 1 + class_six
    if y_train[i] == 7:
        class_seven = 1 + class_seven
    if y_train[i] == 8:
        class_eight = 1 + class_eight
    if y_train[i] == 9:
        class_nine = 1 + class_nine

print('Class imbalance:')
print('Class 0:   ', class_zero)
print('Class 1:   ', class_one)
print('Class 2:   ', class_two)
print('Class 3:   ', class_three)
print('Class 4:   ', class_four)
print('Class 5:   ', class_five)
print('Class 6:   ', class_six)
print('Class 7:   ', class_seven)
print('Class 8:   ', class_eight)
print('Class 9:   ', class_nine)

#########################################################
print(X_train)
print('\n', X_test)

# PLOTS
for q in range(10):
    plt.figure(q)
    for i in range(len(X_train)):
        if y_train[i] == q:
            plt.plot(X_train[i])
    plt.savefig('data_distribution_class_' + str(q) + '.jpg')
