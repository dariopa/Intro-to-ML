import os
import numpy as np
import string

def PrintOutput_1a(avg_RMSE, filename="avg_RMSE.csv"):

    with open(filename, 'w+') as fp:

        for i in range(len(avg_RMSE)):
            val = str(avg_RMSE[i])[1:-1] # get rid of first and last element!
            print(val)
            line = str(val) + "\n"
            fp.write(line)

def PrintOutput_1b(weights, filename="y_test.csv"):

    with open(filename, 'w+') as fp:

        for i in range(len(weights)):
            line = str(weights[i]) + "\n"
            fp.write(line)
