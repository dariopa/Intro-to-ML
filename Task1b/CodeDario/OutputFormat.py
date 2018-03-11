import os
import numpy as np

def PrintOutput(predictions, filename="y_test.csv"):

    with open(filename, 'w+') as fp:
        fp.write("Id,y\n")

        for i in range(predictions.shape[0]):
            line = str(10000 + i) + "," + str(predictions[i]) + "\n"
            fp.write(line)

def PrintOutput_1b(weights, filename="y_test.csv"):

    with open(filename, 'w+') as fp:

        for i in range(len(weights)):
            line = str(weights[i]) + "\n"
            fp.write(line)
        
        if i < 20:
            print(i)
            for j in range(20-i):
                line = str(0.0) + "\n"
                fp.write(line)
