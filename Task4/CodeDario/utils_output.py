import os
import numpy as np

def PrintOutput(predictions, filename="y_test.csv"):

    with open(filename, 'w+') as fp:
        fp.write("Id,y\n")

        for i in range(predictions.shape[0]):
            line = str(45324 + i) + "," + str(predictions[i]) + "\n"
            fp.write(line)