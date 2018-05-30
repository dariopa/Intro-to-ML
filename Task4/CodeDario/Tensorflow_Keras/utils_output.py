import os
import numpy as np

def PrintOutput(predictions, sample_number, filename="y_test.csv"):

    with open(filename, 'w+') as fp:
        fp.write("Id,y\n")

        for i in range(predictions.shape[0]):
            line = str(sample_number + i) + "," + str(predictions[i]) + "\n"
            fp.write(line)