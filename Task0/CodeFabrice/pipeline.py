import os
from sklearn.pipeline import Pipeline
import preprocessing

dataloader = preprocessing.loadfiles('E:\Polybox\ETH\2. Semester FS18\Introduction to Machine Learning\Projects\git\Intro-to-ML\Task0\Raw_Data') # Define datafolder
X_train = dataloader.loadX_train()
y_train = dataloader.loady_train()
X_test = dataloader.loadX_test()

