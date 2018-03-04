import os
from sklearn.pipeline import Pipeline
import preprocessing

dataloader = preprocessing.loadfiles('C:\\Users\\fabri\\git\\Intro-to-ML\\Task0\\Raw_Data') # Define datafolder - HomePC
# dataloader = preprocessing.loadfiles('C:\\Users\\fabri\\git\\Intro-to-ML\\Task0\\Raw_Data') # Surface
X_train = dataloader.loadX_train()
y_train = dataloader.loady_train()
X_test = dataloader.loadX_test()

