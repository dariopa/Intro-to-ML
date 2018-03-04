import os
from sklearn.pipeline import Pipeline
import preprocessing
import regression
import output

## Load Data
dataloader = preprocessing.loadfiles('C:\\Users\\fabri\\git\\Intro-to-ML\\Task0\\Raw_Data') # Define datafolder - HomePC
# dataloader = preprocessing.loadfiles('C:\\Users\\fabri\\git\\Intro-to-ML\\Task0\\Raw_Data') # Surface
X_train = dataloader.loadX_train()
y_train = dataloader.loady_train()
X_test = dataloader.loadX_test()

## Linear Regression
LinReg = regression.LinearRegression()
LinReg.fit(X_train, y_train) 
y_pred = LinReg.predict(X_test)

## Output Generation
datasaver = output.savedata('C:\\Users\\fabri\\git\\Output', 'C:\\Users\\fabri\\git\\Intro-to-ML\\Task0\\Raw_Data') # Savepath, Datapath
datasaver.saveprediction(y_pred)