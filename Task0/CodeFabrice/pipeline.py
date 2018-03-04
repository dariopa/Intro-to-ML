import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import preprocessing
import regression
import output
import scoring

##############################################################################################################
# Configuration
## General
BLoadData = 1
BFinalPrediction = 0
## Preprocessing
## Regression
BLinearRegression = 1
## Postprocessing
## Score
BRMSEScore = 1
##############################################################################################################

## Load Data
if BLoadData == 1:
    dataloader = preprocessing.loadfiles('C:\\Users\\fabri\\git\\Intro-to-ML\\Task0\\Raw_Data') # Define datafolder - HomePC
    # dataloader = preprocessing.loadfiles('C:\\Users\\fabri\\git\\Intro-to-ML\\Task0\\Raw_Data') # Surface
    X_train = dataloader.loadX_train()
    y_train = dataloader.loady_train()
    X_test = dataloader.loadX_test()

## Train / Test Split
if BFinalPrediction == 0:
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42) 

## Linear Regression
if BLinearRegression == 1:
    LinReg = regression.LinearRegression()
    LinReg.fit(X_train, y_train) 
    y_pred = LinReg.predict(X_test)

## Score
if BRMSEScore == 1 and BFinalPrediction == 0:
    scorer = scoring.score()
    score = scorer.RMSE(y_test, y_pred)
    print('RMSE score is = ', repr(score))
 
## Output Generation
if BFinalPrediction == 1:
    datasaver = output.savedata('C:\\Users\\fabri\\git\\Output', 'C:\\Users\\fabri\\git\\Intro-to-ML\\Task0\\Raw_Data') # Savepath, Datapath
    datasaver.saveprediction(y_pred)