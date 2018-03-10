import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import preprocessing
import regression
import output
import scoring

############################################ Configuration ###################################################
## General
BLoadData = 1
BFinalPrediction = 0
## Preprocessing
BTask1bTransformation = 1
## Regression
BLinearRegression = 0
BRidgeRegression = 0
BKFoldCrossValidation = 0
## Postprocessing
## Score
BRMSEScore = 0
##############################################################################################################

## Load Data
if BLoadData == 1:
    dataloader = preprocessing.loadfiles('C:\\Users\\fabri\\git\\Intro-to-ML\\Task1b\\Raw_Data') # Define datafolder - HomePC
    # dataloader = preprocessing.loadfiles('C:\\Users\\fabri\\git\\Intro-to-ML\\Task0\\Raw_Data') # Surface
    X_train = dataloader.loadX_train()
    y_train = dataloader.loady_train()
    print('X_train shape: ', X_train.shape)
    print('y_train shape: ', y_train.shape)
    

## Train / Test Split
if BFinalPrediction == 0:
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42) 

## Preprocessiong
if BTask1bTransformation == 1: 
    trafo = preprocessing.task1btransformation()
    X_train = trafo.transform(X_train)

## Linear Regression
if BLinearRegression == 1:
    LinReg = regression.LinearRegression()
    LinReg.fit(X_train, y_train) 
    y_pred = LinReg.predict(X_test)

if BRidgeRegression == 1:
    l = 1
    RidgeReg = regression.RidgeRegression(alpha = l)
    RidgeReg.fit(X_train, y_train) 
    y_pred = RidgeReg.predict(X_test)

## KFold Cross validation
if BKFoldCrossValidation == 1:
    lambda_array = [0.1, 1, 10, 100, 1000]
    scores = np.empty([5,1])
    i = 0
    X = X_train
    y = y_train
    for l in lambda_array:
        scoresum = 0.0
        kf = KFold(n_splits=10, random_state=42)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            RidgeReg = regression.RidgeRegression(alpha = l)
            RidgeReg.fit(X_train, y_train)
            y_pred = RidgeReg.predict(X_test)
            scorer = scoring.score()
            score = scorer.RMSE(y_test, y_pred)
            scoresum = scoresum + score
        scores[i] = scoresum / 10
        i += 1
    print(scores)

## Score
if BRMSEScore == 1 and BFinalPrediction == 0:
    scorer = scoring.score()
    score = scorer.RMSE(y_test, y_pred)
    print('RMSE score is = ', repr(score))
 
## Output Generation
if BFinalPrediction == 1:
    datasaver = output.savetask1a('C:\\Users\\fabri\\git\\Output', 'C:\\Users\\fabri\\git\\Intro-to-ML\\Task1b\\Raw_Data') # Savepath, Datapath
    datasaver.saveprediction(scores)