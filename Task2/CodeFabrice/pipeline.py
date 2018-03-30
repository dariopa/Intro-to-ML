import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import preprocessing
import regression
import output
import scoring
import classification

############################################ Configuration ###################################################
## General
BLoadData = 1
BFinalPrediction = 0
## Preprocessing
BTask1bTransformation = 0
## Regression
BLinearRegression = 0
BRidgeRegression = 0
BLassoRegression = 0
## Classification
BSVClassification = 0
BMultiClassSVC = 0
BMLPClassification = 1
## Misc
BKFoldCrossValidation = 0
## Postprocessing
## Score
BRMSEScore = 0
BAccuracy = 1
##############################################################################################################

## Load Data
if BLoadData == 1:
    dataloader = preprocessing.loadfiles('C:\\Users\\fabri\\git\\Intro-to-ML\\Task2\\Raw_Data') # Define datafolder - HomePC
    # dataloader = preprocessing.loadfiles('C:\\Users\\fabri\\git\\Intro-to-ML\\Task0\\Raw_Data') # Surface
    X_train = dataloader.loadX_train()
    y_train = dataloader.loady_train()
    X_test = dataloader.loadX_test()
    print('X_train shape: ', X_train.shape)
    print('y_train shape: ', y_train.shape)
    

## Train / Test Split
if BFinalPrediction == 0:
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42) 

## Preprocessiong
if BTask1bTransformation == 1: 
    trafo = preprocessing.task1btransformation()
    X_train = trafo.transform(X_train)
    if BFinalPrediction == 0:
        X_test = trafo.transform(X_test)

## Linear Regression
if BLinearRegression == 1:
    LinReg = regression.LinearRegression()
    LinReg.fit(X_train, y_train)
    if BFinalPrediction == 0: 
        y_pred = LinReg.predict(X_test)
    w = LinReg.getcoeff()

if BRidgeRegression == 1:
    l = 15
    RidgeReg = regression.RidgeRegression(alpha = l)
    RidgeReg.fit(X_train, y_train) 
    if BFinalPrediction == 0: 
        y_pred = RidgeReg.predict(X_test)
    w = RidgeReg.getcoeff()

if BLassoRegression == 1:
    l = 1
    LassoReg = regression.LassoRegression(alpha = l)
    LassoReg.fit(X_train, y_train) 
    if BFinalPrediction == 0: 
        y_pred = LassoReg.predict(X_test)
    w = LassoReg.getcoeff()

## SVM
if BSVClassification == 1:
    SVClass = classification.SVClassification()
    SVClass.fit(X_train, y_train) 
    y_pred = SVClass.predict(X_test)

## Multiclass SVC
if BMultiClassSVC == 1:
    SVClass = classification.MultiClassSVC()
    SVClass.fit(X_train, y_train) 
    y_pred = SVClass.predict(X_test)

## Multilayer Perceptron
if BMLPClassification == 1:
    MLPClass = classification.MLPClassification()
    MLPClass.fit(X_train, y_train) 
    y_pred = MLPClass.predict(X_test)


## KFold Cross validation
if BKFoldCrossValidation == 1:
    lambda_array = [0.00005, 0.0001, 0.0005]
    scores = np.empty([3,1])
    i = 0
    X = X_train
    y = y_train
    for l in lambda_array:
        scoresum = 0.0
        kf = KFold(n_splits=5, random_state=21)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if BRidgeRegression == 1:
                RidgeReg = regression.RidgeRegression(alpha = l)
                RidgeReg.fit(X_train, y_train)
                y_pred = RidgeReg.predict(X_test)
            if BLassoRegression == 1:
                LassoReg = regression.LassoRegression(alpha = l)
                LassoReg.fit(X_train, y_train) 
                y_pred = LassoReg.predict(X_test)
            scorer = scoring.score()
            score = scorer.RMSE(y_test, y_pred)
            scoresum = scoresum + score
        scores[i] = scoresum / 5
        i += 1
    print(scores)

## Score
if BRMSEScore == 1 and BFinalPrediction == 0:
    scorer = scoring.score()
    score = scorer.RMSE(y_test, y_pred)
    print('RMSE score is = ', repr(score))

if BAccuracy == 1 and BFinalPrediction == 0:
    scorer = scoring.score()
    score = scorer.Accuracy(y_test, y_pred)
    print('Accuracy score is = ', repr(score))

## Output Generation
if BFinalPrediction == 1:
    datasaver = output.savetask2('C:\\Users\\fabri\\git\\Output', 'C:\\Users\\fabri\\git\\Intro-to-ML\\Task2\\Raw_Data') # Savepath, Datapath
    datasaver.saveprediction(y_pred)