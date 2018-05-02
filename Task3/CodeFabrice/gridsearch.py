import numpy as np
import tensorflow as tf
import time
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import tensorflow.contrib.keras as keras
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from tf_models import KERAS
import preprocessing
import scoring
import output
from sklearn.model_selection import train_test_split
import math

############################################ Configuration ###################################################
# Decide whether self-evaluation or final submission
final_submission = False
Train_split = 8./10

## General
BLoadData = 1
BFinalPrediction = 1

# Hyperparameters
# epochs = 100
# param = 30
# layers = 2
# batch_size = 32

epochs = 90
param = 30
layers = 2
batch_size = 32

# Gridsearch
BGridSearch = 1
epoch_list = [90, 100]
param_list = [30, 36]
layer_list = [2, 3]
batch_size_list = [32, 64]

## Postprocessing
## Score
BRMSEScore = 0
BAccuracy = 1
##############################################################################################################

## Load Data
if BLoadData == 1:
    dataloader = preprocessing.loadfiles3('C:\\Users\\fabri\\git\\Intro-to-ML\\Task3\\Raw_Data') # Define datafolder - HomePC
    # dataloader = preprocessing.loadfiles('C:\\Users\\fabri\\git\\Intro-to-ML\\Task0\\Raw_Data') # Surface
    X_train = dataloader.loadX_train()
    y_train = dataloader.loady_train()
    X_test = dataloader.loadX_test()
    print('X_train shape: ', X_train.shape)
    print('y_train shape: ', y_train.shape)

## Train / Test Split 
if BFinalPrediction == 0:
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=32) 
    
if BGridSearch == 0 or BFinalPrediction == 1: 
    ## Create one hot format
    y_train_onehot = keras.utils.to_categorical(y_train)
    print('First 3 labels: ', y_train[:3])
    print('First 3 onehot labels:\n', y_train_onehot[:3])

    # build model:
    model = KERAS.build(X_train, y_train_onehot, param, layers)
    # train model:
    trained_model, losses = KERAS.fit(model, X_train, y_train_onehot, epochs, batch_size)
    # predict labels:
    y_pred = KERAS.predict(trained_model, X_test)

    ## Score
    if BAccuracy == 1 and BFinalPrediction == 0:
        scorer = scoring.score()
        score = scorer.Accuracy(y_test, y_pred)
        print('Accuracy score is = ', repr(score))

if BGridSearch == 1 and BFinalPrediction == 0: 
    iters = len(epoch_list)*len(param_list)*len(layer_list)*len(batch_size_list)
    i = 0
    score_best = 0
    while i < iters:
        epochs = epoch_list[i%len(epoch_list)]
        param = param_list[math.floor((i/len(param_list))%len(epoch_list))]
        layers = layer_list[math.floor((i/(len(param_list)*len(layer_list)))%len(epoch_list))]
        batch_size = batch_size_list[math.floor((i/(len(param_list)*len(layer_list)*len(batch_size_list)))%len(epoch_list))]
        print('epoch = ', repr(epochs), '| param = ', repr(param), '| layers = ', repr(layers), '| batch_size = ', repr(batch_size))
        ## Create one hot format
        y_train_onehot = keras.utils.to_categorical(y_train)
        print('First 3 labels: ', y_train[:3])
        print('First 3 onehot labels:\n', y_train_onehot[:3])

        # build model:
        model = KERAS.build(X_train, y_train_onehot, param, layers)
        # train model:
        trained_model, losses = KERAS.fit(model, X_train, y_train_onehot, epochs, batch_size)
        # predict labels:
        y_pred = KERAS.predict(trained_model, X_test)

        ## Score
        if BAccuracy == 1 and BFinalPrediction == 0:
            scorer = scoring.score()
            score = scorer.Accuracy(y_test, y_pred)
            print('Accuracy score is = ', repr(score))
        
        if score > score_best:
            epoch_best = epochs
            param_best = param
            layers_best = layers
            batch_size_best = batch_size
            score_best = score
        
        i += 1
    print('epoch = ', repr(epoch_best), '| param = ', repr(param_best), '| layers = ', repr(layers_best), '| batch_size = ', repr(batch_size_best), '| score = ', repr(score_best))

## Output Generation
if BFinalPrediction == 1:
    datasaver = output.savetask3('C:\\Users\\fabri\\git\\Output', 'C:\\Users\\fabri\\git\\Intro-to-ML\\Task3\\Raw_Data') # Savepath, Datapath
    datasaver.saveprediction(y_pred)