import numpy as np
import tensorflow as tf
import time
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import tensorflow.contrib.keras as keras
from utils_NN import NeuralNetworks as NN
from utils_training import train, predict
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
random_seed = 42

epochs = 90
param = 30
layers = 2
batch_size = 32
learning_rate = 0.001

# Gridsearch
BGridSearch = 1
epoch_list = [90, 100]
param_list = [30, 36]
batch_size_list = [32, 64]
learning_rate_list = [0.001, 0.02]

## Postprocessing
## Score
BRMSEScore = 0
BAccuracy = 1
##############################################################################################################

np.random.seed(random_seed)
tf.set_random_seed(random_seed)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True #Do not assign whole gpu memory, just use it on the go
config.allow_soft_placement = True #If an operation is not defined in the default device, let it execute in another.


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

classes = np.max(y_train)+1
(X_train, y_train) = shuffle(X_train, y_train)

BDownsampling = 1
############ Class imbalance ################################
### Downsampling ###
if BDownsampling == 1: 
    downsampler = preprocessing.downsampling()
    (X_train, y_train) = downsampler.transform(X_train, y_train)

if BGridSearch == 0 or BFinalPrediction == 1: 
    ##################
    # CREATE GRAPH
    ## create a graph
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(random_seed)
        ## build the graph
        NN.build_NN(classes, learning_rate, params)

    ##################
    # TRAINING & PREDICTION
    print()
    print('Training... ')
    with tf.Session(graph=g, config=config) as sess:
        [avg_loss_plot, test_accuracy_plot] = train(sess=sess, epochs=epochs,
                                                    random_seed=random_seed,
                                                    batch_size=batch_size,                                                                 
                                                    training_set=(X_train, y_train),
                                                    test_set=None)

        np.save(os.path.join(StoreFolder, timestr + '_avg_loss_plot.npy'), avg_loss_plot)

        y_test_pred = predict(sess, X_test)

    ## Score
    if BAccuracy == 1 and BFinalPrediction == 0:
        scorer = scoring.score()
        score = scorer.Accuracy(y_test, y_test_pred)
        print('Accuracy score is = ', repr(score))

if BGridSearch == 1 and BFinalPrediction == 0: 
    iters = len(epoch_list)*len(param_list)*len(layer_list)*len(batch_size_list)
    i = 0
    score_best = 0
    while i < iters:
        epochs = epoch_list[i%len(epoch_list)]
        params = param_list[math.floor((i/len(param_list))%len(epoch_list))]
        learning_rate= learning_rate_list[math.floor((i/(len(param_list)*len(layer_list)))%len(epoch_list))]
        batch_size = batch_size_list[math.floor((i/(len(param_list)*len(layer_list)*len(batch_size_list)))%len(epoch_list))]
        print('epoch = ', repr(epochs), '| param = ', repr(param), '| layers = ', repr(layers), '| batch_size = ', repr(batch_size))
        ##################
        # CREATE GRAPH
        ## create a graph
        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(random_seed)
            ## build the graph
            NN.build_NN(classes, learning_rate, params)

        ##################
        # TRAINING & PREDICTION
        print()
        print('Training... ')
        with tf.Session(graph=g, config=config) as sess:
            [avg_loss_plot, test_accuracy_plot] = train(sess=sess, epochs=epochs,
                                                        random_seed=random_seed,
                                                        batch_size=batch_size,                                                                 
                                                        training_set=(X_train_selfeval, y_train_selfeval),
                                                        test_set=(X_test_selfeval, y_test_selfeval))

            np.save(os.path.join(StoreFolder_selfeval, timestr + '_avg_loss_plot.npy'), avg_loss_plot)
            np.save(os.path.join(StoreFolder_selfeval, timestr + '_test_accuracy_plot.npy'), test_accuracy_plot)

        y_test_pred = predict(sess, X_test)

        ## Score
        if BAccuracy == 1 and BFinalPrediction == 0:
            scorer = scoring.score()
            score = scorer.Accuracy(y_test, y_test_pred)
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