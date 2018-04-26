import os
import numpy as np
import tensorflow as tf
import time
import pandas as pd
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from utils_output import PrintOutput
from utils_NN import NeuralNetworks as NN
from utils_training import train, predict
from utils_preprocessing import centering

config = tf.ConfigProto()
config.gpu_options.allow_growth = True #Do not assign whole gpu memory, just use it on the go
config.allow_soft_placement = True #If an operation is not defined in the default device, let it execute in another.

timestr = time.strftime("%Y%m%d-%H%M%S")

random_seed = 123
np.random.seed(random_seed)
tf.set_random_seed(random_seed)

# Data Path
CallFolder = '../Raw_Data/'
StoreFolder ='Final_Results/'
if not os.path.isdir(StoreFolder):
    os.makedirs(StoreFolder)

StoreFolder_selfeval ='Selfeval_Results/'
if not os.path.isdir(StoreFolder_selfeval):
    os.makedirs(StoreFolder_selfeval)

#########################################################
# Decide whether self-evaluation or final submission
final_submission = True
Train_split = 9./10

# You want to preprocess the data?
preprocessing = True

# Hyperparameters
epochs = 81
batch_size = 32
learning_rate = 0.001
params = 88

#########################################################
# LOAD AND SHUFFLE DATA!
DataTrain = np.array(pd.read_hdf(CallFolder + "train.h5", "train"))
X_train = DataTrain[:, 1:]
y_train = DataTrain[:, 0]
classes = np.max(y_train)+1

X_test = np.array(pd.read_hdf(CallFolder + "test.h5", "test"))
print('X_train:   ', X_train.shape, end=' ||  ')
print('y_train:   ', y_train.shape)
print('X_test:    ', X_test.shape)

(X_train, y_train) = shuffle(X_train, y_train)

#########################################################
# TRAIN DATA
if final_submission == True:
    if preprocessing == True:
        X_train, X_test = centering(X_train, X_test)

    print('Shape of X_train:', X_train.shape)
    print('Shape of y_train:', y_train.shape)
    print('Shape of X_test:', X_test.shape, '\n')

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
    
    
    PrintOutput(y_test_pred, os.path.join(StoreFolder, timestr + '_' + str(epochs) + '_' + str(batch_size) + '_' + str(params) + '_y_test.csv'))

else:
    samples = len(X_train)
    X_train_selfeval = X_train[0:int(Train_split * samples), :]
    y_train_selfeval = y_train[0:int(Train_split * samples)]
    print('Shape of X_train:', X_train_selfeval.shape)
    print('Shape of y_train:', y_train_selfeval.shape)

    X_test_selfeval = X_train[int(Train_split * samples):samples, :]
    y_test_selfeval = y_train[int(Train_split * samples):samples]
    print('Shape of X_test:', X_test_selfeval.shape)
    print('Shape of y_test:', y_test_selfeval.shape)

    if preprocessing == True:
        X_train_selfeval, X_test_selfeval = centering(X_train_selfeval, X_test_selfeval)

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
    
    
    PrintOutput(y_test_pred, os.path.join(StoreFolder_selfeval, timestr + '_' + str(epochs) + '_' + str(batch_size) + '_' + str(params) + '_y_test.csv'))

##################
# POSTPROCESS

plt.figure(1)
plt.plot(range(1, len(avg_loss_plot) + 1), avg_loss_plot)
plt.title('Training loss')
plt.xlabel('Epoch')
plt.ylabel('Average Training Loss')
if final_submission == True:
    plt.savefig(os.path.join(StoreFolder, timestr + '_' + str(epochs) + '_' + str(batch_size) + '_' + str(params) + '_TrainLoss.jpg'))
else:
    plt.savefig(os.path.join(StoreFolder_selfeval, timestr + '_' + str(epochs) + '_' + str(batch_size) + '_' + str(params) + '_TrainLoss.jpg'))

if final_submission == False:
    plt.figure(2)
    plt.plot(range(1, len(test_accuracy_plot) + 1), test_accuracy_plot, label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(StoreFolder_selfeval, timestr + '_' + str(epochs) + '_' + str(batch_size) + '_' + str(params) + '_TestAccuracy.jpg'))

print('\nJob Done!')