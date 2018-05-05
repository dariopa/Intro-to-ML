import os
import shutil
import numpy as np
import tensorflow as tf
import time
import pandas as pd
import random
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from utils_output import PrintOutput
from utils_NN import NeuralNetworks as NN
from utils_training import train, predict, load, save
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

StoreFolder_Model ='Models/'
if os.path.exists(StoreFolder_Model) and os.path.isdir(StoreFolder_Model):
    shutil.rmtree(StoreFolder_Model)
if not os.path.isdir(StoreFolder_selfeval):
    os.makedirs(StoreFolder_selfeval)

#########################################################
# Decide whether self-evaluation or final submission
final_submission = True
Test_split = 9.5/10
Val_split = 9.5/10

# You want to preprocess the data?
preprocessing = True

# how many random samples?
random_selection = False
random_samples = 45000

# Hyperparameters
epochs = 30
batch_size = 254
learning_rate = 0.001
params = 380
activation = tf.nn.relu

#########################################################
# LOAD AND SHUFFLE DATA!
DataTrain = np.array(pd.read_hdf(CallFolder + "train.h5", "train"))
X_train = DataTrain[:, 1:]
y_train = DataTrain[:, 0]
classes = np.max(y_train) + 1

X_test = np.array(pd.read_hdf(CallFolder + "test.h5", "test"))
print('Unpreprocessed Data')
print('X_train:   ', X_train.shape, end=' ||  ')
print('y_train:   ', y_train.shape)
print('X_test:    ', X_test.shape, '\n')

(X_train, y_train) = shuffle(X_train, y_train)

#########################################################
# RANDOMSELECTOR
if random_selection == True:
    samples = len(X_train)
    random_list = []
    for i in range(random_samples):
        random_list.append(random.randint(0, samples-1))

    X_train = X_train[random_list, :]
    y_train = y_train[random_list]
    print('Randomly selected data')
    print('X_train:   ', X_train.shape, end=' ||  ')
    print('y_train:   ', y_train.shape)
    print('X_test:    ', X_test.shape, '\n')

#########################################################
# FINAL DATA
if final_submission == True:
    if preprocessing == True:
        X_train, X_test = centering(X_train, X_test)
        X_train = normalize(X_train)
        X_test = normalize(X_test)

    samples = len(X_train)
    X_valid = X_train[int(Val_split * samples):samples, :]
    y_valid = y_train[int(Val_split * samples):samples]
    X_train = X_train[0:int(Val_split * samples), :]
    y_train = y_train[0:int(Val_split * samples)]
    print('Final Data')
    print('Shape of X_train:', X_train.shape)
    print('Shape of y_train:', y_train.shape)
    print('Shape of X_valid:', X_valid.shape)
    print('Shape of y_valid:', y_valid.shape, '\n')

    ##################
    # CREATE GRAPH
    g = tf.Graph()
    with g.as_default():
        # build the graph
        NN.build_NN(classes, learning_rate, params, activation)

    ##################
    # TRAINING
    print()
    print('Training... ')
    with tf.Session(graph=g, config=config) as sess:
        [avg_loss_plot, valid_accuracy_plot, test_accuracy_plot] = train(path=StoreFolder_Model, sess=sess, epochs=epochs,
                                                                         random_seed=random_seed,
                                                                         batch_size=batch_size,                                                                 
                                                                         training_set=(X_train, y_train),
                                                                         validation_set=(X_valid, y_valid),
                                                                         test_set=None)

        np.save(os.path.join(StoreFolder, timestr + '_avg_loss_plot.npy'), avg_loss_plot)
    del g

    ##################
    # CREATE GRAPH
    g2 = tf.Graph()
    with g2.as_default():
        # build the graph
        NN.build_NN(classes, learning_rate, params, activation)

        # Saver
        saver = tf.train.Saver()

    ##################
    # PREDICTION
    with tf.Session(graph=g2, config=config) as sess:
        epoch = np.argmax(valid_accuracy_plot) + 1
        load(saver=saver, sess=sess, epoch=epoch, path=StoreFolder_Model)
        y_test_pred = predict(sess, X_test)

    PrintOutput(y_test_pred, os.path.join(StoreFolder, timestr + '_' + str(epochs) + '_' + str(batch_size) + '_' + str(params) + '_y_test.csv'))

#################################################################################################
#################################################################################################
# SELFEVALUATION
else:
    samples = len(X_train)
    X_train_selfeval = X_train[0:int(Test_split * samples), :]
    y_train_selfeval = y_train[0:int(Test_split * samples)]
    X_test_selfeval = X_train[int(Test_split * samples):samples, :]
    y_test_selfeval = y_train[int(Test_split * samples):samples]
    print('Self-evaluation data')
    print('Shape of X_train:', X_train_selfeval.shape)
    print('Shape of y_train:', y_train_selfeval.shape)
    print('Shape of X_test:', X_test_selfeval.shape)
    print('Shape of y_test:', y_test_selfeval.shape)

    if preprocessing == True:
        X_train_selfeval, X_test_selfeval = centering(X_train_selfeval, X_test_selfeval)
        X_train_selfeval = normalize(X_train_selfeval)
        X_test_selfeval = normalize(X_test_selfeval)

    ##################
    # CREATE GRAPH TRAINING
    g = tf.Graph()
    with g.as_default():
        # build the graph
        NN.build_NN(classes, learning_rate, params, activation)

    ##################
    # TRAINING
    print()
    print('Training... ')
    with tf.Session(graph=g, config=config) as sess:
        [avg_loss_plot, valid_accuracy_plot, test_accuracy_plot] = train(path=StoreFolder_Model, sess=sess, epochs=epochs,
                                                                         random_seed=random_seed,
                                                                         batch_size=batch_size,                                                                 
                                                                         training_set=(X_train_selfeval, y_train_selfeval),
                                                                         validation_set=None,
                                                                         test_set=(X_test_selfeval, y_test_selfeval))

        np.save(os.path.join(StoreFolder_selfeval, timestr + '_avg_loss_plot.npy'), avg_loss_plot)
        np.save(os.path.join(StoreFolder_selfeval, timestr + '_test_accuracy_plot.npy'), test_accuracy_plot)

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