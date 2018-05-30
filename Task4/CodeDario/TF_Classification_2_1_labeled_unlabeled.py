import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
import shutil
import numpy as np
import tensorflow as tf
import time
import pandas as pd
import random
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

StoreFolder_all_labeled ='All_labeled_data/'
if not os.path.isdir(StoreFolder_all_labeled):
    os.makedirs(StoreFolder_all_labeled)

StoreFolder_Model ='Models/'
if os.path.exists(StoreFolder_Model) and os.path.isdir(StoreFolder_Model):
    shutil.rmtree(StoreFolder_Model)
if not os.path.isdir(StoreFolder_selfeval):
    os.makedirs(StoreFolder_selfeval)

#########################################################
# Decide whether self-evaluation or final submission
Val_split = 9.5/10

# You want to preprocess the data?
preprocessing = True

# Hyperparameters
epochs = 400
batch_size = 128
learning_rate = 0.0002
params = 800
activation = tf.nn.relu

# At which sample starts the prediction for the test data?
sample_number = 30000

#########################################################
# LOAD AND SHUFFLE DATA!
DataTrain = np.array(pd.read_hdf(CallFolder + "train_labeled.h5", "train"))
X_train_labeled = DataTrain[:, 1:]
features = X_train_labeled.shape[1]
y_train_labeled = DataTrain[:, 0]
classes = np.max(y_train_labeled) + 1

X_test = np.array(pd.read_hdf(CallFolder + "train_unlabeled.h5", "train")) # X_test = unlabeled data
print('Unpreprocessed Data')
print('X_train_labeled:   ', X_train_labeled.shape, end=' ||  ')
print('y_train:   ', y_train_labeled.shape)
print('X_test:    ', X_test.shape, '\n')

(X_train_labeled, y_train_labeled) = shuffle(X_train_labeled, y_train_labeled)

X_train = np.concatenate((X_train_labeled, X_test), axis=0)
np.save(os.path.join(StoreFolder_all_labeled, 'X_train.npy'), X_train) # STORE BEFORE PREPROCESSING, BUT AFTER SHUFFLING!

#########################################################
if preprocessing == True:
    X_train_labeled, X_test = centering(X_train_labeled, X_test)
    X_train_labeled = normalize(X_train_labeled)
    X_test = normalize(X_test)

samples = len(X_train_labeled)
X_valid = X_train_labeled[int(Val_split * samples):samples, :]
y_valid = y_train_labeled[int(Val_split * samples):samples]
X_train = X_train_labeled[0:int(Val_split * samples), :]
y_train = y_train_labeled[0:int(Val_split * samples)]
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
    NN.build_NN(features, classes, learning_rate, params, activation)

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

del g

##################
# CREATE GRAPH
g2 = tf.Graph()
with g2.as_default():
    # build the graph
    NN.build_NN(features, classes, learning_rate, params, activation)

    # Saver
    saver = tf.train.Saver()

##################
# PREDICTION
with tf.Session(graph=g2, config=config) as sess:
    epoch = np.argmax(valid_accuracy_plot) + 1
    load(saver=saver, sess=sess, epoch=epoch, path=StoreFolder_Model)
    y_test_pred = predict(sess, X_test)

##################
#  CREATE NEW DATASET
y_train = np.concatenate((y_train_labeled, y_test_pred), axis=0)
np.save(os.path.join(StoreFolder_all_labeled, 'y_train.npy'), y_train)