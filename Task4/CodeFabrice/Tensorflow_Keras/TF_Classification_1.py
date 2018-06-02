import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import random
import matplotlib
matplotlib.use('PS') 
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import tensorflow.contrib.keras as keras
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from utils_models import KERAS
from utils_output import PrintOutput
from utils_preprocessing import centering


np.random.seed(123)
tf.set_random_seed(123)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True #Do not assign whole gpu memory, just use it on the go
config.allow_soft_placement = True #If an operation is not defined in the default device, let it execute in another.


# Data Path
CallFolder = '../../Raw_Data/'

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
final_submission = True

# Hyperparameters
epochs = 500
param = 3000
layers = 33
batch_size = 128

# At which sample starts the prediction for the test data?
sample_number = 30000

#########################################################
# LOAD AND SHUFFLE DATA!
DataTrain = np.array(pd.read_hdf(CallFolder + "train_labeled.h5", "train"))
X_train= DataTrain[:, 1:]
y_train = DataTrain[:, 0]
X_test = np.array(pd.read_hdf(CallFolder + "train_unlabeled.h5", "train")) # X_test = unlabeled data

(X_train, y_train) = shuffle(X_train, y_train)

y_train_onehot = keras.utils.to_categorical(y_train)
print('First 3 labels: ', y_train[:3])
print('First 3 onehot labels:\n', y_train_onehot[:3])

X_train_all = np.concatenate((X_train, X_test), axis=0)
np.save(os.path.join(StoreFolder_all_labeled, 'X_train.npy'), X_train_all) # STORE BEFORE PREPROCESSING, BUT AFTER SHUFFLING!
 
#########################################################
# TRAIN DATA
X_train, X_test = centering(X_train, X_test)

# build model:
model = KERAS.build(X_train, y_train_onehot, param, layers)
# train model:
trained_model, losses = KERAS.fit(model, X_train, y_train_onehot, epochs, batch_size)
# predict labels:
y_test_pred = KERAS.predict(trained_model, X_test)

timestr = time.strftime("%Y%m%d-%H%M%S")
PrintOutput(y_test_pred, sample_number, os.path.join(StoreFolder_selfeval, timestr + '_' + str(epochs) + '_' + str(param) + '_' + str(layers) + '_' + str(batch_size) + '_y_test.csv'))

y_train = np.concatenate((y_train, y_test_pred), axis=0)
np.save(os.path.join(StoreFolder_all_labeled, 'y_train.npy'), y_train)

plt.figure(1)
plt.plot(range(1, len(losses) + 1), losses)
plt.title('Training loss')
plt.xlabel('Epoch')
plt.ylabel('Average Training Loss')
plt.savefig(os.path.join(StoreFolder_selfeval, timestr + '_' + str(epochs) + '_' + str(param) + '_' + str(layers) + '_' + str(batch_size) + '_TrainLoss.jpg'))

print('\nJob Done!')
