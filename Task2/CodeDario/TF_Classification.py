import numpy as np
import tensorflow as tf
import time
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import tensorflow.contrib.keras as keras
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from utils_models import KERAS
from utils_output import PrintOutput

np.random.seed(123)
tf.set_random_seed(123)

#########################################################
# Decide whether self-evaluation or final submission
final_submission = True
Train_split = 8./10

# Hyperparameters
epochs = 100
param = 30
layers = 2
batch_size = 32

#########################################################
# LOAD AND SHUFFLE DATA!
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")

(X_train, y_train) = shuffle(X_train, y_train)

y_train_onehot = keras.utils.to_categorical(y_train)
print('First 3 labels: ', y_train[:3])
print('First 3 onehot labels:\n', y_train_onehot[:3])
               
#########################################################
# TRAIN DATA
if final_submission == True:

    print('Shape of X_train:', X_train.shape)
    print('Shape of y_train:', y_train.shape)
    print('Shape of X_test:', X_test.shape)

    # build model:
    model = KERAS.build(X_train, y_train_onehot, param, layers)
    # train model:
    trained_model, losses = KERAS.fit(model, X_train, y_train_onehot, epochs, batch_size)
    # predict labels:
    y_test_pred = KERAS.predict(trained_model, X_test)
    
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    PrintOutput(y_test_pred, timestr + '_' + str(epochs) + '_' + str(param) + '_' + str(layers) + '_' + str(batch_size) + '_y_test.csv')
    print('\nJob Done!')

else:
    timestr = time.strftime("%Y%m%d-%H%M%S")
    samples = len(X_train)
    X_train_selfeval = X_train[0:int(Train_split * samples), :]
    y_train_onehot_selfeval = y_train_onehot[0:int(Train_split * samples)]
    y_train_selfeval = y_train[0:int(Train_split * samples)]
    print('Shape of X_train:', X_train_selfeval.shape)
    print('Shape of y_train_onehot:', y_train_onehot_selfeval.shape)
    print('Shape of y_train:', y_train_selfeval.shape)

    X_test_selfeval = X_train[int(Train_split * samples):samples, :]
    y_test_onehot_selfeval = y_train_onehot[int(Train_split * samples):samples]
    y_test_selfeval = y_train[int(Train_split * samples):samples]
    print('Shape of X_test:', X_test_selfeval.shape)
    print('Shape of y_test_onehot:', y_test_onehot_selfeval.shape)
    print('Shape of y_test:', y_test_selfeval.shape)

    # build model:
    model = KERAS.build(X_train_selfeval, y_train_onehot, param, layers)
    # train model:
    trained_model, losses = KERAS.fit(model, X_train, y_train_onehot, epochs, batch_size)
    # predict labels:
    y_test_pred = KERAS.predict(trained_model, X_test_selfeval)

    # PREDICT SELF-EVALUATION!
    correct_preds = np.sum(y_test_pred == y_test_selfeval, axis=0)
    test_acc = correct_preds / y_test_selfeval.shape[0]
    print('Test Accuracy:  %.2f%%' % (test_acc * 100))
    print('\nJob Done!')


plt.figure(1)
plt.plot(range(1, len(losses) + 1), losses)
plt.title('Training loss')
plt.xlabel('Epoch')
plt.ylabel('Average Training Loss')
plt.savefig(timestr + '_' + str(epochs) + '_' + str(param) + '_' + str(layers) + '_' + str(batch_size) + '_TrainLoss.jpg')