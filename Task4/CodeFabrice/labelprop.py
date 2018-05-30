from sklearn import datasets
from sklearn.semi_supervised import LabelSpreading
from sklearn.utils import shuffle
from scipy.sparse import csgraph
import preprocessing
import output
import numpy as np

########################################################
BLoadData = 1
BFinalPrediction = 1


########################################################

if BLoadData == 1:
    dataloader = preprocessing.loadfiles4('C:\\Users\\fabri\\git\\Intro-to-ML\\Task4\\Raw_Data') # Define datafolder - HomePC
    # dataloader = preprocessing.loadfiles('C:\\Users\\fabri\\git\\Intro-to-ML\\Task0\\Raw_Data') # Surface
    X_train_labeled = dataloader.loadX_train_labeled()
    X_train_unlabeled = dataloader.loadX_train_unlabeled()
    y_train_labeled = dataloader.loady_train()
    X_test = dataloader.loadX_test()
    print('-----------------------------------------------')
    print('X_train_labeled shape: ', X_train_labeled.shape)
    print('X_train_unlabeled shape: ', X_train_unlabeled.shape)
    print('y_train_labeled shape: ', y_train_labeled.shape)
    print('X_test shape: ', X_test.shape)
    print('-----------------------------------------------')
    labelpropmodel = LabelSpreading(n_neighbors=100, kernel='knn', max_iter=100)
    y_train_unlabeled = -1*np.ones(X_train_unlabeled.shape[0])
    y_train = np.concatenate((y_train_labeled, y_train_unlabeled), axis=0)
    X_train = np.concatenate((X_train_labeled, X_train_unlabeled), axis=0)
    (X_train, y_train) = shuffle(X_train, y_train)
    labelpropmodel.fit(X_train, y_train)
    y_pred = labelpropmodel.predict(X_test)

if BFinalPrediction == 1:
    datasaver = output.savetask4('C:\\Users\\fabri\\git\\Output', 'C:\\Users\\fabri\\git\\Intro-to-ML\\Task4\\Raw_Data') # Savepath, Datapath
    datasaver.saveprediction(y_pred)