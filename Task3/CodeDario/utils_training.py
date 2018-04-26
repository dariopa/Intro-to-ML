import os
import numpy as np
import tensorflow as tf
import time


def batch_generator(X, y, batch_size, random_seed, shuffle=False):
    idx = np.arange(y.shape[0])
    
    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]
    
    for i in range(0, X.shape[0], batch_size):
        yield (X[i:i+batch_size, :], y[i:i+batch_size])

def train(sess, epochs, random_seed, batch_size, training_set,
          test_set=None, initialize=True, shuffle=True,
          dropout=0.5):

    X_data_train = np.array(training_set[0])
    y_data_train = np.array(training_set[1])
    avg_loss_plot = []
    test_accuracy_plot = []

    saver = tf.train.Saver()

    # initialize variables
    if initialize:
        sess.run(tf.global_variables_initializer())

    for epoch in range(1, epochs+1):
        avg_loss = 0.0   
        start_time = time.time()
        batch_gen = batch_generator(X_data_train, y_data_train, batch_size, random_seed, shuffle=shuffle)
        for i,(batch_x, batch_y) in enumerate(batch_gen):
            feed = {'tf_x:0': batch_x, 
                    'tf_y:0': batch_y, 
                    'fc_keep_prob:0': dropout}
            loss, _ = sess.run(['cross_entropy_loss:0', 'train_op'], feed_dict=feed)
            avg_loss += loss

        avg_loss_plot.append(np.mean(avg_loss))
        print('\nEpoch %02d Training Avg. Loss: %7.3f' % (epoch, avg_loss), end=' ')

        if test_set is not None:
            feed = {'tf_x:0': test_set[0],
                    'tf_y:0': test_set[1],
                    'fc_keep_prob:0':1.0}
            test_acc = 100*sess.run('accuracy:0', feed_dict=feed)
            test_accuracy_plot.append(test_acc)
            print(' Test Acc: %7.3f%%' % test_acc)
        else:
            print()

        end_time = time.time()
        print("Total time taken this loop [s]: ", end_time - start_time)
        if epoch == 1:
            print('Termination time will be:  ', time.ctime(start_time + (end_time - start_time)*epochs))

    return avg_loss_plot, test_accuracy_plot

def predict(sess, X, return_proba=False):
    feed = {'tf_x:0': X, 'fc_keep_prob:0': 1.0}
    if return_proba:
        return sess.run('probabilities:0', feed_dict=feed)
    else:
        return sess.run('labels:0', feed_dict=feed)
