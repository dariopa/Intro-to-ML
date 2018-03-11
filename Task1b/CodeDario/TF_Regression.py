import tensorflow as tf
import numpy as np
from utils import train_linreg, TfLinreg
from OutputFormat import PrintOutput_1b
 
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

lrmodel = TfLinreg(x_dim=X_train.shape[1], learning_rate=0.001)

sess = tf.Session(graph=lrmodel.g)
training_costs, session = train_linreg(sess, lrmodel, X_train, y_train, 1500)

weights = np.array(session.run('weight:0'), dtype=np.float32)
print(weights)
PrintOutput_1b(weights)

import matplotlib.pyplot as plt

plt.plot(range(1,len(training_costs) + 1), training_costs)
plt.title('Training loss')
plt.xlabel('Epoch')
plt.ylabel('Training Cost')
plt.savefig('TrainLoss.jpg')