import tensorflow as tf
import tensorflow.contrib.keras as keras
import numpy as np

class KERAS():
    def build(X_train, y_train_onehot):
        model = keras.models.Sequential()

        model.add(keras.layers.Dense(
                units=30,
                input_dim=X_train.shape[1],
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                activation='relu'))

        model.add(keras.layers.Dense(
                units=30,
                input_dim=30,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                activation='relu'))

        model.add(keras.layers.Dense(
                units=y_train_onehot.shape[1],
                input_dim=30,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                activation='softmax'))

        sgd_optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-9)

        model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy')
        return model
    
    def fit(model, X_train, y_train_onehot, epochs):
        history = model.fit(X_train, y_train_onehot, batch_size=128, epochs=epochs,
                            verbose=1,
                            validation_split=0.001)
        return model

    def predict(model, X_test):
        return model.predict_classes(X_test, verbose=0)
