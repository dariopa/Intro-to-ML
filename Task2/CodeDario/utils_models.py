import tensorflow as tf
import tensorflow.contrib.keras as keras
import numpy as np

class KERAS():
    def build(X_train, y_train_onehot, param, layers):
        model = keras.models.Sequential()

        model.add(keras.layers.Dense(
                units=param,
                input_dim=X_train.shape[1],
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                activation='relu'))
        
        for i in range(layers):
                model.add(keras.layers.Dense(
                        units=param,
                        input_dim=param,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        activation='relu'))

        model.add(keras.layers.Dense(
                units=y_train_onehot.shape[1],
                input_dim=param,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                activation='softmax'))

        # sgd_optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-9)
        sgd_optimizer = keras.optimizers.Adam(lr=0.001)

        model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy')
        return model
    
    def fit(model, X_train, y_train_onehot, epochs, batch_size):
        history = model.fit(X_train, y_train_onehot, batch_size=batch_size, epochs=epochs,
                            verbose=1,
                            validation_split=None)
        return model, history.history['loss']

    def predict(model, X_test):
        return model.predict_classes(X_test, verbose=0)
