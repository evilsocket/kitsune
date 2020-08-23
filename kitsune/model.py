import numpy as np
import tensorflow as tf

def build_for(X, Y, optimizer='adam'):
    n_inputs       = X.shape[1]
    n_outputs      = Y.shape[1]
    dropout        = 0.2

    print("building model: inputs=%d outputs=%d" % (n_inputs, n_outputs))

    tf.keras.backend.set_floatx('float64')

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_inputs,), dtype=np.float64, name='input'),

        tf.keras.layers.Dense(units=256, activation='relu', name='hidden_0'),
        tf.keras.layers.Dropout(dropout, name='dropout_0'),

        tf.keras.layers.Dense(units=128, activation='relu', name='hidden_1'),
        tf.keras.layers.Dropout(dropout, name='dropout_1'),

        tf.keras.layers.Dense(units=64, activation='relu', name='hidden_2'),
        tf.keras.layers.Dropout(dropout, name='dropout_2'),

        tf.keras.layers.Dense(units=n_outputs, activation='softmax', name='output') 
    ])

    loss      = 'binary_crossentropy' if n_outputs == 2 else 'categorical_crossentropy'
    metrics   = [ tf.keras.metrics.binary_crossentropy, tf.keras.metrics.binary_accuracy ]

    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

    return model