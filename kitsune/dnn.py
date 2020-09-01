import numpy as np
import tensorflow as tf

def build_for(X, Y):
    n_inputs       = X.shape[1]
    n_outputs      = Y.shape[1]
    dropout        = 0.2

    print("building neural network for: inputs=%d outputs=%d" % (n_inputs, n_outputs))

    """
    if i try to do this, for some freaking reason i get 0.5 accuracy
    during training ... i am doing something wrong ... what?

    ref. https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Normalization

    norm = tf.keras.layers.experimental.preprocessing.Normalization()
    norm.adapt(X.to_numpy())
    norm.trainable = False 

    x = norm(x)

    """

    normalized_features_input = tf.keras.layers.Input(shape=(n_inputs,), name='normalized_features_input')

    x = tf.keras.layers.Dense(units=256, activation='relu')(normalized_features_input)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(units=128, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x) 
    x = tf.keras.layers.Dense(units=64, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x) 

    normalized_features_output = tf.keras.layers.Dense(units=n_outputs, activation='softmax')(x)

    model = tf.keras.Model(inputs=[normalized_features_input],
                           outputs=[normalized_features_output])

    optimizer = tf.keras.optimizers.SGD()
    loss      = 'binary_crossentropy'
    metrics   = [ tf.keras.metrics.binary_crossentropy, tf.keras.metrics.binary_accuracy ]

    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

    return model