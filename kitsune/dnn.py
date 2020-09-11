import numpy as np
import tensorflow as tf

def build_for(X, Y, epochs, num_hidden_layers=5, size_first_hidden=512, dropout=0.3, learning_rate=0.1, momentum=0.8):
    n_inputs       = X.shape[1]
    n_outputs      = Y.shape[1]

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
    x = normalized_features_input
    size = size_first_hidden

    for i in range(num_hidden_layers):
        x = tf.keras.layers.Dense(units=size, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        size /= 2
        if size < 16:
            break

    normalized_features_output = tf.keras.layers.Dense(units=n_outputs, activation='softmax')(x)

    model = tf.keras.Model(inputs=[normalized_features_input],
                           outputs=[normalized_features_output])

    decay_rate = learning_rate / epochs
    optimizer = tf.keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    loss      = 'binary_crossentropy'
    metrics   = [ tf.keras.metrics.binary_crossentropy, tf.keras.metrics.binary_accuracy ]

    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

    return model