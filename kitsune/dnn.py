import numpy as np
import tensorflow as tf

import kitsune.features as features

def build_symbols_network(dropout=0.2):
    description_symbols_input = tf.keras.layers.Input(
        name='encoded-description',
        shape=(features.MAX_SYMBOLS_IN_PHRASE,), 
        dtype=np.uint16)

    x = tf.keras.layers.Embedding(features.MAX_DICTIONARY_SIZE + 2, 64)(description_symbols_input)
    x = tf.keras.layers.Conv1D(32, 4, activation='relu')(x)
    x = tf.keras.layers.Conv1D(32, 4, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(32, 4, activation='relu')(x)
    x = tf.keras.layers.Conv1D(32, 4, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Flatten()(x)

    symbols_dense = tf.keras.layers.Dense(64, activation='relu')(x)

    # symbols_dense = tf.keras.layers.Dropout(dropout)(x)

    return (description_symbols_input, symbols_dense)

def build_features_network(X, dropout=0.2):
    num_normalized_inputs = len(X['normalized_features'][0])

    normalized_features_input = tf.keras.layers.Input(
        name='normalized-features',
        shape=(num_normalized_inputs,)
    )
    x = tf.keras.layers.Dense(units=128, activation='relu')(normalized_features_input)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(units=128, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x) 
    x = tf.keras.layers.Dense(units=64, activation='relu')(x)
    
    normalized_features_dense = tf.keras.layers.Dropout(dropout)(x)

    return (normalized_features_input, normalized_features_dense)
    
def build_for(X, Y):
    num_normalized_inputs = len(X['normalized_features'][0])
    num_classes           = Y.shape[1]

    print("building neural network for: normalized inputs=%d description inputs=%d outputs=%d" % (
        num_normalized_inputs, 
        features.MAX_SYMBOLS_IN_PHRASE,
        num_classes))

    # create two networks with their own inputs and dense layers
    (normalized_features_input, normalized_features_dense) = build_features_network(X)
    (description_symbols_input, symbols_dense) = build_symbols_network()

    # concatenate the dense layers
    concat = tf.keras.layers.Concatenate()([normalized_features_dense, symbols_dense])

    # add final stage dense layers
    x = tf.keras.layers.Dense(128, activation = 'relu')(concat)
    x = tf.keras.layers.Dense(64, activation = 'relu')(x)

    # output dense layer
    output = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = tf.keras.Model(inputs=[normalized_features_input, description_symbols_input],
                           outputs=[output])

    optimizer = tf.keras.optimizers.SGD()
    loss      = 'binary_crossentropy'
    metrics   = [ tf.keras.metrics.binary_crossentropy, tf.keras.metrics.binary_accuracy ]

    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

    return model