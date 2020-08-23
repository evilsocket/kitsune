#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

import kitsune.data as data
import kitsune.model as model

filename = 'dataset.csv'
output = 'model.h5'

(datamin, datamax, dataset) = data.load(filename)

print("data shape: %s" % str(dataset.shape))

neg, pos = np.bincount(dataset['label'])

print("bots:%d legit:%d" % (pos, neg))

X_train, Y_train, X_test, Y_test, X_val, Y_val = data.split(dataset, p_test=0.15, p_val=0.15)

m = model.build_for(X_train, Y_train)

m.summary()

print("training model ...")

earlystop_cb = tf.keras.callbacks.EarlyStopping(monitor = 'val_binary_accuracy', min_delta=0.000001, patience = 15, mode = 'auto')

history = m.fit( X_train, Y_train,
        validation_data = (X_val, Y_val),
        batch_size = 16,
        epochs = 100,
        verbose = 2,
        callbacks=[earlystop_cb])

print( "\nevaluating on the test dataset ...\n")

m.evaluate(X_test,  Y_test, verbose=2)

data.save_normalizer("norm.json", datamin, datamax)

print("saving model to %s ..." % output)

m.save(output)