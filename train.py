#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

import kitsune.data as data
import kitsune.dnn as dnn

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", help="Dataset CSV file.", default='dataset.csv')
parser.add_argument("--output", help="Output model file.", default='model.h5')

args = parser.parse_args()

(datamin, datamax, dataset) = data.load(args.dataset)

feature_names = list(dataset.columns)
feature_names.remove('label')

print("data shape: %s (%d features)" % (str(dataset.shape), len(feature_names)))

neg, pos = np.bincount(dataset['label'])

print("bots:%d legit:%d" % (pos, neg))

X_train, Y_train, X_test, Y_test, X_val, Y_val = data.split(dataset, p_test=0.15, p_val=0.15)

model = dnn.build_for(X_train, Y_train)

model.summary()

print("training model ...")

earlystop_cb = tf.keras.callbacks.EarlyStopping(monitor = 'val_binary_accuracy', min_delta=0.000001, patience = 15, mode = 'auto')

history = model.fit( X_train, Y_train,
        validation_data = (X_val, Y_val),
        batch_size = 16,
        epochs = 100,
        verbose = 2,
        callbacks=[earlystop_cb])

print()

print("saving model to %s ..." % args.output)

output_path = os.path.dirname(args.output)
model.save(args.output)
data.save_normalizer(os.path.join(output_path, "norm.json"), datamin, datamax)

print("running differential evaluation on %d features ..." % len(feature_names))

# baseline

ref_metrics = model.evaluate(X_test,  Y_test, verbose=0)
ref_metrics_names = ['loss', 'binary_crossentropy', 'bin_accuracy']
sort_metric = 'bin_accuracy'
by_feature = {}

for feature_name in feature_names:
    # print("testing feature %s ..." % feature_name)

    X_test_run = X_test.copy()
    X_test_run[feature_name] = 0.0

    metrics = model.evaluate(X_test_run,  Y_test, verbose=0)
    deltas  = {}

    for metric_index, metric_name in enumerate(ref_metrics_names):
        deltas[metric_name] = metrics[metric_index] - ref_metrics[metric_index]
    
    by_feature[feature_name] = deltas

print()

by_feature = dict(sorted(by_feature.items(), key=lambda kv: kv[1][sort_metric]))
for feature_name, deltas in by_feature.items():
    delta = deltas[sort_metric]
    if delta == 0.0:
        continue

    if delta < 0:
        print("ok: %s %f%%" % (feature_name, delta))
    else:
        print("ko: %s %f%%" % (feature_name, delta))

with open(os.path.join(output_path, 'relevances.json'), 'w+t') as fp:
    json.dump(by_feature, fp, indent=2, sort_keys=True)