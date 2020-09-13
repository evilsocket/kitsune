#!/usr/bin/env python3
import os
import json
import argparse
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf

import kitsune.data as data
import kitsune.dnn as dnn

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", help="Dataset CSV file.", default='dataset.csv')
parser.add_argument("--output", help="Output model file.", default='model.h5')
parser.add_argument("--epochs", help="Epochs to train for.", type=int, default=200)
parser.add_argument("--params", help="Read hyper parameters from this file.", default=None)
parser.add_argument("--batch_size", help="Batch size.", type=int, default=64)

args = parser.parse_args()

(datamin, datamax, dataset) = data.load(args.dataset)

output_path = os.path.dirname(args.output)

data.save_normalizer(os.path.join(output_path, "norm.json"), datamin, datamax)

feature_names = list(dataset.columns)
feature_names.remove('label')

print("data shape: %s (%d features)" % (str(dataset.shape), len(feature_names)))

neg, pos = np.bincount(dataset['label'])

print("bots:%d legit:%d" % (pos, neg))

X_train, Y_train, X_test, Y_test, X_val, Y_val = data.split(dataset, p_test=0.15, p_val=0.15)

grid_search = args.params is None
if grid_search:
    h_params = { 
        'num_hidden_layers': [6, 5, 4, 3, 2, 1],
        'size_first_hidden': [512, 256, 128, 64],
        'dropout': [0.0, 0.1, 0.2, 0.3], 
        'learning_rate': [0.1, 0.2],
        'momentum': [0.8, 0.7, 0.6]
    }
else:
    with open(args.params) as fp:
        h_params = json.load(fp)

    h_params = { key : [value] for key, value in h_params.items() }
    
best_accuracy = 0.0
best_params = None
best_model = None

for param_values in itertools.product(*h_params.values()):
    params = dict(zip(h_params, param_values))

    print("training with %s" % params)

    model = dnn.build_for(X_train, Y_train, args.epochs, **params)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', min_delta=0.00000001, patience = 25, mode = 'auto')

    model.fit( X_train, Y_train,
            validation_data = (X_val, Y_val),
            batch_size = args.batch_size,
            epochs = args.epochs,
            verbose = 0,
            callbacks=[early_stop])

    metrics = model.evaluate(X_test,  Y_test, verbose=0)
    if metrics[2] > best_accuracy:
        best_accuracy = metrics[2]
        best_params = params
        best_model = model
        if grid_search:
            print("NEW BEST (%f): %s" % (best_accuracy, best_params))

print()

if grid_search:
    print("best with %f: %s" % (best_accuracy, best_params))
    print()

best_model.save(args.output)

print("model saved to %s ..." % args.output)

print("running on %d test samples ..." % X_test.shape[0])

metrics = best_model.evaluate(X_test,  Y_test, verbose=0)
metrics_names = ['loss', 'binary_crossentropy', 'bin_accuracy']
print()

for i in range(len(metrics)):
    print("%20s : %f" % (metrics_names[i], metrics[i]))

if grid_search:
    params_file = os.path.join(output_path, "hyper_params.json")
    with open(params_file, 'w+t') as fp:
        json.dump(best_params, fp, indent=4, sort_keys=True)
    print("optimal hyper parameters saved to %s" % params_file)

else:
    print("running differential evaluation on %d features ..." % len(feature_names))

    # baseline

    ref_metrics = best_model.evaluate(X_test,  Y_test, verbose=0)
    ref_metrics_names = ['loss', 'binary_crossentropy', 'bin_accuracy']
    sort_metric = 'bin_accuracy'
    by_feature = {}

    for feature_name in feature_names:
        print("testing feature %s ..." % feature_name)

        X_test_run = X_test.copy()
        X_test_run[feature_name] = 0.0

        metrics = best_model.evaluate(X_test_run,  Y_test, verbose=0)
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
 