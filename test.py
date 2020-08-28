#!/usr/bin/env python3
import os
import sys
import json
import glob
import random
import csv
import tensorflow as tf

import kitsune.profile as profile
import kitsune.data as data

if len(sys.argv) < 2:
    print("usage: %s <profile path>" % sys.argv[0])
    quit()

def print_pred(screen_name, prediction):
    if prediction[0] > prediction[1]:
        label = 'legit'
        confidence = prediction[0]
        print("%20s   legit   %f %%" % (screen_name, prediction[0] * 100.0))
    else:
        label = 'bot'
        confidence = prediction[1]
        print("%20s   bot     %f %%" % (screen_name, prediction[1] * 100.0))

    return (label, confidence * 100.0)

def get_class(prediction):
    if prediction[0] > prediction[1]:
        label = 'legit'
        confidence = prediction[0]
    else:
        label = 'bot'
        confidence = prediction[1]
    return (label, confidence * 100.0) 

def compare_predictions(prev, next):
    (prev_label, prev_conf) = get_class(prev)
    (next_label, next_conf) = get_class(next)

    if prev_label == next_label:
        return "%f" % (prev_conf - next_conf)
    else:
        return "%f" % (10.0 + (prev_conf - next_conf))


norm = data.load_normalizer('norm.json')
model = tf.keras.models.load_model('model.h5')

# model.summary()

profile_path = sys.argv[1]
profile_file = os.path.join(profile_path, 'profile.json')
output_file  = os.path.join(profile_path, 'predictions.csv')
profile_paths = []
single_mode = False

if os.path.exists(profile_file):
    profile_paths = [profile_path]
    single_mode   = True
else:
    profile_paths = list(glob.glob(os.path.join(profile_path, "*")))

print("writing predictions to %s ..." % output_file)

print("\n-------\n")
print("         screen_name | class | confidence\n")

samples_per_inference = 5
samples = {}

with open(output_file, 'w+t', newline='') as fp:
    w = csv.writer(fp)

    w.writerow(['user_id', 'user_name', 'prediction', 'confidence'])

    for profile_path in profile_paths:
        prof = profile.load(profile_path, quiet=True)
        if prof is None:
            continue

        ((user, tweets, replies, retweets), vector) = prof
        
        vector = data.nomalized_from_dict(norm, vector)
        prediction = model.predict(vector)[0]
        
        (label, confidence) = print_pred(user['screen_name'], prediction)

        w.writerow([user['id'], user['screen_name'], label, confidence])
        """
        if single_mode:
            for c in vector.columns:
                v = vector.copy()
                v[c] = 0.0
                new_prediction = model.predict(v)[0]
                print("%30s : %s" % (c, compare_predictions(prediction, new_prediction)))
        """
