#!/usr/bin/env python3
import os
import sys
import json
import glob
import random
import tensorflow as tf

import kitsune.profile as profile
import kitsune.data as data

if len(sys.argv) < 2:
    print("usage: %s <profile path>" % sys.argv[0])
    quit()

def print_pred(screen_name, prediction):
    if prediction[0] > prediction[1]:
        print("%20s   legit   %f %%" % (screen_name, prediction[0] * 100.0))
    else:
        print("%20s   bot     %f %%" % (screen_name, prediction[1] * 100.0))


norm = data.load_normalizer('norm.json')
model = tf.keras.models.load_model('model.h5')

# model.summary()

profile_path = sys.argv[1]
profile_file = os.path.join(profile_path, 'profile.json')
profile_paths = []

if os.path.exists(profile_file):
    profile_paths = [profile_path]
else:
    profile_paths = list(glob.glob(os.path.join(profile_path, "*")))

print("\n-------\n")
print("         screen_name | class | confidence\n")

for profile_path in profile_paths:
    prof = profile.load(profile_path, quiet=True)
    if prof is None:
        continue

    ((user, tweets, replies, retweets), vector) = prof
    
    vector = data.nomalized_from_dict(norm, vector)

    prediction = model.predict(vector)[0]
    
    print_pred(user['screen_name'], prediction)

