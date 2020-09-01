#!/usr/bin/env python3
import os
import multiprocessing as mp
import glob
import collections
import csv
import json
import argparse

import kitsune.profile as profile
import kitsune.features as features

def profile_only_wrapper(profile_path):
    return profile.load(profile_path, limit=0, quiet=True)

parser = argparse.ArgumentParser()

parser.add_argument("--label_a", help="Label of the first group of accounts.", default='bot')
parser.add_argument("--path_a", help="Path of the first group of accounts.", required=True)

parser.add_argument("--label_b", help="Label of the second group of accounts.", default='legit')
parser.add_argument("--path_b", help="Path of the second group of accounts.", required=True)

parser.add_argument("--output", help="Output CSV file.", default='dataset.csv')

args = parser.parse_args()

profile_paths = {
    args.label_a: [args.path_a],
    args.label_b: [args.path_b],
}

profiles_data = {
    args.label_a: [],
    args.label_b: []
}

for profiles_label, profiles_paths in profile_paths.items():
    for profiles_path in profiles_paths:
        for filename in glob.glob(os.path.join(profiles_path, "*")):
            profiles_data[profiles_label].append(filename)

agents = mp.cpu_count()
chunksize = 16
num_profiles = len(profiles_data[args.label_a]) + len(profiles_data[args.label_b])

corpus              = []
dictionary          = collections.Counter()

print("building dictionary from %d profiles ..." % num_profiles)

for label, files in profiles_data.items():
    with mp.Pool(processes=agents) as pool:
        results = pool.map(profile_only_wrapper, files, chunksize)
        corpus += [p[0][0]['description'] for p in results 
                    if p is not None and p[0][0]['description'] is not None]

# 0: padding
# 1: unknown
symbols = features.get_corpus_dictionary(corpus)
symbols_filename = os.path.join(os.path.dirname(args.output), 'symbols.json')

print("built dictionary of %d symbols, saving to %s" % (len(symbols), symbols_filename))

with open(symbols_filename, 'w+t') as fp:
    json.dump(features.symbols, fp, indent=2, sort_keys=True)    

for label, files in profiles_data.items():
    with mp.Pool(processes=agents) as pool:
        results = pool.map(profile.load, files, chunksize)
        profiles_data[label] = [r for r in results if r is not None]

first_row = None
for label, records in profiles_data.items():
    for (data, vector) in records:
        first_row = vector
        break
    break

print("saving to %s ..." % args.output)   

with open(args.output, 'w+t') as fp:
    w = csv.DictWriter(fp, fieldnames=['label'] + list(first_row.keys()))
    w.writeheader()

    for label, records in profiles_data.items():
        for (data, vector) in records:
            vector['label'] = label
            w.writerow(vector)