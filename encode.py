#!/usr/bin/env python3
import os
import multiprocessing as mp
import glob
import csv
import argparse

import kitsune.profile as profile

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

profiles = {
    args.label_a: [],
    args.label_b: []
}


for profiles_label, profiles_paths in profile_paths.items():
    for profiles_path in profiles_paths:
        for filename in glob.glob(os.path.join(profiles_path, "*")):
            profiles[profiles_label].append(filename)

agents = mp.cpu_count()
chunksize = 16

for label, files in profiles.items():
    with mp.Pool(processes=agents) as pool:
        results = pool.map(profile.load, files, chunksize)
        profiles[label] = [r for r in results if r is not None]

first_row = None
for label, records in profiles.items():
    for (data, vector) in records:
        first_row = vector
        break
    break

print("saving to %s ..." % args.output)   

with open(args.output, 'w+t') as fp:
    w = csv.DictWriter(fp, fieldnames=['label'] + list(first_row.keys()))
    w.writeheader()

    for label, records in profiles.items():
        for (data, vector) in records:
            vector['label'] = label
            w.writerow(vector)