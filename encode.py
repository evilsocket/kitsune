#!/usr/bin/env python3
import os
import multiprocessing as mp
import glob
import csv

import kitsune.profile as profile

output = 'dataset.csv'

profile_paths = {
    'bot': ['../icsy_who/profiles/', '../icsy_salvini/profiles/'],
    'legit': ['../icsy_legit/profiles/']
}

profiles = {
    'bot': [],
    'legit': []
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

print("saving to %s ..." % output)   

with open(output, 'w+t') as fp:
    w = csv.DictWriter(fp, fieldnames=['label'] + list(first_row.keys()))
    w.writeheader()

    for label, records in profiles.items():
        for (data, vector) in records:
            vector['label'] = label
            w.writerow(vector)