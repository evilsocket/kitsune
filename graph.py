#!/usr/bin/env python3
import sys
import os
import glob
import csv
import pandas as pd
import multiprocessing as mp
import collections
import networkx as nx
import matplotlib.pyplot as plt

import kitsune.profile as profile

def loader(profile_path):
    return profile.load(profile_path, extract_features_vector=False)

def node_color(user_name, predictions):
    if user_name not in predictions:
        return 'grey'
    (label, confidence) = predictions[user_name]
    confidence = float(confidence)

    if confidence < 90.0:
        return 'orange'

    return 'red' if label == 'bot' else 'blue'

def is_bot(user_name, predictions):
    return user_name in predictions and predictions[user_name][0] == 'bot'

if len(sys.argv) < 2:
    print("usage: %s <profile path>" % sys.argv[0])
    quit()

profile_path = sys.argv[1]
profile_paths = list(glob.glob(os.path.join(profile_path, "*")))
predictions_file = os.path.join(profile_path, 'predictions.csv')
graph_file = os.path.join(profile_path, 'graph.gml')
predictions = {}

with open(predictions_file, 'rt') as fp:
    for no, line in enumerate(fp):
        if no == 0:
            continue

        (user_id, user_name, prediction, confidence) = line.strip().split(',')
        predictions[user_name] = (prediction, confidence)

agents = mp.cpu_count()

with mp.Pool(processes=agents) as pool:
    users = {r[0][0]['screen_name'] : r for r in pool.map(loader, profile_paths, 16) if r is not None}

edges = {}

for screen_name, data in users.items():
    ((user, tweets, replies, retweets), _) = data

    #if not is_bot(screen_name, predictions):
    #    continue

    num_retweets = len(retweets)
    counters     = collections.Counter()

    for rt in retweets:
        counters.update([rt['retweeted_status']['user']['screen_name']])

    for retweeted_screen_name, count in counters.items():
        edge_index        = (screen_name, retweeted_screen_name)
        edges[edge_index] = count / num_retweets

print("generating graph ...")

G = nx.MultiDiGraph()

for edge, weight in edges.items():
    left, right = edge
    if weight >= 0.0:
        G.add_edge( left, right, weight = weight)

pos = nx.spring_layout(G)
pos_labels = {}
for node, coords in pos.items():
    pos_labels[node] = (coords[0] + 0.01, coords[1])


min_size = 50
max_size = 1000
w_threshold = 0.3

elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] >= w_threshold]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] < w_threshold]     

nx.draw_networkx_edges(G, pos, edgelist=elarge, width=1, arrows=True)
nx.draw_networkx_edges(G, pos, edgelist=esmall, width=0.3, alpha=0.5, arrows=True, edge_color='b', style='dashed')

colors = [ node_color(p, predictions) for p in G.nodes() ]

nx.draw_networkx_nodes(G, pos, node_shape='o', node_color=colors)
nx.draw_networkx_labels(G, pos_labels)

nx.write_gml(G, graph_file)

print("graph saved to %s" % graph_file)

plt.axis('off')
plt.show()            