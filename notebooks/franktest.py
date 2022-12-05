import json
import statistics
import time
from statistics import mode
from time import perf_counter
import os

import numpy as np
from pkg_resources import resource_filename
import pandas as pd
from entity_resolution_evaluation.evaluation import evaluate
from deduplipy.clustering.clustering import markov_clustering, hierarchical_clustering, connected_components
from deduplipy.datasets import load_data
from deduplipy.deduplicator import Deduplicator

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

from collections import Counter

from numpy import mean
from numpy import std

import pickle


def most_common(List):
    return (mode(List))


def intersection(lst1, lst2):
    # Use of hybrid method
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3


def mapping(gtc, data, fc, cn):
    returner = {}  # dictionary where the key is the groundtruth clusternumber and the value the mapped found clusternumber
    for i, ground_cluster in gtc.items():
        # high = 0
        # ind = -1
        # t1_start = perf_counter()
        found_in = []
        results = {}
        for val in ground_cluster:
            clust_id = data.iloc[val][cn]
            if clust_id not in found_in:
                # clust_size = len(data[data["deduplication_id"] == clust_id])
                result = intersection(ground_cluster, data.index[data[
                                                                     cn] == clust_id].tolist())  # intersection between one of the foundclusters with the groundtruth cluster
                results[clust_id] = len(result)  # / clust_size
            found_in.append(clust_id)
        f = max(results, key=results.get)
        returner[i] = f
    return returner


def precision(mapp, gtc, fc):
    pris = {}
    for i, ground_cluster in gtc.items():
        fgi = fc[mapp[i]]
        gi = ground_cluster
        up = len(intersection(fgi, gi))
        down = len(fgi)
        pri = up / down
        pris[i] = pri

    return pris


def recall(mapp, gtc, fc):
    pres = {}
    for i, ground_cluster in gtc.items():
        fgi = fc[mapp[i]]
        gi = ground_cluster
        up = len(intersection(fgi, gi))
        down = len(gi)
        pri = up / down
        pres[i] = pri

    return pres


def evaluate_(ground_truth_cluster, all_data, found_clusters, node_amount=10, coll_name="deduplication_id"):
    mapp = mapping(ground_truth_cluster, all_data, found_clusters, coll_name)
    precisions = precision(mapp, ground_truth_cluster, found_clusters)
    total_precision = 0
    for i, pre in precisions.items():
        total_precision += ((len(ground_truth_cluster[i]) / node_amount) * pre)

    recalls = recall(mapp, ground_truth_cluster, found_clusters)
    total_recall = 0
    for i, rec in recalls.items():
        total_recall += ((len(ground_truth_cluster[i]) / node_amount) * rec)

    f_measure = (2 * total_precision * total_recall) / (total_recall + total_precision)
    print(f"Total weighted precision: {total_precision}, Total weighted recall: {total_recall}\nF1: {f_measure}")


dataset = 'musicbrainz20k'
learning = False
pairs = None

if dataset == 'musicbrainz20k':
    df = load_data(kind='musicbrainz20k')
    groupby_name = 'CID'
    group = df.groupby([groupby_name])  # CID for musicbrainz, id for stoxx50
    groundtruth = group.indices
    myDedupliPy = Deduplicator(['title', 'artist', 'album'])  # ,voter: 'suburb', 'postcode'])
    if learning:
        myDedupliPy.fit(df)
        with open('musicbrainz20kfulltest.pkl', 'wb') as f:
            pickle.dump(myDedupliPy, f)
    else:
        with open('musicbrainz20kfulltest.pkl', 'rb') as f:
            myDedupliPy = pickle.load(f)
    myDedupliPy.verbose = True
    pairs = pd.read_csv(os.path.join('./', 'scored_pairs_table_musicbrainz20k.csv'), sep="|")

elif dataset == 'stoxx50':
    df = load_data(kind='stoxx50')
    groupby_name = 'id'
    group = df.groupby([groupby_name])
    groundtruth = group.indices
    myDedupliPy = Deduplicator(['name'])
    if learning:
        myDedupliPy.fit(df)
        with open('stoxx50.pkl', 'wb') as f:
            pickle.dump(myDedupliPy, f)
    else:
        with open('stoxx50.pkl', 'rb') as f:
            myDedupliPy = pickle.load(f)

    myDedupliPy.verbose = True
    pairs = pd.read_csv(os.path.join('./', 'scored_pairs_table_stoxx50.csv'), sep="|")
    # myDedupliPy.save_intermediate_steps = True
elif dataset == 'voters':
    df = load_data(kind='voters5m')

    group = df.groupby(['CID'])  # UNKNOWN
    groundtruth = group.indices
    myDedupliPy = Deduplicator(['name', 'suburb', 'postcode'])
    myDedupliPy.verbose = True
    if learning:
        myDedupliPy.fit(df)
        with open('voters5m.pkl', 'wb') as f:
            pickle.dump(myDedupliPy, f)
    else:
        with open('voters5m.pkl', 'rb') as f:
            myDedupliPy = pickle.load(f)
else:
    print("unknown")
    exit(0)

amount = len(df)
markov_col = 'deduplication_id_' + markov_clustering.__name__
hierar_col = 'deduplication_id_' + hierarchical_clustering.__name__
connected_col = 'deduplication_id_' + connected_components.__name__
myDedupliPy.verbose = True

cluster_algos = [connected_components, hierarchical_clustering, markov_clustering]
args = {hierarchical_clustering.__name__: {'cluster_threshold': 0.5},
        markov_clustering.__name__: {'inflation': 2}}
# res = myDedupliPy.predict(df, clustering=cluster_algos, old_scored_pairs=pairs, args=args)
args['use_cc'] = True
res, stat = myDedupliPy.predict(df, clustering=cluster_algos, old_scored_pairs=pairs, args=args)

sorted_actual = res.sort_values(groupby_name)
sorted_res_cc = res.sort_values(connected_col)
sorted_res = res.sort_values(hierar_col)
sorted_res_mc = res.sort_values(markov_col)

r0 = list(res.groupby([connected_col]).groups.values())
r1 = list(res.groupby([hierar_col]).groups.values())
r2 = list(res.groupby([markov_col]).groups.values())

s = list(group.groups.values())
rs = [("Connected_Components", r0), ("Hierarchical", r1), ("Markov", r2)]
evaluations = ['precision', 'recall', 'f1', 'bmd']
result = {}
result['config'] = args
result['dataset'] = dataset
print("----------------------------")
for r in rs:
    print(f"Clustering method:{r[0]}")
    result[r[0]] = {}
    for eva in evaluations:
        result[r[0]][eva] = evaluate(r[1], s, eva)
        print(f"{eva}: {result[r[0]][eva]:.4f}")
    print("----------------------------\n")

with open(os.path.join('./testruns/' + str(int(time.time())) + '.json'), 'w') as outfile:
    json.dump(result, outfile)

markovwins = []
hierarchicalwins = []
max_cluster_id = res[connected_col].nunique()
modelstats = np.empty((len(stat), 9))
modelstatspy = []
labels = []
counter = 0
for g in r0:
    rows = res.loc[g]
    if len(rows) < 2:
        continue
    connectid = rows[connected_col].iloc[0]
    print(f"ccid:{connectid}")
    gt = list(rows.groupby([groupby_name]).groups.values())
    hgroup = list(rows.groupby([hierar_col]).groups.values())
    hresult = evaluate(hgroup, gt, 'f1')

    mgroup = list(rows.groupby([markov_col]).groups.values())
    mresult = evaluate(mgroup, gt, 'f1')
    print(f"Hierarchical - F1: {hresult:.4f}, Markov - F1: {mresult:.4f}")

    if mresult > hresult:
        print("MARKOV WINS")
        if connectid in stat:
            print(stat[connectid], "\n")
        markovwins.append(connectid)
        labels.append(1)

    elif hresult > mresult:
        print("Hierarchical wins-------------")
        if connectid in stat:
            print(stat[connectid], "\n")
        hierarchicalwins.append(connectid)
        labels.append(2)
    else:
        print("Draw...")
        continue
        labels.append(0)

    if connectid in stat:
        result_ = stat[connectid].values()

        # Convert object to a list
        data_ = list(result_)
        modelstatspy.append(data_)
        # Convert list to an array
        numpyArray = np.array(data_)
        modelstats[counter] = numpyArray
        counter += 1
    else:
        d = {'clustcoefficient': 0, 'transitivity': 0,
             'diameter': 0, 'radius': 0,
             'nodecount': 1,
             'edgecount': 0, 'maxedgeweight': 0,
             'minedgeweight': 0, 'avgedgeweight': 0}
        data_ = list(d.values())
        modelstatspy.append(data_)
        # Convert list to an array
        numpyArray = np.array(data_)
        modelstats[counter] = numpyArray
        counter += 1


labels = np.array(labels)
print(modelstats.shape, labels.shape)
print(Counter(labels))
modelstats = np.array(modelstatspy)
print(modelstats.shape, labels.shape)

X_train, X_test, Y_train, Y_test = train_test_split(modelstats, labels, test_size=0.20)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model = model.fit(X_train, Y_train)

output2 = model.predict(X_test)
print(accuracy_score(Y_test, output2))
print(model.intercept_)
print(model.coef_)
print(model.classes_)
print(Counter(output2))

#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#n_scores = cross_val_score(model, modelstats, labels, scoring='accuracy', cv=cv, n_jobs=-1)
#print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
print("done")
