import statistics
from statistics import mode
from time import perf_counter
import os
from pkg_resources import resource_filename
import pandas as pd
from entity_resolution_evaluation.evaluation import evaluate
from deduplipy.clustering.clustering import markov_clustering, hierarchical_clustering, connected_components
from deduplipy.datasets import load_data
from deduplipy.deduplicator import Deduplicator
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
                result = intersection(ground_cluster, data.index[data[cn] == clust_id].tolist())  # intersection between one of the foundclusters with the groundtruth cluster
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
    #pairs = pd.read_csv(os.path.join('./', 'scored_pairs_table_stoxx50.csv'), sep="|")
    myDedupliPy.save_intermediate_steps = True
elif dataset == 'voters':
    df = load_data(kind='voters5m')

    group = df.groupby(['CID']) #UNKNOWN
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
markov_col = 'deduplication_id_'+markov_clustering.__name__
hierar_col = 'deduplication_id_'+hierarchical_clustering.__name__
connected_col = 'deduplication_id_'+connected_components.__name__
myDedupliPy.verbose = True

cluster_algos = [connected_components, hierarchical_clustering, markov_clustering]
args = {hierarchical_clustering.__name__: {'cluster_threshold': 0.7},
        markov_clustering.__name__: {'inflation': 2}}
res = myDedupliPy.predict(df, clustering=cluster_algos, old_scored_pairs=pairs, args=args)

sorted_actual = res.sort_values(groupby_name)
sorted_res_cc = res.sort_values(connected_col)
sorted_res = res.sort_values(hierar_col)
sorted_res_mc = res.sort_values(markov_col)

start_eval_0 = perf_counter()
evaluate_(groundtruth, res, res.groupby([connected_col]).indices, amount, connected_col)
stop_eval_0 = perf_counter()
print(f'Evaluation 0 took:{stop_eval_0 - start_eval_0:.4f} seconds\n')

start_eval_1 = perf_counter()
evaluate_(groundtruth, res, res.groupby([hierar_col]).indices, amount, hierar_col)
stop_eval_1 = perf_counter()
print(f'Evaluation 1 took:{stop_eval_1 - start_eval_1:.4f} seconds\n')

start_eval_2 = perf_counter()
evaluate_(groundtruth, res, res.groupby([markov_col]).indices, amount, markov_col)
stop_eval_2 = perf_counter()
print(f'Evaluation 2 took:{stop_eval_2 - start_eval_2:.4f} seconds\n')
#
r0 = list(res.groupby([connected_col]).groups.values())
r1 = list(res.groupby([hierar_col]).groups.values())
r2 = list(res.groupby([markov_col]).groups.values())

s = list(group.groups.values())
rs = [("Connected", r0), ("Hierarchical", r1), ("Markov", r2)]
evaluations = ['precision', 'recall', 'f1', 'bmd']
for r in rs:
    print(f"Clustering method:{r[0]}")
    for eva in evaluations:
        print(f"{eva}: {evaluate(r[1],s,eva)}")
    print("----------------------------\n")
print("done")
