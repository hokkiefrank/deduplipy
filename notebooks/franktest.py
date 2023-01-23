import json
import operator
import random
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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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


dataset = 'stoxx50'
learning = False
pairs = None
pairs_name = None

if dataset == 'musicbrainz20k':
    df = load_data(kind='musicbrainz20k')
    groupby_name = 'CID'
    group = df.groupby([groupby_name])  # CID for musicbrainz
    groundtruth = group.indices
    myDedupliPy = Deduplicator(['title', 'artist', 'album'])
    myDedupliPy.verbose = True
    if learning:
        myDedupliPy.fit(df)
        with open('musicbrainz20kfulltest2.pkl', 'wb') as f:
            pickle.dump(myDedupliPy, f)
    else:
        with open('musicbrainz20kfulltest2.pkl', 'rb') as f:
            myDedupliPy = pickle.load(f)
    pairs_name = "scored_pairs_table_musicbrainz20k_full.csv"
    pairs = pd.read_csv(os.path.join('./', pairs_name), sep="|")
    # myDedupliPy.save_intermediate_steps = True

elif dataset == 'musicbrainz200k':
    df = load_data(kind='musicbrainz200k')
    groupby_name = 'CID'
    group = df.groupby([groupby_name])  # CID for musicbrainz
    groundtruth = group.indices
    myDedupliPy = Deduplicator(['title', 'artist', 'album'])
    myDedupliPy.verbose = True
    if learning:
        myDedupliPy.fit(df)
        with open('musicbrainz200k.pkl', 'wb') as f:
            pickle.dump(myDedupliPy, f)
    else:
        with open('musicbrainz200k.pkl', 'rb') as f:
            myDedupliPy = pickle.load(f)
    # pairs_name = ".csv"
    # pairs = pd.read_csv(os.path.join('./', pairs_name), sep="|")

elif dataset == 'stoxx50':
    df = load_data(kind='stoxx50')
    groupby_name = 'id'
    group = df.groupby([groupby_name])
    groundtruth = group.indices
    myDedupliPy = Deduplicator(['name'])
    myDedupliPy.verbose = True
    if learning:
        myDedupliPy.fit(df)
        with open('stoxx50extrapyminhash.pkl', 'wb') as f:
            pickle.dump(myDedupliPy, f)
    else:
        with open('stoxx50.pkl', 'rb') as f:
            myDedupliPy = pickle.load(f)
    pairs_name = "scored_pairs_table_stoxx50.csv"
    pairs = pd.read_csv(os.path.join('./', pairs_name), sep="|")
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
score_thresh = 0.1
feature_count = 15
cluster_algos = [connected_components, hierarchical_clustering, markov_clustering]
cluster_algo_names = [name.__name__ for name in cluster_algos]
args = {hierarchical_clustering.__name__: {'cluster_threshold': 0.7, 'fill_missing': True},
        markov_clustering.__name__: {'inflation': 2}, 'use_cc': True,
        'score_threshold': score_thresh,
        'feature_count': feature_count}
res, stat = myDedupliPy.predict(df, clustering=cluster_algos, old_scored_pairs=pairs, score_threshold=score_thresh, args=args)

sorted_actual = res.sort_values(groupby_name)
sorted_res_cc = res.sort_values(connected_col)
sorted_res = res.sort_values(hierar_col)
sorted_res_mc = res.sort_values(markov_col)

r0 = list(res.groupby([connected_col]).groups.values())
r1 = list(res.groupby([hierar_col]).groups.values())
r2 = list(res.groupby([markov_col]).groups.values())
r3 = []
r4 = []
s = list(group.groups.values())
rs = [("Connected_Components", r0), ("Hierarchical", r1), ("Markov", r2)]
evaluations = ['precision', 'recall', 'f1', 'bmd']
result = {}
result['changes_description'] = input("Please give a short description as to what this experiment entails")
result['config'] = args
result['dataset'] = dataset
result['scored_pairs_table'] = pairs_name
print("----------------------------")

markovwins = []
hierarchicalwins = []
draws = []
allwins = {'draw': []}
for names in cluster_algo_names:
    allwins[names] = []
max_cluster_id = res[connected_col].nunique()
xfxf = list(stat)[0]
modelstats = np.empty((len(stat), len(stat[xfxf])))
connectids = []
modelstatspy = []
labels = []
counter = 0
evaluations.append('variation_of_information')


def get_cluster_column_name(clusteringalgorithm: type(abs)) -> str:
    return 'deduplication_id_' + clusteringalgorithm.__name__


def select_winner(evaluation_metrics_results: dict, evaluation_metrics_importance: dict, algo_names: list) -> str:
    evaluation_metrics_importance = dict(sorted(evaluation_metrics_importance.items(), key=operator.itemgetter(1)))
    algo_names = [name.__name__ for name in algo_names]# if name.__name__ != 'connected_components']
    losers = []
    for metricName in evaluation_metrics_importance.keys():
        func = max
        if metricName == 'bmd' or metricName == 'variation_of_information':
            func = min
        for loser in losers:
            del evaluation_metrics_results[metricName][loser]
        win = func(evaluation_metrics_results[metricName], key=evaluation_metrics_results[metricName].get)
        max_value = func(evaluation_metrics_results[metricName].values())
        winners = {key for key, value in evaluation_metrics_results[metricName].items() if value == max_value}
        if len(winners) == 1:
            # we have a clear winner on a metric, therefor we return this winner
            return win
        elif len(winners) != len(evaluation_metrics_results[metricName]):
            # we didn't have a winner, but we do have a loser somewhere, so remove the loser from the list to check.
            for clust_name in algo_names:
                if clust_name not in winners:
                    losers.append(clust_name)

    if len(winners) == len(algo_names):
        returnval = 'draw'
    else:
        returnval = random.choice(list(winners))
    return returnval


eval_prios = {'f1': 1, 'bmd': 2, 'variation_of_information': 3, 'recall': 4, 'precision': 5}
for g in r0:
    rows = res.loc[g]
    connectid = rows[connected_col].iloc[0]
    if len(rows) < 2:
        connectids.append(connectid)
        r3.append(list(rows.groupby([hierar_col]).groups.values())[0])
        r4.append(list(rows.groupby([hierar_col]).groups.values())[0])
        continue

    gt = list(rows.groupby([groupby_name]).groups.values())
    hgroup = list(rows.groupby([hierar_col]).groups.values())
    hresult = evaluate(hgroup, gt, 'f1')

    mgroup = list(rows.groupby([markov_col]).groups.values())
    mresult = evaluate(mgroup, gt, 'f1')
    temp = {'hierarchical': {}, 'markov': {}}

    cluster_groups = {}
    for cluster_algo in cluster_algos:
        algorith_name = cluster_algo.__name__
        #if algorith_name == 'connected_components':
        #    continue
        cluster_groups[algorith_name] = list(rows.groupby([get_cluster_column_name(cluster_algo)]).groups.values())
        for eva in evaluations:
            if eva not in temp:
                temp[eva] = {}
            temp[eva][algorith_name] = evaluate(cluster_groups[algorith_name], gt, eva)
            temp['hierarchical'][eva] = evaluate(hgroup, gt, eva)
            temp['markov'][eva] = evaluate(mgroup, gt, eva)
            # print(f"{eva} - Hierarchical: {evaluate(hgroup, gt, eva):.4f}")
            # print(f"{eva} - Markov: {evaluate(mgroup, gt, eva):.4f}")
    algo_winner = select_winner(temp, eval_prios, cluster_algos)
    if algo_winner == 'connected_components':
        allwins[algo_winner].append(stat[connectid])
        for gs in cluster_groups[algo_winner]:
            r3.append(gs)
        #continue
        labels.append(0)
    elif algo_winner == 'markov_clustering':
        markovwins.append(stat[connectid])
        allwins[algo_winner].append(stat[connectid])
        for gs in cluster_groups[algo_winner]:
            r3.append(gs)
        labels.append(1)
    elif algo_winner == 'hierarchical_clustering':
        allwins[algo_winner].append(stat[connectid])
        for gs in cluster_groups[algo_winner]:
            r3.append(gs)
        labels.append(2)
    elif algo_winner == 'draw':
        allwins[algo_winner].append(stat[connectid])
        for gs in cluster_groups[random.choice(cluster_algo_names)]:
            r3.append(gs)
        #continue
        labels.append(3)

    if connectid in stat:
        result_ = stat[connectid].values()

        # Convert object to a list
        data_ = list(result_)
        modelstatspy.append(data_)
        # Convert list to an array
        numpyArray = np.array(data_)
        modelstats[counter] = numpyArray
        connectids.append(connectid)
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

rs.append(("Mixed_best", list(r3)))
markovwins = pd.DataFrame(markovwins)
hierarchicalwins = pd.DataFrame(hierarchicalwins)
draws = pd.DataFrame(draws)

for keys in allwins.keys():
    allwins[keys] = pd.DataFrame(allwins[keys])

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


for (columnname, columndata) in markovwins.iteritems():
    #temp_df = pd.DataFrame(columns=[columnname+"_Draw", columnname+"_Markov", columnname+"_Hierar"])
    temp_df = pd.DataFrame(columns=['variable', 'value'])
    for key in allwins.keys():
        if key == 'draw':
            continue
        if columnname not in allwins[key]:
            continue
        temp = pd.DataFrame(allwins[key][columnname]).rename(columns={columnname: columnname+f"_{key}"}).melt()
        temp_df = pd.concat([temp_df, temp])
    #temp_df['value'] = temp_df['value'].astype(float)
    #melted_adjusted = temp_df[~is_outlier(temp_df['value'], 4.0)]
    plot = sns.histplot(temp_df, x='value', hue='variable', multiple='dodge', shrink=.75, bins=20 )
    #ax = temp_df.plot.hist(bins=20, alpha=0.5)
    plt.show()

#labels = np.array(labels)
#print(modelstats.shape, labels.shape)
#print(f"Amount of records per classlabel:{Counter(labels)}")
#
#modelstats = np.array(modelstatspy)
#print(modelstats.shape, labels.shape)
#
#X_train, X_test, Y_train, Y_test = train_test_split(modelstats, labels, test_size=0.3, stratify=labels)
#
#skb = SelectKBest(chi2, k=feature_count)
#new_modelstats = skb.fit_transform(X_train, Y_train)
#supp = skb.get_support(indices=True)
#print(supp)
#
#model = LogisticRegression()  # , multi_class='multinomial')
#print(f"Amount of records per classlabel in the trainingset:{Counter(Y_train)}")
#from imblearn.over_sampling import SMOTE
#
#sm = SMOTE(random_state=42)
#X_res, y_res = sm.fit_resample(X_train, Y_train)
#print(Counter(y_res))
#model = model.fit(X_res, y_res)
#
#output2 = model.predict(X_test)
#
#acc_score = accuracy_score(Y_test, output2)
#cfmatrix = confusion_matrix(Y_test, output2)
#result['model'] = {}
#result['model']['selected_features'] = supp.tolist()
#result['model']['records_per_class'] = str(Counter(labels))
#result['model']['accuracy_test_score'] = acc_score
#result['model']['intercept'] = model.intercept_.tolist()
#result['model']['coefficients'] = model.coef_.tolist()
#result['model']['predicted_test_classes'] = str(Counter(output2))
#result['model']['confusion_matrix'] = cfmatrix.tolist()
#print(accuracy_score(Y_test, output2))
#print(cfmatrix)
#print(model.intercept_)
#print(model.coef_)
#print(model.classes_)
#print(f"Amount of predicted records per classlabel:{Counter(output2)}")
#
#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#n_scores = cross_val_score(model, modelstats, labels, scoring='accuracy', cv=cv, n_jobs=-1)
#print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
#
#
#all_predictions = model.predict(modelstats)
#acc_score = accuracy_score(labels, all_predictions)
#for i in range(len(all_predictions)):
#    label = all_predictions[i]
#    conid = connectids[i]
#    if label == 0:
#        columnname = connected_col
#    elif label == 1:
#        columnname = markov_col
#    elif label == 2:
#        columnname = hierar_col
#    elif label == 3:
#        columnname = markov_col
#
#    clusters_ = list(res[res[connected_col] == conid].groupby([columnname]).groups.values())
#    for gs in clusters_:
#        r4.append(gs)

#rs.append(("Predicted clustering", list(r4)))
for r in rs:
    print(f"Clustering method:{r[0]}")
    result[r[0]] = {}
    for eva in evaluations:
        result[r[0]][eva] = evaluate(r[1], s, eva)
        print(f"{eva}: {result[r[0]][eva]:.4f}")
    print("----------------------------\n")
print("Markov own:")
evaluate_(groundtruth, res, res.groupby([markov_col]).indices, amount, markov_col)
print("Hierar own:")
evaluate_(groundtruth, res, res.groupby([hierar_col]).indices, amount, hierar_col)

with open(os.path.join('./testruns/' + str(int(time.time())) + '.json'), 'w') as outfile:
    json.dump(result, outfile)

print("done")
