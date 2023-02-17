import json
import operator
import random
import time
import os

import numpy as np
import pandas as pd
from entity_resolution_evaluation.evaluation import evaluate
from deduplipy.clustering.clustering import markov_clustering, hierarchical_clustering, connected_components
from deduplipy.datasets import load_data
from deduplipy.deduplicator import Deduplicator
from deduplipy.blocking import first_letter
from deduplipy.evaluation.pairwise_evaluation import perform_evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter

from numpy import mean
from numpy import std

import pickle

dataset = 'musicbrainz20k'
learning = False
pairs = None
pairs_name = None
save_intermediate = False
pickle_name = None
groupby_name = None

if dataset == 'musicbrainz20k':
    df = load_data(kind='musicbrainz20k')
    groupby_name = 'CID'
    group = df.groupby([groupby_name])  # CID for musicbrainz
    groundtruth = group.indices
    myDedupliPy = Deduplicator(['title', 'artist', 'album'], rules={'album': [first_letter]})
    myDedupliPy.verbose = True
    pickle_name = 'musicbrainz20kcustomblocking.pkl'
    if learning:
        myDedupliPy.fit(df)
        with open(pickle_name, 'wb') as f:
            pickle.dump(myDedupliPy, f)
    else:
        with open(pickle_name, 'rb') as f:
            myDedupliPy = pickle.load(f)
            myDedupliPy.save_intermediate_steps = save_intermediate
    pairs_name = "scored_pairs_table_custom_blocking.csv"
    pairs = pd.read_csv(os.path.join('./', pairs_name), sep="|")

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
            myDedupliPy.save_intermediate_steps = save_intermediate
    # pairs_name = ".csv"
    # pairs = pd.read_csv(os.path.join('./', pairs_name), sep="|")

elif dataset == 'stoxx50':
    df = load_data(kind='stoxx50')
    groupby_name = 'id'
    group = df.groupby([groupby_name])
    groundtruth = group.indices
    myDedupliPy = Deduplicator(['name'])
    myDedupliPy.verbose = True
    pickle_name = 'stoxx50.pkl'
    if learning:
        myDedupliPy.fit(df)
        with open('stoxx50extrapyminhash.pkl', 'wb') as f:
            pickle.dump(myDedupliPy, f)
    else:
        with open(pickle_name, 'rb') as f:
            myDedupliPy = pickle.load(f)
            myDedupliPy.save_intermediate_steps = save_intermediate
    pairs_name = "scored_pairs_table_stoxx50.csv"
    pairs = pd.read_csv(os.path.join('./', pairs_name), sep="|")

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
            myDedupliPy.save_intermediate_steps = save_intermediate
else:
    print("unknown")
    exit(0)

amount = len(df)


def get_cluster_column_name(clusteringalgorithm) -> str:
    if callable(clusteringalgorithm):
        return 'deduplication_id_' + clusteringalgorithm.__name__
    elif isinstance(clusteringalgorithm, str):
        return 'deduplication_id_' + clusteringalgorithm
    else:
        pass


markov_col = 'deduplication_id_' + markov_clustering.__name__
hierar_col = 'deduplication_id_' + hierarchical_clustering.__name__
connected_col = 'deduplication_id_' + connected_components.__name__
score_thresh = 0.4
feature_count = 15
train_test_split_number = 0.3
random_state_number = 100
cluster_algos = [connected_components, hierarchical_clustering, markov_clustering]
cluster_algo_names = [name.__name__ for name in cluster_algos]
args = {hierarchical_clustering.__name__: {'cluster_threshold': 0.7, 'fill_missing': True},
        markov_clustering.__name__: {'inflation': 2}, 'use_cc': True,
        'score_threshold': score_thresh,
        'feature_count': feature_count,
        'train_test_split': train_test_split_number,
        'random_state': random_state_number}
res, stat = myDedupliPy.predict(df, clustering=cluster_algos, old_scored_pairs=pairs, score_threshold=score_thresh,
                                args=args)

rs = {}
label_dict = {}
for i in range(len(cluster_algos)):
    algo = cluster_algos[i]
    col = get_cluster_column_name(algo)
    rs[algo.__name__] = list(res.groupby([col]).groups.values())
    label_dict[algo.__name__] = i

label_dict['draw'] = len(label_dict)
r4 = []
s = list(group.groups.values())

rs['mixed_best'] = []
evaluations = ['precision', 'recall', 'f1', 'bmd', 'variation_of_information']
result = {'changes_description': input("Please give a short description as to what this experiment entails"),
          'config': args, 'dataset': dataset, 'scored_pairs_table': pairs_name, 'pickle_object_used': pickle_name}
print("----------------------------")

allwins = {'draw': []}
for names in cluster_algo_names:
    allwins[names] = []
#xfxf = list(stat)[0]
#modelstats = np.empty((len(stat), len(stat[xfxf])))
connectids = []
modelstatspy = []
labels = []
counter = 0


def select_winner(evaluation_metrics_results: dict, evaluation_metrics_importance: dict, algo_names: list) -> str:
    evaluation_metrics_importance = dict(sorted(evaluation_metrics_importance.items(), key=operator.itemgetter(1)))
    algo_names = [name.__name__ for name in algo_names]
    losers = []
    winners = None
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
        returnval = 'markov_clustering' if 'markov_clustering' in winners else random.choice(list(winners))
    return returnval


cluster_groups_with_id = {'mixed_best': {}}
for cluster_algo in cluster_algos:
    algorith_name = cluster_algo.__name__
    cluster_groups_with_id[algorith_name] = {}
eval_prios = {'f1': 1, 'bmd': 2, 'variation_of_information': 3, 'recall': 4, 'precision': 5}
# loop over the results of the connected components
for g in rs[connected_components.__name__]:
    # get the rows from the dataframe belonging to that connected component
    rows = res.loc[g]
    connectid = rows[connected_col].iloc[0]
    # if there is only one row for this connected component then we don't have to pick a clustering and just use the connected component.
    if len(rows) < 2:
        connectids.append(connectid)
        rs['mixed_best'].append(list(rows.groupby([connected_col]).groups.values())[0])
        continue
    # get the ground truth clusters for this connected component
    gt = list(rows.groupby([groupby_name]).groups.values())
    temp = {}
    cluster_groups = {}
    # loop over the cluster algorithms to get the resulting clusters per connected component.
    for cluster_algo in cluster_algos:
        algorith_name = cluster_algo.__name__
        cluster_groups[algorith_name] = list(rows.groupby([get_cluster_column_name(cluster_algo)]).groups.values())
        cluster_groups_with_id[algorith_name][connectid] = list(
            rows.groupby([get_cluster_column_name(cluster_algo)]).groups.values())
        for eva in evaluations:
            if eva not in temp:
                temp[eva] = {}
            temp[eva][algorith_name] = evaluate(cluster_groups[algorith_name], gt, eva)
    # get the algo winner and add the features of this connected component to the name of the winner
    algo_winner = select_winner(temp, eval_prios, cluster_algos)
    allwins[algo_winner].append(stat[connectid])
    # add the label of the winner to the labels array
    labels.append(label_dict[algo_winner])

    # if it was a draw, there are no
    if algo_winner == 'draw':
        algo_winner = cluster_algo_names[0]
    for gs in cluster_groups[algo_winner]:
        rs['mixed_best'].append(gs)
    cluster_groups_with_id['mixed_best'][connectid] = cluster_groups[algo_winner]

    if connectid in stat:
        result_ = stat[connectid].values()

        # Convert object to a list
        data_ = list(result_)
        modelstatspy.append(data_)
        # Convert list to an array
        #numpyArray = np.array(data_)
        #modelstats[counter] = numpyArray
        connectids.append(connectid)
        #counter += 1


for keys in allwins.keys():
    allwins[keys] = pd.DataFrame(allwins[keys])

labels = np.array(labels)
modelstats = np.array(modelstatspy)
print(f"Amount of records per classlabel:{sorted(Counter(labels).items())}")
print(modelstats.shape, labels.shape)

indices = np.arange(len(modelstats))
X_train, X_test, Y_train, Y_test, indices_train, indices_test = train_test_split(modelstats, labels, indices,
                                                                                 test_size=train_test_split_number,
                                                                                 stratify=labels,
                                                                                 random_state=random_state_number)

skb = SelectKBest(chi2, k=feature_count)
new_modelstats = skb.fit_transform(X_train, Y_train)
supp = skb.get_support(indices=True)
print(supp)

model = LogisticRegression(random_state=random_state_number)  # , multi_class='multinomial')
print(f"Amount of records per classlabel in the trainingset:{sorted(Counter(Y_train).items())}")
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=random_state_number)
X_res, y_res = sm.fit_resample(X_train, Y_train)
print(sorted(Counter(y_res).items()))
model = model.fit(X_res, y_res)

output2 = model.predict(X_test)

acc_score = accuracy_score(Y_test, output2)
cfmatrix = confusion_matrix(Y_test, output2)
result['model'] = {}
result['model']['selected_features'] = supp.tolist()
result['model']['records_per_class'] = str(sorted(Counter(labels).items()))
result['model']['accuracy_test_score'] = acc_score
result['model']['intercept'] = model.intercept_.tolist()
result['model']['coefficients'] = model.coef_.tolist()
result['model']['predicted_test_classes'] = str(sorted(Counter(output2).items()))
result['model']['confusion_matrix'] = cfmatrix.tolist()
print(accuracy_score(Y_test, output2))
print(cfmatrix)
print(model.intercept_)
print(model.coef_)
print(model.classes_)
print(f"Amount of predicted records per classlabel:{sorted(Counter(output2).items())}")

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=random_state_number)
n_scores = cross_val_score(model, modelstats, labels, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

all_predictions = output2  # model.predict(X_test)
acc_score = accuracy_score(Y_test, all_predictions)


def predictions_to_clusters(label_predictions, data_indices, connected_component_ids, total_clustering_result, labels_dictionary):
    my_gt = []
    sub = {}
    sub['mixed_best'] = []
    sub['predicted_clustering'] = []
    for cluster_algo in cluster_algos:
        algorith_name = cluster_algo.__name__
        sub[algorith_name] = []

    for i in range(len(label_predictions)):
        label = label_predictions[i]
        conid = connected_component_ids[data_indices[i]]
        algo_name = [i for i in labels_dictionary if labels_dictionary[i] == label][0]
        if algo_name == 'draw':
            algo_name = cluster_algo_names[0]
        columnname = get_cluster_column_name(algo_name)
        clusters_ = list(total_clustering_result[total_clustering_result[connected_col] == conid].groupby([columnname]).groups.values())
        for gs in clusters_:
            sub['predicted_clustering'].append(gs)
        gt_clusters = list(total_clustering_result[total_clustering_result[connected_col] == conid].groupby([groupby_name]).groups.values())
        for gt_ in gt_clusters:
            my_gt.append(gt_)

        for key in sub.keys():
            if key not in cluster_groups_with_id:
                continue
            for x in cluster_groups_with_id[key][conid]:
                sub[key].append(x)
    # sub is a dictionary with as key the algorithm name and value the list of clusterings
    return sub, my_gt
# rs.append(("Predicted clustering", list(r4)))
sub, gt = predictions_to_clusters(all_predictions, indices_test, connectids, res, label_dict)
# perform pairwise evaluation on the entire clustering
result |= perform_evaluation(rs, s)

print("ONLY ON THE TEST SPLIT AFTER THIS LINE -----------------------------------------\n")
# convert the
rf = {}
for key in sub.keys():
    rf["test_split_" + key] = sub[key]

# perform pairwise evaluation on the clusterings on the selected connected components by the test/train split
result |= perform_evaluation(rf, gt)

with open(os.path.join('./testruns/' + str(int(time.time())) + '.json'), 'w') as outfile:
    json.dump(result, outfile)

print("done")
