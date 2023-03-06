import json
import operator
import random
import time
import os

import numpy as np
import pandas as pd
from entity_resolution_evaluation.evaluation import evaluate

from deduplipy.analyzing.cluster_method_prediction import predictions_to_clusters, get_mixed_best, \
    get_cluster_column_name
from deduplipy.clustering.clustering import markov_clustering, hierarchical_clustering, connected_components, \
    affinity_propagation, optics
from deduplipy.datasets import load_data
from deduplipy.deduplicator import Deduplicator
from deduplipy.blocking import first_letter, first_three_letters
from deduplipy.evaluation.pairwise_evaluation import perform_evaluation
from deduplipy.config import DEDUPLICATION_ID_NAME
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

dataset = 'musicbrainz20k_single'
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
    #pairs_train = pd.read_csv(os.path.join('./', "train_scored_pairs_table_musicbrainz_state_100.csv"), sep="|")
    #pairs_test = pd.read_csv(os.path.join('./', "test_scored_pairs_table_musicbrainz_state_100.csv"), sep="|")

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
elif dataset == 'musicbrainz20k_single':
    dataset_count = 20
    df = load_data(kind='musicbrainz20k_single', count=dataset_count)
    dataset = f'{dataset}_{dataset_count}'
    groupby_name = 'CID'
    group = df.groupby([groupby_name])  # CID for musicbrainz
    groundtruth = group.indices
    myDedupliPy = Deduplicator(['value'], rules={'value': [first_letter]})
    myDedupliPy.verbose = True
    pickle_name = 'musicbrainz20kcustomblocking_single_one_letter.pkl'
    if learning:
        myDedupliPy.fit(df)
        with open(pickle_name, 'wb') as f:
            pickle.dump(myDedupliPy, f)
    else:
        with open(pickle_name, 'rb') as f:
            myDedupliPy = pickle.load(f)
            myDedupliPy.save_intermediate_steps = save_intermediate
    # pairs_name = "scored_pairs_table_custom_blocking.csv"
    # pairs = pd.read_csv(os.path.join('./', pairs_name), sep="|")
    #pairs_train = pd.read_csv(os.path.join('./', 'train_scored_pairs_table_musicbrainz200_threeletter.csv'), sep="|")
    #pairs_test = pd.read_csv(os.path.join('./', 'test_scored_pairs_table_musicbrainz200_threeletter.csv'), sep="|")
    pairs_train = None
    pairs_test = None

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

markov_col = get_cluster_column_name(markov_clustering.__name__)
hierar_col = get_cluster_column_name(hierarchical_clustering.__name__)
connected_col = get_cluster_column_name(connected_components.__name__)
score_thresh = 0.4
feature_count = 15
train_test_split_number = 0.3
random_state_number = 1000
cluster_algos = [connected_components, markov_clustering, hierarchical_clustering, optics, affinity_propagation]
cluster_algo_names = [name.__name__ for name in cluster_algos]
args = {hierarchical_clustering.__name__: {'cluster_threshold': 0.7, 'fill_missing': True},
        markov_clustering.__name__: {'inflation': 2},
        affinity_propagation.__name__: {'random_state': random_state_number},
        optics.__name__: {'min_samples': 2},
        'use_cc': True,
        'score_threshold': score_thresh,
        'feature_count': feature_count,
        'train_test_split': train_test_split_number,
        'random_state': random_state_number}
indices = np.arange(len(df))
X_train, X_test = train_test_split(df, shuffle=True, test_size=train_test_split_number,
                                   random_state=random_state_number)
# pairs = None
pairs_train = None
pairs_test = None
# get the resulting clusters of each algorithm on the dataset. Also gets the stats of each connected component
result = {'changes_description': input("Please give a short description as to what this experiment entails"),
          'config': args, 'dataset': dataset, 'scored_pairs_table': pairs_name, 'pickle_object_used': pickle_name,
          'split_version': 'data_split'}

res, stat = myDedupliPy.predict(X_train, clustering=cluster_algos, old_scored_pairs=pairs_train,
                                score_threshold=score_thresh, suffix="train_", args=args)
res_test, stat_test = myDedupliPy.predict(X_test, clustering=cluster_algos, old_scored_pairs=pairs_test,
                                          score_threshold=score_thresh, suffix="test_", args=args)

rs = {}
label_dict = {}
for i in range(len(cluster_algos)):
    algo = cluster_algos[i]
    col = get_cluster_column_name(algo)
    rs[algo.__name__] = list(res.groupby([col]).groups.values())
    label_dict[algo.__name__] = i

label_dict['draw'] = len(label_dict)
r4 = []
s = list(res.groupby([groupby_name]).groups.values())
evaluations = ['precision', 'recall', 'f1', 'bmd', 'variation_of_information']
result['label_dict'] = label_dict
print("----------------------------")

# connectids = res[connected_col].unique()

eval_prios = {'f1': 1, 'bmd': 2, 'variation_of_information': 3, 'recall': 4, 'precision': 5}

groups_with_id, labels, mixed_best_array, connectids = get_mixed_best(rs[connected_components.__name__], res,
                                                                      cluster_algos, label_dict, eval_prios,
                                                                      connected_col, groupby_name)
test_groups_with_id, test_winner_ids, test_mixed_best_array, connectids_test = get_mixed_best(
    list(res_test.groupby([connected_col]).groups.values()), res_test, cluster_algos, label_dict, eval_prios,
    connected_col, groupby_name)

rs['mixed_best'] = mixed_best_array
labels = np.array(labels)
print(f"Amount of records per classlabel:{sorted(Counter(labels).items())}")
modelstatspy = []
for key, value in stat.items():
    result_ = value.values()

    # Convert object to a list
    data_ = list(result_)
    modelstatspy.append(data_)
modelstats = np.array(modelstatspy)
print(modelstats.shape, labels.shape)

indices = np.arange(len(modelstats))

skb = SelectKBest(chi2, k=feature_count)
new_modelstats = skb.fit_transform(modelstats, labels)
supp = skb.get_support(indices=True)
print(supp)

model = LogisticRegression(random_state=random_state_number)  # , multi_class='multinomial')
print(f"Amount of records per classlabel in the trainingset:{sorted(Counter(labels).items())}")
from imblearn.over_sampling import SMOTE

#sm = SMOTE(random_state=random_state_number)
#X_res, y_res = sm.fit_resample(modelstats, labels)
#print(sorted(Counter(y_res).items()))
model = model.fit(modelstats, labels)
data_to_predict_on = []
for key, value in stat_test.items():
    result_ = value.values()

    # Convert object to a list
    data_ = list(result_)
    data_to_predict_on.append(data_)
output2 = model.predict(data_to_predict_on)

# acc_score = accuracy_score(Y_test, output2)
# cfmatrix = confusion_matrix(Y_test, output2)
result['model'] = {}
result['model']['selected_features'] = supp.tolist()
result['model']['records_per_class'] = str(sorted(Counter(labels).items()))
# result['model']['accuracy_test_score'] = acc_score
result['model']['intercept'] = model.intercept_.tolist()
result['model']['coefficients'] = model.coef_.tolist()
result['model']['predicted_test_classes'] = str(sorted(Counter(output2).items()))
# result['model']['confusion_matrix'] = cfmatrix.tolist()
# print(accuracy_score(Y_test, output2))
# print(cfmatrix)
print(model.intercept_)
print(model.coef_)
print(model.classes_)
print(f"Amount of predicted records per classlabel:{sorted(Counter(output2).items())}")

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=random_state_number)
n_scores = cross_val_score(model, modelstats, labels, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

all_predictions = output2  # model.predict(X_test)
# acc_score = accuracy_score(Y_test, all_predictions)


# perform pairwise evaluation on the entire clustering
result |= perform_evaluation(rs, s)

print("ONLY ON THE TEST SPLIT AFTER THIS LINE -----------------------------------------\n")
connectids_test_ = res_test[connected_col].unique()
gt_ = list(res_test.groupby([groupby_name]).groups.values())

indices_test = range(len(connectids_test))
sub, gt = predictions_to_clusters(all_predictions, indices_test, connectids_test, res_test, label_dict,
                                  test_groups_with_id, cluster_algos, connected_col, groupby_name)

# convert the
rf = {}
for key in sub.keys():
    rf["test_split_" + key] = sub[key]

# perform pairwise evaluation on the clusterings on the selected connected components by the test/train split
result |= perform_evaluation(rf, gt_)

with open(os.path.join('./testruns/' + str(int(time.time())) + '.json'), 'w') as outfile:
    json.dump(result, outfile)

print("done")
