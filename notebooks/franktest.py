import itertools
import json
import operator
import random
import time
import os

import numpy as np
import pandas as pd
from deduplipy.analyzing.cluster_method_prediction import get_cluster_column_name, get_mixed_best, \
    predictions_to_clusters, learn_ensemble_weights
from deduplipy.clustering.clustering import *
from deduplipy.datasets import load_data
from deduplipy.deduplicator import Deduplicator
from deduplipy.blocking import first_letter, first_three_letters, first_four_letters_no_space
from deduplipy.evaluation.pairwise_evaluation import perform_evaluation
from deduplipy.string_metrics import three_gram
from deduplipy.config import MIN_PROBABILITY
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

dataset = 'cora'
learning = False
pairs = None
pairs_name = None
save_intermediate = False
pickle_name = None
groupby_name = None
field_info = None
mes = None

if dataset == 'musicbrainz20k':
    df = load_data(kind='musicbrainz20k')
    groupby_name = 'CID'
    group = df.groupby([groupby_name])  # CID for musicbrainz
    groundtruth = group.indices
    field_info = {'title': [three_gram]}
    myDedupliPy = Deduplicator(['title', 'artist', 'album'], rules={'album': [first_letter]}, field_info=field_info)
    myDedupliPy.verbose = True
    pickle_name = 'musicbrainz20kfulltest2.pkl'
    if learning:
        myDedupliPy.fit(df)
        with open(pickle_name, 'wb') as f:
            pickle.dump(myDedupliPy, f)
        myDedupliPy.save_intermediate_steps = save_intermediate
    else:
        with open(pickle_name, 'rb') as f:
            myDedupliPy = pickle.load(f)
            myDedupliPy.save_intermediate_steps = save_intermediate
    pairs_name = "scored_pairs_table_musicbrainz20k_full.csv"
    pairs = pd.read_csv(os.path.join('./', pairs_name), sep="|")

elif dataset == 'musicbrainz200k':
    df = load_data(kind='musicbrainz200k')
    groupby_name = 'CID'
    group = df.groupby([groupby_name])  # CID for musicbrainz
    groundtruth = group.indices
    myDedupliPy = Deduplicator(['title', 'artist', 'album'], rules={'album': [first_three_letters]})
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
    pickle_name = 'musicbrainz20kcustomblocking_single_first_letter.pkl'
    if learning:
        myDedupliPy.save_intermediate_steps = save_intermediate
        myDedupliPy.fit(df)
        with open(pickle_name, 'wb') as f:
            pickle.dump(myDedupliPy, f)
    else:
        with open(pickle_name, 'rb') as f:
            myDedupliPy = pickle.load(f)
            myDedupliPy.save_intermediate_steps = save_intermediate
    pairs_name = "scored_pairs_table_musicbrainz20k_single_oneletterblocking.csv"
    pairs = pd.read_csv(os.path.join('./', pairs_name), sep="|")

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
    myDedupliPy.verbose = True
    pairs_name = "scored_pairs_table_stoxx50.csv"
    pairs = pd.read_csv(os.path.join('./', pairs_name), sep="|")

elif dataset == 'voters':
    df = load_data(kind='voters5m')
    groupby_name = 'recid'
    group = df.groupby([groupby_name])
    groundtruth = group.indices
    myDedupliPy = Deduplicator(['givenname', 'surname', 'suburb', 'postcode'], rules={'givenname': [first_three_letters], 'surname': [first_three_letters]})
    myDedupliPy.verbose = True
    if learning:
        myDedupliPy.fit(df)
        with open('voters5m.pkl', 'wb') as f:
            pickle.dump(myDedupliPy, f)
    else:
        with open('voters5m.pkl', 'rb') as f:
            myDedupliPy = pickle.load(f)
            myDedupliPy.save_intermediate_steps = save_intermediate

elif dataset == 'cora':
    df = load_data(kind='cora')
    groupby_name = ''
    group = df.groupby([groupby_name])
    groundtruth = group.indices
    myDedupliPy = Deduplicator()
    myDedupliPy.verbose = True
    pickle_name = ''
    if learning:
        myDedupliPy.save_intermediate_steps = save_intermediate
        myDedupliPy.fit(df)
        with open(pickle_name, 'wb') as f:
            pickle.dump(myDedupliPy, f)
    else:
        with open(pickle_name, 'rb') as f:
            myDedupliPy = pickle.load(f)
            myDedupliPy.save_intermediate_steps = save_intermediate
    #pairs_name = "scored_pairs_table_musicbrainz20k_single_oneletterblocking.csv"
    #pairs = pd.read_csv(os.path.join('./', pairs_name), sep="|")
else:
    print("unknown")
    exit(0)

amount = len(df)


markov_col = get_cluster_column_name(markov_clustering.__name__)
hierar_col = get_cluster_column_name(hierarchical_clustering.__name__)
connected_col = get_cluster_column_name(connected_components.__name__)
score_threshes = [0.3]#, 0.35, 0.4, 0.45]
random_states = range(1)
config_options = list(itertools.product(score_threshes, random_states))
for config_option in config_options:
    score_thresh = config_option[0]
    random_state_number = config_option[1]
    mes = f"Running with scorethreshold, random_state of: {score_thresh}, {random_state_number} weighted, unweighted and multiple probabilities at once"
    feature_count = 15
    train_test_split_number = 0.2
    np.random.seed(random_state_number)
    cluster_algos = [connected_components, hierarchical_clustering, markov_clustering]#, optics, cdlib_ipca, cdlib_dcs, cdlib_pycombo, louvain, walktrap, greedy_modularity, cdlib_der, cdlib_scan, affinity_propagation]
    cluster_algo_names = [name.__name__ for name in cluster_algos]
    args = {hierarchical_clustering.__name__: {'cluster_threshold': 0.7, 'fill_missing': True},
            markov_clustering.__name__: {'inflation': 2},
            affinity_propagation.__name__: {'random_state': random_state_number},
            optics.__name__: {'min_samples': 2},
            'use_cc': True,
            'score_threshold': score_thresh,
            'feature_count': feature_count,
            'train_test_split': train_test_split_number,
            'random_state': random_state_number,
            'ensemble_cut_probability': MIN_PROBABILITY}
    if mes is None:
        mes = input("Please give a short description as to what this experiment entails")
    result = {'changes_description': mes,
              'config': args, 'dataset': dataset, 'scored_pairs_table': pairs_name, 'pickle_object_used': pickle_name,
              'split_version': 'features_split'}

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
    result['label_dict'] = label_dict
    #rs['mixed_best'] = []
    evaluations = ['precision', 'recall', 'f1', 'bmd', 'variation_of_information']
    print("----------------------------")


    eval_prios = {'adjusted_rand_score': 10, 'normalized_mutual_info_score': 20, 'fowlkes_mallows_score': 30, 'f1': 5, 'bmd': 6, 'variation_of_information': 7, 'recall': 9, 'precision': 8}
    result['eval_prios'] = eval_prios

    # perform pairwise evaluation on the entire clustering
    result |= perform_evaluation(rs, s)
    weights = {}
    for name in cluster_algo_names:
        weights[name] = result[name]['f1']

    groups_with_id, labels, mixed_best_array, connectids, ensemble_clusterings, obtained_weights = get_mixed_best(rs[connected_components.__name__], res, cluster_algos, label_dict, eval_prios, connected_col, groupby_name, colnames=myDedupliPy.col_names, random_state=random_state_number)
    result['weights'] = obtained_weights
    rs2 = {}
    rs2['mixed_best'] = mixed_best_array
    for key, value in ensemble_clusterings.items():
        rs2[key] = value
    result |= perform_evaluation(rs2, s)

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
    ### maybe split on all components with nodecount > 1 (to predict on) and split on all the components with size 1.
    ### this way we evaluate on the total clustering but only predict on components of size > 1.
    X_train, X_test, Y_train, Y_test, indices_train, indices_test = train_test_split(modelstats, labels, indices,
                                                                                     test_size=train_test_split_number,
                                                                                     #stratify=labels,
                                                                                     random_state=random_state_number)
    singleton_CC = res[connected_col].value_counts()
    singleton_CC = singleton_CC[singleton_CC == 1].index
    CCindices_train, CCindices_test = train_test_split(singleton_CC, test_size=train_test_split_number,random_state=random_state_number)
    indices_to_append = []
    for val in CCindices_test:
        indices_to_append.append(connectids.index(val))
    indices_test = np.concatenate((indices_test, np.array(indices_to_append)))

    skb = SelectKBest(chi2, k=feature_count)
    new_modelstats = skb.fit_transform(X_train, Y_train)
    supp = skb.get_support(indices=True)
    print(supp)

    model = LogisticRegression(random_state=random_state_number, class_weight='balanced')  # , multi_class='multinomial')
    print(f"Amount of records per classlabel in the trainingset:{sorted(Counter(Y_train).items())}")
    #from imblearn.over_sampling import SMOTE
    #
    #sm = SMOTE(random_state=random_state_number)
    #X_res, y_res = sm.fit_resample(X_train, Y_train)
    #print(sorted(Counter(y_res).items()))
    model = model.fit(X_train, Y_train)

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

    #gt_ = list(res[res[connected_col].isin(connectids)].groupby([groupby_name]).groups.values())

    print("ONLY ON THE TEST SPLIT AFTER THIS LINE -----------------------------------------\n")
    sub, gt = predictions_to_clusters(all_predictions, indices_test, connectids, res, label_dict, groups_with_id, cluster_algos, connected_col, groupby_name)
    # convert the
    rf = {}
    for key in sub.keys():
        rf["test_split_" + key] = sub[key]

    # perform pairwise evaluation on the clusterings on the selected connected components by the test/train split
    result |= perform_evaluation(rf, gt)

    with open(os.path.join('./testruns/' + str(int(time.time())) + '.json'), 'w') as outfile:
        json.dump(result, outfile)

    print("done")
