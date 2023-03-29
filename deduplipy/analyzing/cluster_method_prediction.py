import itertools
import operator
import random
import numpy
import numpy as np
from sklearn.cluster import AffinityPropagation, MeanShift, MiniBatchKMeans
from sklearn.linear_model import LinearRegression
from scipy.sparse.csgraph import connected_components
from deduplipy.analyzing.metrics_collection import perform_scoring
from deduplipy.clustering.ensemble_clustering import ClusterSimilarityMatrix
from deduplipy.config import DEDUPLICATION_ID_NAME, MIN_PROBABILITY
from deduplipy.clustering.clustering import markov_clustering_on_adjacency
from entity_resolution_evaluation.evaluation import evaluate


def get_cluster_column_name(clusteringalgorithm) -> str:
    if callable(clusteringalgorithm):
        return '_'.join([DEDUPLICATION_ID_NAME, clusteringalgorithm.__name__])
    elif isinstance(clusteringalgorithm, str):
        return '_'.join([DEDUPLICATION_ID_NAME, clusteringalgorithm])
    else:
        pass


def select_winner(evaluation_metrics_results: dict, evaluation_metrics_importance: dict, algo_names: list, random_state_per_component: int) -> str:
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
        winners = sorted(winners)
        if len(winners) == 1:
            # we have a clear winner on a metric, therefor we return this winner
            return win
        elif len(winners) != len(evaluation_metrics_results[metricName]):
            # we didn't have a winner, but we do have a loser somewhere, so remove the loser from the list to check.
            for clust_name in algo_names:
                if clust_name not in winners and clust_name not in losers:
                    losers.append(clust_name)

    #if len(winners) == len(algo_names):
    #    returnval = 'draw'
    #else:
    #    r = np.random.RandomState(random_state_per_component)
    #    returnval = r.choice(list(winners))
    r = np.random.RandomState(random_state_per_component)
    returnval = r.choice(list(winners))
    #if random_state_per_component < 100:
    #    print(f"random state (cc id): {random_state_per_component}\nWinners: {winners}\nChosen winner: {returnval}\n")
    return returnval



def get_mixed_best(connected_components_clusters, total_resulting_clusters, cluster_algorithms, label_diction, eval_priorities, connected_col, groundtruth_name, evaluations=None, scoring=None, labelless_scoring=None, colnames=None, weights=None, random_state=42):
    if evaluations is None:
        evaluations = ['precision', 'recall', 'f1', 'bmd', 'variation_of_information']
    if scoring is None:
        scoring = ['adjusted_rand_score','normalized_mutual_info_score','fowlkes_mallows_score']
    if labelless_scoring is None:
        labelless_scoring = ['silhouette_score']
    labels = []
    mixed_best = []
    ensemble = {}

    learn_weights = True
    cluster_groups_with_id = {'mixed_best': {}}
    for probs in MIN_PROBABILITY:
        ensemble['ensemble_no_weight_'+str(probs)] = []
        ensemble['ensemble_weighted_' + str(probs)] = []
        cluster_groups_with_id['ensemble_no_weight_'+str(probs)] = {}
        cluster_groups_with_id['ensemble_weighted_' + str(probs)] = {}
    cluster_algo_names = [name.__name__ for name in cluster_algorithms]
    component_ids = []
    for cluster_algo in cluster_algorithms:
        algorith_name = cluster_algo.__name__
        cluster_groups_with_id[algorith_name] = {}

    if learn_weights:
        counter = 200
        weights = learn_ensemble_weights(connected_components_clusters, counter, groundtruth_name, total_resulting_clusters, cluster_algorithms, random_state)

    # loop over the results of the connected components to label which algorithm wins the connected component for the model
    for g in connected_components_clusters:
        # get the rows from the dataframe belonging to that connected component
        rows = total_resulting_clusters.loc[g]
        connectid = rows[connected_col].iloc[0]
        component_ids.append(connectid)
        # get the ground truth clusters for this connected component
        gt = list(rows.groupby([groundtruth_name]).groups.values())
        temp = {}
        labelless_temp = {}
        cluster_groups = {}
        clt_sim_matrix = ClusterSimilarityMatrix()
        clt_sim_matrix_unweighted = ClusterSimilarityMatrix()
        # loop over the cluster algorithms to get the resulting clusters per connected component.
        for cluster_algo in cluster_algorithms:
            algorith_name = cluster_algo.__name__
            cluster_groups[algorith_name] = list(rows.groupby([get_cluster_column_name(cluster_algo)]).groups.values())
            cluster_groups_with_id[algorith_name][connectid] = list(rows.groupby([get_cluster_column_name(cluster_algo)]).groups.values())
            for eva in evaluations:
                if eva not in temp:
                    temp[eva] = {}
                temp[eva][algorith_name] = evaluate(cluster_groups[algorith_name], gt, eva)
            if scoring is not None:
                for score in scoring:
                    if score not in temp:
                        temp[score] = {}
                    true_labels = rows[groundtruth_name].values
                    pred_labels = rows[get_cluster_column_name(algorith_name)].values
                    temp[score][algorith_name] = perform_scoring(score, pred_labels, true_labels)
            #if labelless_scoring is not None:
            #    for labelless in labelless_scoring:
            #        if labelless not in labelless_temp:
            #            labelless_temp[labelless] = {}
            #        clust_labels = rows[get_cluster_column_name(algorith_name)].values
            #        data_values = rows[colnames]
            #        labelless_temp[labelless][algorith_name] = perform_scoring(labelless, data_values, clust_labels)
            if weights is not None:
                w = weights[algorith_name]
            else:
                w = 1
            clt_sim_matrix.fit(rows[get_cluster_column_name(algorith_name)].values, w)
            clt_sim_matrix_unweighted.fit(rows[get_cluster_column_name(algorith_name)].values, 1)

        sim_matrix = clt_sim_matrix.similarity
        sim_matrix_unweighted = clt_sim_matrix_unweighted.similarity

        norm_sim_matrix = sim_matrix / sim_matrix.diagonal()
        norm_sim_matrix_unweighted = sim_matrix_unweighted / sim_matrix_unweighted.diagonal()
        rows, ensemble, cluster_groups_with_id = ensemble_matrix_to_clusters(norm_sim_matrix, connectid, rows, ensemble, cluster_groups_with_id, 'ensemble_weighted_')
        rows, ensemble, cluster_groups_with_id = ensemble_matrix_to_clusters(norm_sim_matrix_unweighted, connectid, rows, ensemble,cluster_groups_with_id, 'ensemble_no_weight_')

        # if there is only one row for this connected component then we don't have to pick a clustering and just use the connected component.
        if len(rows) < 2:
            resul = list(rows.groupby([connected_col]).groups.values())
            for gs_ in resul:
                mixed_best.append(gs_)
            cluster_groups_with_id['mixed_best'][connectid] = resul
            continue

        # get the algo winner and add the features of this connected component to the name of the winner
        algo_winner = select_winner(temp, eval_priorities, cluster_algorithms, connectid)
        # add the label of the winner to the labels array
        labels.append(label_diction[algo_winner])

        # if it was a draw, there are no groups associated with that name, so pick one
        if algo_winner == 'draw':
            algo_winner = cluster_algo_names[0]
        for gs in cluster_groups[algo_winner]:
            mixed_best.append(gs)
        cluster_groups_with_id['mixed_best'][connectid] = cluster_groups[algo_winner]

    return cluster_groups_with_id, labels, mixed_best, component_ids, ensemble, weights


def ensemble_matrix_to_clusters(normalized_similarity_matrix, connect_id, rows, ensemble, cluster_groups_with_id, name_start):
    for probability in MIN_PROBABILITY:
        graph = (normalized_similarity_matrix > probability).astype(int)

        n_clusters, y_ensemble = connected_components(graph, directed=False, return_labels=True)
        # y_ensemble = markov_clustering_on_adjacency(norm_sim_matrix, pruning_value=0.1)
        ensemble_ids = [x + connect_id for x in y_ensemble]
        rows[DEDUPLICATION_ID_NAME + "_" + name_start + str(probability)] = ensemble_ids
        ens = list(rows.groupby([DEDUPLICATION_ID_NAME + "_" + name_start + str(probability)]).groups.values())
        for gs_ in ens:
            ensemble[name_start + str(probability)].append(gs_)
        cluster_groups_with_id[name_start + str(probability)][connect_id] = ens

    return rows, ensemble, cluster_groups_with_id

def predictions_to_clusters(label_predictions, data_indices, connected_component_ids, total_clustering_result, labels_dictionary, _cluster_groups_with_id, cluster_algos, connected_col, groundtruth_name):
    my_gt = []
    sub = {}
    for key in _cluster_groups_with_id.keys():
        sub[key] = []
    sub['predicted_clustering'] = []

    cluster_algo_names = [name.__name__ for name in cluster_algos]

    for i in range(len(data_indices)):
        conid = connected_component_ids[data_indices[i]]
        if i < len(label_predictions):
            label = label_predictions[i]
            algo_name = [i for i in labels_dictionary if labels_dictionary[i] == label][0]
            if algo_name == 'draw':
                algo_name = cluster_algo_names[0]
            columnname = get_cluster_column_name(algo_name)
            clusters_ = list(total_clustering_result[total_clustering_result[connected_col] == conid].groupby([columnname]).groups.values())
            for gs in clusters_:
                sub['predicted_clustering'].append(gs)
        else:
            clusters_ = list(total_clustering_result[total_clustering_result[connected_col] == conid].groupby([connected_col]).groups.values())
            for gs in clusters_:
                sub['predicted_clustering'].append(gs)
        gt_clusters = list(total_clustering_result[total_clustering_result[connected_col] == conid].groupby([groundtruth_name]).groups.values())
        for gt_ in gt_clusters:
            my_gt.append(gt_)
        for key in sub.keys():
            if key not in _cluster_groups_with_id:
                continue
            for x in _cluster_groups_with_id[key][conid]:
                sub[key].append(x)
    # sub is a dictionary with as key the algorithm name and value the list of clusterings
    return sub, my_gt


def learn_ensemble_weights(connected_clusters, min_edge_count, name, total_resulting_clusters, cluster_algorithms, random_state):
    total_edgelist = {}
    labels = []
    for g in connected_clusters:
        if min_edge_count < len(labels):
            break

        edgelist = {}
        # get the rows from the dataframe belonging to that connected component
        rows = total_resulting_clusters.loc[g]
        # get the ground truth clusters for this connected component

        for cluster_algo in cluster_algorithms:
            algorith_name = cluster_algo.__name__
            clusters = list(rows.groupby([get_cluster_column_name(cluster_algo)]).groups.values())
            all_combs = []
            for cluster in clusters:
                combs = list(itertools.combinations(cluster, 2))
                all_combs += combs
            if algorith_name == 'connected_components':
                for edge in combs:
                    if edge not in edgelist:
                        edgelist[edge] = []
                    edgelist[edge].append(1)
            else:
                for key in edgelist.keys():
                    if key in all_combs:
                        edgelist[key].append(1)
                    else:
                        edgelist[key].append(0)
        total_edgelist |= edgelist
        gt = list(rows.groupby([name]).groups.values())
        all_gt = []
        for cluster in gt:
            combs = list(itertools.combinations(cluster, 2))
            all_gt += combs

        for key in edgelist.keys():
            if key in all_gt:
                labels.append(1)
            else:
                labels.append(0)

    modelstatspy = []
    for key, value in total_edgelist.items():
        # Convert object to a list
        data_ = list(value)
        modelstatspy.append(data_)
    modelstats = np.array(modelstatspy)
    labels = np.array(labels)
    final_labels_true = np.where(labels == 1)[0]
    final_labels_false = np.where(labels == 0)[0]

    if len(final_labels_true) > len(final_labels_false):
        random.Random(random_state).shuffle(final_labels_true)
        final_labels_true = final_labels_true[:len(final_labels_false)]
    else:
        random.Random(random_state).shuffle(final_labels_false)
        final_labels_false = final_labels_false[:len(final_labels_true)]
    final_label_indices = np.concatenate((final_labels_true, final_labels_false), axis=None)
    actual_labels = []
    actual_edgelist = []
    for label_index in final_label_indices:
        actual_labels.append(labels[label_index])
        actual_edgelist.append(modelstats[label_index])
    reg = LinearRegression().fit(actual_edgelist, actual_labels)
    score = reg.score(actual_edgelist, actual_labels)
    print(f"The models score is: {score}")
    coefs = reg.coef_
    coef_per_name = {}
    teller = 0
    for cluster_algo in cluster_algorithms:
        algorith_name = cluster_algo.__name__
        coef_per_name[algorith_name] = coefs[teller]
        teller += 1

    print(f"Coefficients per name in the linear regression:{coef_per_name}")
    return coef_per_name
