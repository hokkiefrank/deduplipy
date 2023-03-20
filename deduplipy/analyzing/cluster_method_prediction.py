import operator
import random
import numpy as np
from sklearn.cluster import AffinityPropagation, MeanShift, MiniBatchKMeans
from scipy.sparse.csgraph import connected_components
from deduplipy.analyzing.metrics_collection import perform_scoring
from deduplipy.clustering.ensemble_clustering import ClusterSimilarityMatrix
from deduplipy.config import DEDUPLICATION_ID_NAME
from entity_resolution_evaluation.evaluation import evaluate

MIN_PROBABILITY = 0.75
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



def get_mixed_best(connected_components_clusters, total_resulting_clusters, cluster_algorithms, label_diction, eval_priorities, connected_col, groundtruth_name, evaluations=None, scoring=None, labelless_scoring=None, colnames=None):
    if evaluations is None:
        evaluations = ['precision', 'recall', 'f1', 'bmd', 'variation_of_information']
    if scoring is None:
        scoring = ['adjusted_rand_score','normalized_mutual_info_score','fowlkes_mallows_score']
    if labelless_scoring is None:
        labelless_scoring = ['silhouette_score']
    labels = []
    mixed_best = []
    ensemble = []
    cluster_groups_with_id = {'mixed_best': {},
                              'ensemble_clustering': {}}
    cluster_algo_names = [name.__name__ for name in cluster_algorithms]
    component_ids = []
    for cluster_algo in cluster_algorithms:
        algorith_name = cluster_algo.__name__
        cluster_groups_with_id[algorith_name] = {}
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
            clt_sim_matrix.fit(rows[get_cluster_column_name(algorith_name)].values)
        sim_matrix = clt_sim_matrix.similarity
        norm_sim_matrix = sim_matrix / sim_matrix.diagonal()
        graph = (norm_sim_matrix > MIN_PROBABILITY).astype(int)
        n_clusters, y_ensemble = connected_components(graph, directed=False, return_labels=True)
        ensemble_ids = [x+connectid for x in y_ensemble]
        rows[DEDUPLICATION_ID_NAME + "_ensemble"] = ensemble_ids
        ens = list(rows.groupby([DEDUPLICATION_ID_NAME + "_ensemble"]).groups.values())
        for gs_ in ens:
            ensemble.append(gs_)
        cluster_groups_with_id['ensemble_clustering'][connectid] = ens
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

    return cluster_groups_with_id, labels, mixed_best, component_ids, ensemble


def predictions_to_clusters(label_predictions, data_indices, connected_component_ids, total_clustering_result, labels_dictionary, _cluster_groups_with_id, cluster_algos, connected_col, groundtruth_name):
    my_gt = []
    sub = {}
    sub['mixed_best'] = []
    sub['predicted_clustering'] = []
    sub['ensemble_clustering'] = []
    for cluster_algo in cluster_algos:
        algorith_name = cluster_algo.__name__
        sub[algorith_name] = []
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
