import operator
import random
from deduplipy.config import DEDUPLICATION_ID_NAME


def get_cluster_column_name(clusteringalgorithm) -> str:
    if callable(clusteringalgorithm):
        return '_'.join([DEDUPLICATION_ID_NAME, clusteringalgorithm.__name__])
    elif isinstance(clusteringalgorithm, str):
        return '_'.join([DEDUPLICATION_ID_NAME, clusteringalgorithm])
    else:
        pass


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


def get_mixed_best(connected_components_clusters, total_resulting_clusters, cluster_algorithms, label_diction, eval_priorities):
    labels = []
    mixed_best = []
    cluster_groups_with_id = {'mixed_best': {}}
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
        # if there is only one row for this connected component then we don't have to pick a clustering and just use the connected component.
        if len(rows) < 2:
            mixed_best.append(list(rows.groupby([connected_col]).groups.values())[0])
            continue
        # get the ground truth clusters for this connected component
        gt = list(rows.groupby([groupby_name]).groups.values())
        temp = {}
        cluster_groups = {}

        # loop over the cluster algorithms to get the resulting clusters per connected component.
        for cluster_algo in cluster_algorithms:
            algorith_name = cluster_algo.__name__
            cluster_groups[algorith_name] = list(rows.groupby([get_cluster_column_name(cluster_algo)]).groups.values())
            cluster_groups_with_id[algorith_name][connectid] = list(rows.groupby([get_cluster_column_name(cluster_algo)]).groups.values())
            for eva in evaluations:
                if eva not in temp:
                    temp[eva] = {}
                temp[eva][algorith_name] = evaluate(cluster_groups[algorith_name], gt, eva)
        # get the algo winner and add the features of this connected component to the name of the winner
        algo_winner = select_winner(temp, eval_priorities, cluster_algorithms)
        # add the label of the winner to the labels array
        labels.append(label_diction[algo_winner])

        # if it was a draw, there are no groups associated with that name, so pick one
        if algo_winner == 'draw':
            algo_winner = cluster_algo_names[0]
        for gs in cluster_groups[algo_winner]:
            mixed_best.append(gs)
        cluster_groups_with_id['mixed_best'][connectid] = cluster_groups[algo_winner]

    return cluster_groups_with_id, labels, mixed_best, component_ids
