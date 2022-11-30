import random
from typing import List

import numpy
import pandas as pd
import numpy as np
import networkx as nx
from scipy.cluster import hierarchy
import markov_clustering as mc
import scipy.spatial.distance as ssd

from deduplipy.config import DEDUPLICATION_ID_NAME, ROW_ID
from deduplipy.clustering.fill_missing_edges import fill_missing_links


def basic_clustering_steps(scored_pairs_table: pd.DataFrame, col_names: List,
                           clustering_algorithm, use_cc=True, **args) -> pd.DataFrame:
    """
    Apply the basic steps to start clustering, then apply the given clusteringalgorithm to scored_pairs_table and
    perform the actual deduplication by adding a cluster id to each record

    Args:
        scored_pairs_table: Pandas dataframe containg all pairs and the similarity probability score
        col_names: name to use for deduplication
        clustering_algorithm: the clustering algorithm that has to be used
        **args: the named args for the clustering algorithm

    Returns:
        Pandas dataframe containing records with cluster id

    """

    graph = nx.Graph()
    for j, row in scored_pairs_table.iterrows():
        graph.add_node(row[f'{ROW_ID}_1'], **{col: row[f'{col}_1'] for col in col_names})
        graph.add_node(row[f'{ROW_ID}_2'], **{col: row[f'{col}_2'] for col in col_names})
        graph.add_edge(row[f'{ROW_ID}_1'], row[f'{ROW_ID}_2'], score=row['score'])

    components = nx.connected_components(graph)

    stats = {}
    clustering = {}
    cluster_counter = 0
    if use_cc or clustering_algorithm.__name__ == 'connected_components':
        for component in components:
            subgraph = graph.subgraph(component)
            if clustering_algorithm.__name__ in args['args']:
                clusters = clustering_algorithm(subgraph, **args['args'][clustering_algorithm.__name__])
            else:
                clusters = clustering_algorithm(subgraph)
            clustering.update(dict(zip(subgraph.nodes(), clusters + cluster_counter)))

            if clustering_algorithm.__name__ == 'connected_components':
                clustcoefficient = nx.average_clustering(subgraph)
                trans = nx.transitivity(subgraph)
                ecc = nx.eccentricity(subgraph)
                dia = nx.diameter(subgraph)
                radius = nx.radius(subgraph)
                nodecount = len(subgraph.nodes())
                edgecount = subgraph.number_of_edges()
                edgeweights = nx.get_edge_attributes(subgraph, 'score')
                maxedgeweight = max(edgeweights.values())
                minedgeweight = min(edgeweights.values())
                avgedgeweight = sum(edgeweights.values()) / len(edgeweights)
                stats[cluster_counter + 1] = {'clustcoefficient': clustcoefficient, 'transitivity': trans,
                                              'eccentricity': ecc, 'diameter': dia, 'radius': radius,
                                              'nodecount': nodecount,
                                              'edgecount': edgecount, 'maxedgeweight': maxedgeweight,
                                              'minedgeweight': minedgeweight, 'avgedgeweight': avgedgeweight}

            cluster_counter += len(component)
    else:
        subgraph = graph
        if clustering_algorithm.__name__ in args['args']:
            clusters = clustering_algorithm(subgraph, **args['args'][clustering_algorithm.__name__])
        else:
            clusters = clustering_algorithm(subgraph)
        clustering.update(dict(zip(subgraph.nodes(), clusters + cluster_counter)))


    df_clusters = pd.DataFrame.from_dict(clustering, orient='index',
                                         columns=[DEDUPLICATION_ID_NAME + "_" + clustering_algorithm.__name__])
    df_clusters.sort_values(DEDUPLICATION_ID_NAME + "_" + clustering_algorithm.__name__, inplace=True)
    df_clusters[ROW_ID] = df_clusters.index

    if clustering_algorithm.__name__ == 'connected_components':
        return df_clusters, stats


    return df_clusters


def hierarchical_clustering(subgraph, cluster_threshold: float = 0.5, fill_missing=True) -> np.ndarray:
    """
    Apply hierarchical clustering to scored_pairs_table and perform the actual deduplication by adding a cluster id to
    each record

    Args:
        subgraph: The subgraph given by the connected components algorithm
        cluster_threshold: threshold to apply in hierarchical clustering
        fill_missing: whether to impute missing values in the adjacency matrix using softimpute, otherwise missing
            values in the adjacency matrix are filled with zeros

    Returns:
        ndarray of clusters

    """
    if len(subgraph.nodes) > 1:
        adjacency = nx.to_numpy_array(subgraph, weight='score')
        # if len(subgraph.nodes) > 8:
        #    print("Interesting component here!")
        if fill_missing:
            adjacency = fill_missing_links(adjacency)
        distances = (np.ones_like(adjacency) - np.eye(len(adjacency))) - adjacency
        condensed_distance = ssd.squareform(distances)
        linkage = hierarchy.linkage(condensed_distance, method='centroid')
        clusters = hierarchy.fcluster(linkage, t=1 - cluster_threshold, criterion='distance')
    else:
        clusters = np.array([1])

    return clusters


def markov_clustering(subgraph, inflation: float = 2) -> np.ndarray:
    """
    Apply markov clustering to the given subraph. Markov clustering performs a random walk on the adjacency matrix
    and calculates the possibilities of reaching a certain node
    Args:
        subgraph: The subgraph given by the connected components algorithm
        inflation: The inflation value denotes how "fast" links should be made "stronger". A higher inflation value gives
        a coarser clustering. A lower value gives a more loose clustering

     Returns:
        ndarray of clusters

    """
    # if len(subgraph.nodes) < 2:
    #    print("En nu?")
    # if len(subgraph.nodes) > 8:
    #    print("Interesting component here!")
    matrix = nx.to_scipy_sparse_matrix(subgraph, weight="score")
    result = mc.run_mcl(matrix, inflation=inflation)
    mc_clusters = mc.get_clusters(result)
    clust_ind = 1
    mc_clusters_formatted = [0] * len(subgraph.nodes)
    for clust in mc_clusters:
        for val in clust:
            mc_clusters_formatted[val] = clust_ind
        clust_ind += 1
    clusters = numpy.array(mc_clusters_formatted)

    return clusters


def connected_components(subgraph) -> np.ndarray:
    """
    As the connected components algorithm is already in the base step of clustering, all it has to do is return
    an array of 1's, indicating that the entire component found is a cluster. Purely for baseline
    Returns: a ndarray of 1's

    """
    clusters = np.ones((len(subgraph.nodes),), dtype=int)
    return clusters
