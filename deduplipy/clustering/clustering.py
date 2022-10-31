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


def hierarchical_clustering(scored_pairs_table: pd.DataFrame, col_names: List,
                            cluster_threshold: float = 0.5, fill_missing=True) -> pd.DataFrame:
    """
    Apply hierarchical clustering to scored_pairs_table and perform the actual deduplication by adding a cluster id to
    each record

    Args:
        scored_pairs_table: Pandas dataframe containg all pairs and the similarity probability score
        col_names: name to use for deduplication
        cluster_threshold: threshold to apply in hierarchical clustering
        fill_missing: whether to impute missing values in the adjacency matrix using softimpute, otherwise missing
            values in the adjacency matrix are filled with zeros

    Returns:
        Pandas dataframe containing records with cluster id

    """
    graph = nx.Graph()
    for j, row in scored_pairs_table.iterrows():
        graph.add_node(row[f'{ROW_ID}_1'], **{col: row[f'{col}_1'] for col in col_names})
        graph.add_node(row[f'{ROW_ID}_2'], **{col: row[f'{col}_2'] for col in col_names})
        graph.add_edge(row[f'{ROW_ID}_1'], row[f'{ROW_ID}_2'], score=row['score'])

    components = nx.connected_components(graph)

    clustering = {}
    cluster_counter = 0
    mc_clustering = {}
    mc_cluster_counter = 0
    for component in components:
        subgraph = graph.subgraph(component)
        if len(subgraph.nodes) > 1:
            #perform inline markov clustering for comparison to hierarchical
            matrix = nx.to_scipy_sparse_matrix(subgraph, weight="score")
            result = mc.run_mcl(matrix)
            mc_clusters = mc.get_clusters(result)
            clust_ind = 1
            mc_clusters_formatted = [0] * len(subgraph.nodes)
            for clust in mc_clusters:
                for val in clust:
                    mc_clusters_formatted[val] = clust_ind
                clust_ind += 1
            mc_clusters_formatted = numpy.array(mc_clusters_formatted)
            # end markov clustering

            adjacency = nx.to_numpy_array(subgraph, weight='score')
            if fill_missing:
                adjacency = fill_missing_links(adjacency)
            distances = (np.ones_like(adjacency) - np.eye(len(adjacency))) - adjacency
            condensed_distance = ssd.squareform(distances)
            linkage = hierarchy.linkage(condensed_distance, method='centroid')
            clusters = hierarchy.fcluster(linkage, t=1 - cluster_threshold, criterion='distance')
        else:
            clusters = np.array([1])
            mc_clusters_formatted = np.array([1])
        clustering.update(dict(zip(subgraph.nodes(), clusters + cluster_counter)))
        cluster_counter += len(component)

        # markov clustering update for results
        mc_clustering.update(dict(zip(subgraph.nodes(), mc_clusters_formatted + mc_cluster_counter)))
        mc_cluster_counter += len(component)

    df_clusters = pd.DataFrame.from_dict(clustering, orient='index', columns=[DEDUPLICATION_ID_NAME])
    df_clusters.sort_values(DEDUPLICATION_ID_NAME, inplace=True)
    df_clusters[ROW_ID] = df_clusters.index

    df_clusters_mc = pd.DataFrame.from_dict(mc_clustering, orient='index', columns=[DEDUPLICATION_ID_NAME])
    df_clusters_mc.sort_values(DEDUPLICATION_ID_NAME, inplace=True)
    df_clusters_mc[ROW_ID] = df_clusters_mc.index

    return df_clusters, df_clusters_mc


def markov_clustering(scored_pairs_table: pd.DataFrame, col_names: List):


    print("markov_clustering")
    graph = nx.Graph()
    for j, row in scored_pairs_table.iterrows():
        graph.add_node(row[f'{ROW_ID}_1'], **{col: row[f'{col}_1'] for col in col_names})
        graph.add_node(row[f'{ROW_ID}_2'], **{col: row[f'{col}_2'] for col in col_names})
        graph.add_edge(row[f'{ROW_ID}_1'], row[f'{ROW_ID}_2'], score=row['score'])
#
#
    # then get the adjacency matrix (in sparse form)
    matrix = nx.to_scipy_sparse_matrix(graph, weight="score")
#
    result = mc.run_mcl(matrix, inflation=3.5, pruning_threshold=0.05)
    clusters = mc.get_clusters(result)

    #mc.draw_graph(matrix, clusters, node_size=50, with_labels=True, edge_color="silver")

    return clusters
