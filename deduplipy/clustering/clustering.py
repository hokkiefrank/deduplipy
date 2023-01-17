import random
from typing import List

import numpy
import pandas as pd
import numpy as np
import networkx as nx
import itertools
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
        use_cc: Boolean that indicates whether Connected Components is used before clustering.
        scored_pairs_table: Pandas dataframe containing all pairs and the similarity probability score
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
    if (use_cc and clustering_algorithm.__name__ != 'markov_clustering') or clustering_algorithm.__name__ == 'connected_components':
        print(f"There are {nx.number_connected_components(graph)} components")
        for component in components:
            subgraph = graph.subgraph(component)
            # print(f"This component has {len(subgraph.nodes())} nodes")
            if clustering_algorithm.__name__ in args['args']:
                clusters = clustering_algorithm(subgraph, **args['args'][clustering_algorithm.__name__])
            else:
                clusters = clustering_algorithm(subgraph)
            clustering.update(dict(zip(subgraph.nodes(), clusters + cluster_counter)))

            if clustering_algorithm.__name__ == 'connected_components':
                stats[cluster_counter + 1] = get_cluster_stats(subgraph)

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
        if fill_missing:
            adjacency = fill_missing_links(adjacency)
        distances = (np.ones_like(adjacency) - np.eye(len(adjacency))) - adjacency
        condensed_distance = ssd.squareform(distances)
        linkage = hierarchy.linkage(condensed_distance, method='centroid')
        clusters = hierarchy.fcluster(linkage, t=1 - cluster_threshold, criterion='distance')

        # get_consistency(subgraph, clusters, adjacency)
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

    Returns:
        a ndarray of 1's

    """
    clusters = np.ones((len(subgraph.nodes),), dtype=int)
    return clusters


def get_consistency(subgraph, clustering, adjac):
    """

    Args:
        subgraph: the subgraph (Component) from which the cluster is obtained
        cluster: (one of) the clusters obtained from the connected component

    Returns:

    """
    # this gets all the triangles present in the Component
    all_triangles = nx.enumerate_all_cliques(subgraph)
    existing_triads = [x for x in all_triangles if len(x) == 3]
    adjac = nx.to_numpy_array(subgraph, weight='score')
    pdf = pd.DataFrame(clustering)
    clusters = pdf.groupby([0]).groups.values()
    con_v_incon = []
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        tris_incon = []
        tris_con = []
        inconsistent = 0
        consistent = 0
        for comb in itertools.combinations(cluster, 3):
            # consistent if all are not zero or if 1 is not zero or if all zero
            # inconsistent if 2 are not zero
            non_zero = 0
            e1 = adjac[comb[0], comb[1]]
            if e1 > 0:
                non_zero += 1
            e2 = adjac[comb[0], comb[2]]
            if e2 > 0:
                non_zero += 1
            e3 = adjac[comb[1], comb[2]]
            if e3 > 0:
                non_zero += 1
            if non_zero != 3:
                inconsistent += 1
                tris_incon.append(comb)
            else:
                consistent += 1
                tris_con.append(comb)
        con_v_incon.append(consistent/(consistent+inconsistent))
    #print(con_v_incon)
    return


def center_clustering(subgraph):
    edgelist = subgraph.edges


def get_cluster_stats(subgraph) -> dict:
    """

    Args:
        subgraph: The subgraph for which the stats have to be collected

    Returns:
        A dictionary with the stats in it
    """
    # return {}
    clustcoefficient = nx.average_clustering(subgraph)
    trans = nx.transitivity(subgraph)
    # ecc = nx.eccentricity(subgraph)
    dia = nx.diameter(subgraph)
    radius = nx.radius(subgraph)
    nodecount = len(subgraph.nodes())
    edgecount = subgraph.number_of_edges()
    edgeweights = nx.get_edge_attributes(subgraph, 'score')
    maxedgeweight = max(edgeweights.values())
    minedgeweight = min(edgeweights.values())
    avgedgeweight = sum(edgeweights.values()) / len(edgeweights)
    density = nx.density(subgraph)
    all_degrees = [val for (node, val) in subgraph.degree()]
    maxdegree = max(all_degrees)
    mindegree = min(all_degrees)
    avgdegree = sum(all_degrees) / len(all_degrees)
    # above or below a threshold for a feature (like density or something)
    # Number of triangle
    ts = nx.triangles(subgraph)
    triangles = sum(ts.values()) / 3
    # Connectivity
    # connectivity = nx.average_node_connectivity(subgraph)
    # Centralitiy
    centrality = nx.harmonic_centrality(subgraph, distance='score')
    avgcentral = sum(centrality.values()) / len(centrality)
    stat = {'clustcoefficient': clustcoefficient, 'transitivity': trans,
            'diameter': dia, 'radius': radius,
            'nodecount': nodecount,
            'edgecount': edgecount, 'maxedgeweight': maxedgeweight,
            'minedgeweight': minedgeweight, 'avgedgeweight': avgedgeweight,
            'density': density, 'maxdegree': maxdegree, 'mindegree': mindegree,
            'avgdegree': avgdegree, 'triangles': triangles,  # 'connectivity': connectivity,
            'centrality': avgcentral}

    return stat
