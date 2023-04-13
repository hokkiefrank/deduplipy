import random
from typing import List

import numpy
import pandas as pd
import numpy as np
import networkx as nx
import itertools
from scipy.cluster import hierarchy
from scipy.optimize import curve_fit
from sklearn.cluster import AffinityPropagation, OPTICS, MiniBatchKMeans, SpectralClustering
from cdlib import algorithms
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import markov_clustering as mc
import scipy.spatial.distance as ssd
from scipy.sparse.csgraph import connected_components as actual_connected_components
from deduplipy.clustering.ensemble_clustering import ClusterSimilarityMatrix
from deduplipy.analyzing.metrics_collection import perform_scoring
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
    smallest_cc = min(nx.connected_components(graph), key=len)
    stats = {}
    clustering = {}
    cluster_counter = 0
    if use_cc or clustering_algorithm.__name__ == 'connected_components':
        print(f"There are {nx.number_connected_components(graph)} components")
        for component in components:
            subgraph = graph.subgraph(component)
            # print(f"This component has {len(subgraph.nodes())} nodes")
            if clustering_algorithm.__name__ in args['args']:
                clusters = clustering_algorithm(subgraph, **args['args'][clustering_algorithm.__name__])
            else:
                clusters = clustering_algorithm(subgraph)
            #if len(np.unique(clusters)) > 1:
            #    adj = nx.to_numpy_array(subgraph, weight='score')
            #    silhouettescore = perform_scoring('silhouette_score', adj, clusters)
            #else:
            #    silhouettescore = 0
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


def markov_clustering_on_adjacency(adj, pruning_value=0.001):
    result = mc.run_mcl(adj, pruning_threshold=pruning_value)
    mc_clusters = mc.get_clusters(result)
    clust_ind = 1
    mc_clusters_formatted = [0] * adj.shape[0]
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


def center_clustering(subgraph):
    edgelist = subgraph.edges


def affinity_propagation(subgraph, random_state: int = 10):
    adjacency = nx.to_numpy_array(subgraph, weight='score')
    ap = AffinityPropagation(random_state=random_state).fit(adjacency)
    clusters = ap.labels_
    return clusters


def optics(subgraph, min_samples: int = 5):
    if len(subgraph.nodes) > 1:
        adjacency = nx.to_numpy_array(subgraph, weight='score')
        opt = OPTICS(min_samples=min_samples).fit(adjacency)
        clusters = opt.labels_
        clustindex = max(clusters) + 1
        for clustid in range(len(clusters)):
            clust = clusters[clustid]
            if clust == -1:
                clusters[clustid] = clustindex
                clustindex += 1
    else:
        clusters = np.array([1])

    return clusters


def cdl_communities_to_clusters(communities, subgraph):
    clust_ind = 1
    lv_clusters_formatted = [0] * len(subgraph.nodes)
    nodes = list(subgraph.nodes)
    for clust in communities:
        for val in clust:
            lv_clusters_formatted[nodes.index(val)] = clust_ind
        clust_ind += 1
    clusters = numpy.array(lv_clusters_formatted)
    return clusters


def louvain(subgraph, weight='score', resolution: int = 1., randomize=False):
    if len(subgraph.nodes) > 1:
        communities = algorithms.louvain(subgraph, weight=weight, resolution=resolution,
                                         randomize=randomize).communities
        clusters = cdl_communities_to_clusters(communities, subgraph)
    else:
        clusters = np.array([1])

    return clusters


def leiden(subgraph):
    if len(subgraph.nodes) > 1:
        communities = algorithms.leiden(subgraph).communities
        clusters = cdl_communities_to_clusters(communities, subgraph)
    else:
        clusters = np.array([1])

    return clusters


def walktrap(subgraph):
    if len(subgraph.nodes) > 1:
        communities = algorithms.walktrap(subgraph).communities
        clusters = cdl_communities_to_clusters(communities, subgraph)
    else:
        clusters = np.array([1])

    return clusters


def girvan_newman(subgraph):
    if len(subgraph.nodes) > 1:
        communities = list(nx.algorithms.community.girvan_newman(subgraph))
        #communities = algorithms.girvan_newman(subgraph, level=level).communities
        clusters = cdl_communities_to_clusters(communities, subgraph)
    else:
        clusters = np.array([1])

    return clusters


def leading_eigenvector(subgraph):
    if len(subgraph.nodes) > 1:
        communities = algorithms.eigenvector(subgraph).communities
        clusters = cdl_communities_to_clusters(communities, subgraph)
    else:
        clusters = np.array([1])

    return clusters


def label_propagation(subgraph):
    if len(subgraph.nodes) > 1:
        communities = algorithms.label_propagation(subgraph).communities
        clusters = cdl_communities_to_clusters(communities, subgraph)
    else:
        clusters = np.array([1])

    return clusters


def greedy_modularity(subgraph):
    if len(subgraph.nodes) > 1:
        communities = algorithms.greedy_modularity(subgraph, weight="score").communities
        clusters = cdl_communities_to_clusters(communities, subgraph)
    else:
        clusters = np.array([1])

    return clusters


def paris(subgraph):
    if len(subgraph.nodes) > 1:
        communities = algorithms.paris(subgraph).communities
        #clusters = cdl_communities_to_clusters(communities, subgraph)
        clust_ind = 1
        clusters_formatted = [0] * len(subgraph.nodes)
        for clust in communities:
            for val in clust:
                clusters_formatted[val] = clust_ind
            clust_ind += 1
        clusters = numpy.array(clusters_formatted)
    else:
        clusters = np.array([1])

    return clusters


def cdlib_ipca(subgraph):
    if len(subgraph.nodes) > 1:
        communities = algorithms.ipca(subgraph, weights="score").communities
        clusters = cdl_communities_to_clusters(communities, subgraph)
    else:
        clusters = np.array([1])

    return clusters


def cdlib_dcs(subgraph):
    if len(subgraph.nodes) > 1:
        communities = algorithms.dcs(subgraph).communities
        clusters = cdl_communities_to_clusters(communities, subgraph)
    else:
        clusters = np.array([1])

    return clusters


def cdlib_cpm(subgraph):
    if len(subgraph.nodes) > 1:
        communities = algorithms.cpm(subgraph).communities
        clusters = cdl_communities_to_clusters(communities, subgraph)
    else:
        clusters = np.array([1])

    return clusters


def cdlib_belief(subgraph):
    if len(subgraph.nodes) > 1:
        communities = algorithms.belief(subgraph).communities
        clusters = cdl_communities_to_clusters(communities, subgraph)
    else:
        clusters = np.array([1])

    return clusters


def cdlib_infomap(subgraph):
    if len(subgraph.nodes) > 1:
        communities = algorithms.infomap(subgraph).communities
        clusters = cdl_communities_to_clusters(communities, subgraph)
    else:
        clusters = np.array([1])

    return clusters


def cdlib_kcut(subgraph):
    if len(subgraph.nodes) > 1:
        communities = algorithms.kcut(subgraph, kmax=4).communities
        clusters = cdl_communities_to_clusters(communities, subgraph)
    else:
        clusters = np.array([1])

    return clusters


def cdlib_der(subgraph):
    if len(subgraph.nodes) > 1:
        communities = algorithms.der(subgraph).communities
        clusters = cdl_communities_to_clusters(communities, subgraph)
    else:
        clusters = np.array([1])

    return clusters


def cdlib_pycombo(subgraph):
    if len(subgraph.nodes) > 1:
        communities = algorithms.pycombo(subgraph, weight='score').communities
        clusters = cdl_communities_to_clusters(communities, subgraph)
    else:
        clusters = np.array([1])

    return clusters


def cdlib_scan(subgraph):
    if len(subgraph.nodes) > 1:
        communities = algorithms.scan(subgraph, epsilon=0.7, mu=3).communities
        clusters = cdl_communities_to_clusters(communities, subgraph)
    else:
        clusters = np.array([1])

    return clusters


def cdlib_spinglass(subgraph):
    if len(subgraph.nodes) > 1:
        communities = algorithms.spinglass(subgraph).communities
        clusters = cdl_communities_to_clusters(communities, subgraph)
    else:
        clusters = np.array([1])

    return clusters


def cdlib_ga(subgraph):
    if len(subgraph.nodes) > 1:
        communities = algorithms.ga(subgraph).communities
        clusters = cdl_communities_to_clusters(communities, subgraph)
    else:
        clusters = np.array([1])

    return clusters


def kmeans_with_elbow(subgraph):
    # Generate some sample data

    X = nx.to_numpy_array(subgraph, weight='score')
    max_k = len(subgraph.nodes)
    # Try clustering for different values of k
    wcss = []
    all_labels = []
    for k in range(1, max_k):
        kmeans = SpectralClustering(n_clusters=k, n_init=10, random_state=0)
        kmeans.fit(X)
        all_labels.append(kmeans.labels_)
        wcss.append(kmeans.inertia_)

    # Calculate the first derivative of the WCSS curve
    dx = np.diff(wcss)
    dy = np.diff(range(len(wcss)))
    grad = np.array([dy[i] / dx[i] for i in range(len(dx))])

    # Find the knee point using the find_peaks function from scipy.signal
    peaks, _ = find_peaks(grad)
    knee_point = None
    if len(peaks) > 0:
        knee_point = peaks[0] + 1

    # Plot the WCSS as a function of k and highlight the knee point
    plt.plot(range(1, max_k), wcss)
    if knee_point:
        plt.plot(knee_point, wcss[knee_point - 1], marker='o', color='red')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    if knee_point:
        print(f'The knee point is at k={knee_point}')
    else:
        knee_point = 1
    return all_labels[knee_point - 1]


def kmeans_with_ensemble(subgraph):
    NUM_KMEANS = 32
    MIN_PROBABILITY = 0.6
    X = nx.to_numpy_array(subgraph, weight='score')
    # Generating a "Cluster Forest"
    clustering_models = NUM_KMEANS * [
        # Note: Do not set a random_state, as the variability is crucial
        # This is an extreme simple K-Means
        MiniBatchKMeans(n_clusters=int((len(subgraph.nodes))/2), batch_size=64, n_init=1, max_iter=20)
    ]

    clt_sim_matrix = ClusterSimilarityMatrix()
    for model in clustering_models:
        clt_sim_matrix.fit(model.fit_predict(X=X))

    sim_matrix = clt_sim_matrix.similarity
    norm_sim_matrix = sim_matrix / sim_matrix.diagonal()

    # Transforming the probabilities into graph edges
    # This is very similar to DBSCAN
    graph = (norm_sim_matrix > MIN_PROBABILITY).astype(int)
    n_clusters, y_ensemble = actual_connected_components(graph, directed=False, return_labels=True)
    return y_ensemble

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
        con_v_incon.append(consistent / (consistent + inconsistent))
    # print(con_v_incon)
    return


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
            'avgdegree': avgdegree, 'triangles': triangles,
            'centrality': avgcentral}# 'connectivity': connectivity,

    return stat
