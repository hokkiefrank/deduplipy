import os
from pkg_resources import resource_filename
import numpy as np
import pandas as pd
import networkx as nx

def load_stoxx50() -> pd.DataFrame:
    file_path = resource_filename('deduplipy', os.path.join('data', 'stoxx50_extended_with_id.xlsx'))
    df = pd.read_excel(file_path, engine='openpyxl')
    print("Column names: 'name'")
    return df  # [['name']]


def load_voters() -> pd.DataFrame:
    file_path = resource_filename('deduplipy', os.path.join('data', 'voter_names.csv'))
    df = (pd.read_csv(file_path))
    print("Column names: 'name', 'suburb', 'postcode'")
    return df


def load_musick(amount) -> pd.DataFrame:
    filename = 'musicbrainz_'+(str(amount))+'k.csv'
    filepath = resource_filename('deduplipy', os.path.join('data', filename))
    df = pd.read_csv(filepath)
    df = df[['CID', 'title', 'artist', 'album']]
    #lengte1 = len(df)
    #df.dropna(subset=['title', 'artist'], inplace=True)
    #lengte2 = len(df)
    df['title'] = df['title'].astype('U').values
    df['artist'] = df['artist'].astype('U').values
    df['album'] = df['album'].astype('U').values
    print(
        "Column names: 'TID', 'CID', 'CTID', 'SourceID', 'id', 'number', 'title', 'length', 'artist', 'album', 'year', 'language'")
    return df


def load_voters_5m() -> pd.DataFrame:
    df = None
    for i in range(5):
        filepath = resource_filename('deduplipy', os.path.join('data', f'ncvr_numrec_1000000_modrec_2_ocp_20_myp_{i}_nump_5.csv'))
        df_temp = pd.read_csv(filepath)
        df_temp['givenname'] = df_temp['givenname'].astype('U').values
        df_temp['surname'] = df_temp['surname'].astype('U').values
        df_temp['suburb'] = df_temp['suburb'].astype('U').values
        df_temp['postcode'] = df_temp['postcode'].astype('U').values
        df_temp['recid'] = df_temp['recid'].astype('U').values

        if df is None:
            df = df_temp
        else:
            df = pd.concat([df, df_temp])

    return df


def load_affiliations():
    file_path = resource_filename('deduplipy', os.path.join('data', 'affiliationstrings_ids.csv'))
    df = (pd.read_csv(file_path))


def load_music_single(count):
    df = load_musick(count)
    df['value'] = df[['title', 'artist', 'album']].agg(" ".join, axis=1)
    df = df[['CID', 'value']]
    return df


def load_cora():
    file_path = resource_filename('deduplipy', os.path.join('data', 'cora.tsv'))
    df = pd.read_csv(file_path, sep='\t')
    gt_file = resource_filename('deduplipy', os.path.join('data', 'cora_DPL.tsv'))
    gt_df = pd.read_csv(gt_file, sep='\t')

    # Create an undirected graph from the DataFrame
    G = nx.Graph()
    G.add_edges_from(gt_df.values)

    # Find the connected components and assign a unique ID to each
    clusters = list(nx.connected_components(G))
    singleton_nodes = list(nx.isolates(G))
    singleton_ids = range(len(clusters), len(clusters) + len(singleton_nodes))
    singleton_clusters = [{node} for node in singleton_nodes]
    clusters += singleton_clusters
    cluster_ids = {node: i for i, cluster in enumerate(clusters) for node in cluster}

    # Add the cluster IDs to the DataFrame
    gt_df['cluster_id'] = gt_df['id1'].map(cluster_ids)
    df['gt_id'] = df['id'].map(cluster_ids)
    n_missing = len(df[df['gt_id'].isnull()])
    max_cluster_id = df[df['gt_id'].notnull()]['gt_id'].max()
    df.loc[df['gt_id'].isnull(), 'gt_id'] = np.arange(max_cluster_id + 1, max_cluster_id + 1 + n_missing)

    df = df[['gt_id', 'authors', 'title']]
    df['authors'] = df['authors'].astype('U').values
    df['title'] = df['title'].astype('U').values
    return df


def load_data(kind: str = 'voters', count:int =20) -> pd.DataFrame:
    """
    Load data for experimentation. `kind` can be 'stoxx50', 'voters', 'musicbrainz20k', 'musicbrainz200k', 'affiliations' or 'voters5m'.

    Stoxx 50 data are created by the developer of DedupliPy. Voters data is based on the North Carolina voter registry
    and this dataset is provided by Prof. Erhard Rahm ('Comparative Evaluation of Distributed Clustering Schemes for
    Multi-source Entity Resolution').

    Args:
        kind: 'stoxx50' or 'voters'

    Returns:
        Pandas dataframe containing experimentation dataset
    """
    if kind == 'stoxx50':
        return load_stoxx50()
    elif kind == 'voters':
        return load_voters()
    elif kind == 'musicbrainz20k':
        return load_musick(20)
    elif kind == 'musicbrainz200k':
        return load_musick(200)
    elif kind == 'affiliations':
        return load_affiliations()
    elif kind == 'musicbrainz20k_single':
        return load_music_single(count)
    elif kind == 'voters5m':
        return load_voters_5m()
    elif kind == 'cora':
        return load_cora()
