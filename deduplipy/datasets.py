import os
from pkg_resources import resource_filename

import pandas as pd


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


def load_music_20k() -> pd.DataFrame:
    filepath = resource_filename('deduplipy', os.path.join('data', 'musicbrainz_20k.csv'))
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

        if df is None:
            df = df_temp
        else:
            df = pd.concat([df, df_temp])

    return df

def load_data(kind: str = 'voters') -> pd.DataFrame:
    """
    Load data for experimentation. `kind` can be 'stoxx50' or 'voters' or 'musicbrainz20k' or 'voters5m'.

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
        return load_music_20k()
    elif kind == 'voters5m':
        return load_voters_5m()
