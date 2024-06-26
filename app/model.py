import time
import json
import plotly
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


from datetime import date
from itertools import cycle
from elasticsearch_dsl import Search
from sklearn.neighbors import NearestNeighbors


from .config import elastic_search


def retrieve_data(table):
    """
    Retrieves data from Elasticsearch
    """
    s = Search(using=elastic_search(), index=table).sort('-blockTimestamp')
    df = pd.DataFrame([hit.to_dict() for hit in s.scan()])
    df['assetId'] = df['assetId'].astype(int)
    return df


def expand_metadata(df):
    """
    Expands the metadata column into multiple columns
    """
    if 'meta' in df:
        print('(3/6) Exploding meta data column into mutiple columns')
        df['meta'] = df['meta'].fillna({i: {} for i in df.index})
        df = df.join(pd.json_normalize(df['meta']).add_prefix('meta.'))
        df.drop(['meta'], axis=1, inplace=True)

    if 'meta.tags' in df:
        df1 = pd.get_dummies(df['meta.tags'].apply(pd.Series).stack(), prefix="tags").groupby(level=0).sum()
        df.drop(['meta.tags'], axis=1, inplace=True)
        df = df.join(df1)
        
    df.drop(list(df.filter(regex = 'meta')), axis = 1, inplace = True)
    
    return df


def input_punk(df, id):
    
    df = df.copy()
    today  = date.today()
    
    target = ['totalDecimalPrice', 'usdPrice', 'blockTimestamp']

    # droping target from data matrix
    df_data = df.drop(target, axis=1)

    # set target
    df_target = df[target]

    # fit on data, 12 neighbors
    nn = NearestNeighbors(algorithm='brute', metric='cosine', leaf_size =15, n_neighbors=50, n_jobs=-1)
    nn.fit(df_data)

    # query point 
    input_index = id

    # vectorize 
    data_vect = df_data[df_data.index == input_index].values
    neigh_dist, neigh_indices = nn.kneighbors(data_vect)
    indexs = neigh_indices.flat[0:50].tolist()

    #adding url to each track
    output = df_target.iloc[indexs].copy()
    
    output['image_url'] = 'https://www.larvalabs.com/cryptopunks/cryptopunk' + output.index.astype(str) + '.png'
    output['ranking'] = np.arange(len(output))
    output['blockTimestamp'] = pd.to_datetime(output['blockTimestamp']).dt.date
    output['days_old'] = (today - output['blockTimestamp']).dt.days

    
    output = output.sort_values(by=['blockTimestamp'], ascending=False)
    output = output.loc[~output.index.isin([input_index])]
    output = output[0:7]
    usd_mean = output['usdPrice'].mean()
    eth_mean = output['totalDecimalPrice'].mean()
    
    #output['usdPrice'] = output['usdPrice'].apply(lambda x: "${:.1f}k".format((x/1000)))
    output['usdPrice'] = round(output['usdPrice'], 2)
    final = output[['blockTimestamp', 'usdPrice', 'totalDecimalPrice', 'ranking', 'days_old']]
    final.reset_index(inplace=True)
    final.rename(columns={'assetId': 'ID', 'blockTimestamp': 'Date', 'days_old': 'Days ago', 'usdPrice': 'USD', 'totalDecimalPrice': 'ETH', 'ranking': '°Seperation'}, inplace=True)
    return output, final, round(usd_mean,2), round(eth_mean,2)


def update_punks():
    table = 'nft_sales_cryptopunks_ͼ'
    raw_data = retrieve_data(table)
    expanded = expand_metadata(raw_data)

    df = expanded.sort_values('blockTimestamp').drop_duplicates('assetId',keep='last').copy()
    df = df.dropna(axis=1, how='all')
    to_drop = ['transactionHash', 'nftAddress', 'marketAddress', 'tokenAddress',
            'feeUSDPrice', 'feeTotalPrice', 'feeTotalDecimalPrice', 'tokenTicker', 'blockNumber',
            'feeCollectors', 'buyer', 'seller', 'blockchain', 'project', 'nftTicker',
            'saleType', 'totalPrice', 'image']

    df = df.drop(to_drop, axis=1)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    df['blockTimestamp'] = pd.to_datetime(df['blockTimestamp'])
    df.set_index('assetId', inplace=True)
    return df.to_csv('punks.csv')


def prediction_labels(project, ticker):
    s = Search(using=elastic_search(), index='asset_definitions_nftradar').query('match', project=project).query('match', nftTicker=ticker.upper()).source(['assetId', 'blockNumber', 'nftTicker', 'project', 'meta.tags'])
    df = pd.DataFrame([hit.to_dict() for hit in s.scan()])
    df['blockNumber'].replace('', np.nan, inplace=True)
    df.dropna(subset=['blockNumber'], inplace=True)

    df = df.drop_duplicates(subset=['assetId'], keep='first')
    df['assetId'] = df['assetId'].astype(int)
    df.sort_values(by='assetId', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def cum_graph(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['USD']))
    fig.update_layout(template='seaborn', paper_bgcolor='rgba(0,0,0,0)', 
                    font=dict(family="Tahoma", size=10), hovermode="x unified", plot_bgcolor='rgba(0,0,0,0)')
    line_plot = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return line_plot