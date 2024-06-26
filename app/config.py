import pymongo

from os import getenv
from elasticsearch import Elasticsearch
from dotenv import load_dotenv


load_dotenv()


def elastic_search():
    """Elastic search connection"""
    return Elasticsearch([getenv('ELASTICSEARCH')])


def price_collection():
    """NonFungible mongodb price collection"""
    client = pymongo.MongoClient(getenv('MONGO'), readPreference='secondary')
    db = client.nonfungible 
    return db.historicalusds




