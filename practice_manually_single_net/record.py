import numpy as np
from typing import TypedDict, Optional

import pymongo

from mongo_util import get_collection

collection_name = 'manually_single_layer'

collection = get_collection(collection_name)
collection.create_index('index', unique=True)


class TrainRecord(TypedDict):
    index: int
    w: np.ndarray
    b: np.ndarray
    loss: float
    accuracy: float


def insert(item: TrainRecord):
    def v(src):
        return src.tolist()
    collection.insert_one({
        'index': item['index'],
        'w': v(item['w']),
        'b': v(item['b']),
        'loss': item['loss'],
        'accuracy': item['accuracy'],
    })


def get_latest() -> Optional[TrainRecord]:
    one = collection.find_one({}, sort=[('index', pymongo.DESCENDING)])
    v = lambda src: np.array(src)
    if one:
        return {
            'w': v(one['w']),
            'b': v(one['b']),
            'index': one['index'],
            'loss': one['loss'],
            'accuracy': one['accuracy']
        }
    return None
