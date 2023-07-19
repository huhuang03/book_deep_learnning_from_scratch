import dataclasses

import numpy as np
from typing import TypedDict, Optional

import pymongo

from mongo_util import get_collection

# 不用框架的单层训练过程，一个最简单的，只有Linear - SoftMax的示例
collection_name = 'manually_single_layer_1'

collection = get_collection(collection_name)
collection.create_index('index', unique=True)


class TrainRecord(TypedDict):
    index: int
    w: np.ndarray
    b: np.ndarray
    loss: float
    accuracy: float


class InsertTrainRecord(TrainRecord):
    batch_size: Optional[int]


def insert(item: InsertTrainRecord):
    def v(src):
        return src.tolist()

    to_insert = {
        'index': item['index'],
        'w': v(item['w']),
        'b': v(item['b']),
        'loss': item['loss'],
        'accuracy': item['accuracy']
    }
    if 'batch_size' in item:
        to_insert['batch_size'] = item.get('batch_size')

    collection.insert_one(to_insert)


def get_latest() -> Optional[TrainRecord]:
    one = collection.find_one({}, sort=[('index', pymongo.DESCENDING)])

    def v(src):
        return np.array(src)

    if one:
        return {
            'w': v(one['w']),
            'b': v(one['b']),
            'index': one['index'],
            'loss': one['loss'],
            'accuracy': one['accuracy']
        }
    return None
