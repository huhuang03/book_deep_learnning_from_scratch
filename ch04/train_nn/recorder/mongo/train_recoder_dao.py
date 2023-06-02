from typing import Optional
from ..recoder import TrainRecord

import numpy as np
from ch04.other.mongo.get_db import collection


def insert(item: TrainRecord):
    v = lambda src: src.tolist()
    collection.insert_one({
        'index': item['index'],
        'w1': v(item['w1']),
        'b1': v(item['b1']),
        'w2': v(item['w2']),
        'b2': v(item['b2']),
        'loss': v(item['loss']),
        'accuracy': v(item['accuracy']),
    })

def find_by_index(index: int) -> Optional[TrainRecord]:
    document = collection.find_one({"index": index})
    v = lambda src: np.array(src)
    if document:
        return {
            'w1': v(document['w1']),
            'b1': v(document['b1']),
            'w2': v(document['w2']),
            'b2': v(document['b2']),
            'index': index,
            'loss': document['loss'],
            'accuracy': document['accuracy']
        }