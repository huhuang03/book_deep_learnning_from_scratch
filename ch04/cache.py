from typing import List, Optional
import os
import pickle

cache_path = 'runtime/cache.pickle'

if not os.path.exists(os.path.dirname(cache_path)):
    os.mkdir(os.path.dirname(cache_path))

class Cache:
    """
    how can I load and save?
    """
    def __init__(self):
        self.items: List['CacheItem'] = []

    def add(self, cache_item: 'CacheItem'):
        self.items.append(cache_item)

    def save(self):
        with open(cache_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load() -> Optional['Cache']:
        if not os.path.exists(cache_path):
            return None
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

class CacheItem:
    def __init__(self, no, params):
        self.no = no
        self.params = params
        self.gradient = None
        self.loss = None

    def set_gradient(self, gradient):
        self.gradient = gradient