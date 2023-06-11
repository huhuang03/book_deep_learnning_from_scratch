from mongo_util import get_collection as _get_collection

_collection_name = 'train_recoder'
collection = _get_collection(_collection_name)

collection.create_index('index', unique=True)