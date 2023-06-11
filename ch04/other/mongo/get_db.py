from mongo_util import get_collection

collection = get_collection('train_recoder')

collection.create_index('index', unique=True)