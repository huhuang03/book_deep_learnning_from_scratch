import yaml
from pymongo import MongoClient

with open('local_properties.yml', 'r') as file:
    yaml_data = yaml.load(file, Loader=yaml.FullLoader)
    connect_url = yaml_data['mongo']['connect_url']

client = MongoClient(connect_url)
db = client['book_deep_learning_from_scratch']
collection = db['train_recoder']

collection.create_index('index', unique=True)