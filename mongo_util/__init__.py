import yaml
from pymongo import database, MongoClient


def get_collection(name: str) -> database.Collection:
    with open('local_properties.yml', 'r') as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
        connect_url = yaml_data['mongo']['connect_url']

    client = MongoClient(connect_url)
    db = client['book_deep_learning_from_scratch']
    collection = db[name]
    return collection