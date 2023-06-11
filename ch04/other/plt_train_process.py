from .mongo.get_db import collection


accuracy_list = collection.find(projection={'accuracy': 1})

# how to plot?
for document in accuracy_list:
    pass