from matplotlib import pyplot as plt
from mongo_util import get_collection

conn = get_collection('manually_single_layer')
data = conn.aggregate([
    {'$sample': {'size': 1000}},
    {'$project': {'index': True, 'accuracy': True}},
    {'$sort': {'index': 1}}
])
items = list(data)

plt.plot([x['index'] for x in items], [x['accuracy'] for x in items])
plt.show()