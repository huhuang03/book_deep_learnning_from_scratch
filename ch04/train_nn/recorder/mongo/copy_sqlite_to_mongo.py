import json

import numpy as np
from ch04.other.mongo.get_db import collection
from ..sqlite3.train_recorder import get_cursor

c, sqlite3_db = get_cursor()
c.execute('''SELECT * from train_records''')
rows = c.fetchall()
v = lambda src: np.array(json.loads(src))
for row in rows:
    (index, w1_bytes, b1_bytes, w2_bytes, b2_bytes, loss, accuracy) = row[1:]
    data = {
        'w1': v(w1_bytes),
        'b1': v(b1_bytes),
        'w2': v(w2_bytes),
        'b2': v(b2_bytes),
        'index': index,
        'loss': loss,
        'accuracy': accuracy
    }
    collection.insert_one({
        'w1': data['w1'].tolist(),
        'b1': data['b1'].tolist(),
        'w2': data['w2'].tolist(),
        'b2': data['b2'].tolist(),
        'index': data['index'],
        'loss': data['loss'],
        'accuracy': data['accuracy']
    })
c.close()
sqlite3_db.close()