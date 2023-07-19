import os.path
import pickle

from ..record import get_latest

latest = get_latest()

dist_folder = os.path.join(os.getcwd(), 'runtime')

with open(os.path.join(dist_folder, 'latest.pkl'), 'wb') as f:
    pickle.dump({
     'w': latest['w'],
     'b': latest['b']
    }, f)