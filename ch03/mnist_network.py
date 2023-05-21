import os.path
import pickle

import numpy as np

from . import sigmoid, softmax
from util import mnist


def get_test_data():
    (x_train, t_train), (x_test, t_test) = mnist.load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open(os.path.join(os.path.dirname(__file__), 'sample_weight.pkl'), 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    B1, B2, B3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + B1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + B2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + B3
    z3 = softmax(a3)
    return z3


if __name__ == '__main__':
    x, t = get_test_data()
    accuracy_cnt = 0
    network = init_network()

    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)
        if p == t[i]:
            accuracy_cnt += 1

    print(f"Accuracy: {float(accuracy_cnt) / len(x)}")