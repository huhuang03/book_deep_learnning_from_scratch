import numpy as np

from util.mnist import load_mnist
from .. import train_recorder
from ... import two_layer_net
from ...constant import *

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print(t_test.shape)
network = two_layer_net.TwoNetLayer(input_size=784, hidden_size=50, output_size=10)

train_size = x_train.shape[0]
batch_size = 100

for i in range(10000):
    record = train_recorder.find_by_index(i)
    if not record:
        continue

    network.params = {W1: record['w1'], B1: record['b1'], W2: record['w2'], B2: record['b2']}
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    print(f'no.{i}: loss: {record["loss"]}, accuracy: {record["accuracy"]}, calc loss: {network.lose(x_batch, t_batch)}, calc accuracy: {network.accuracy(x_test, t_test)}')
