import numpy as np

from util.mnist import load_mnist
from .. import two_layer_net
from ..constant import *

is_record = False
if is_record:
    from .recorder.mongo import train_recoder_dao
    from .recorder.recoder import TrainRecord


def start():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rage = 0.1

    network = two_layer_net.TwoNetLayer(input_size=784, hidden_size=50, output_size=10)

    find_start_index = False
    for i in range(iters_num):
        if is_record:
            record = train_recoder_dao.find_by_index(i)
            if record is not None:
                network.params = {W1: record['w1'], B1: record['b1'], W2: record['w2'], B2: record['b2']}
                continue
            if not find_start_index:
                print('start index: ', i)
            find_start_index = True

        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = network.numerical_gradient(x_batch, t_batch)
        for key in (W1, B1, W2, B2):
            network.params[key] -= learning_rage * grad[key]

        if is_record:
            record: TrainRecord = {
                'index': i,
                'w1': network.params[W1],
                'b1': network.params[B1],
                'w2': network.params[W2],
                'b2': network.params[B2],
                'loss': network.lose(x_batch, t_batch),
                'accuracy': network.accuracy(x_batch, t_batch)
            }
            train_recoder_dao.insert(record)
        print(f'{i}/{iters_num}: loss: {network.lose(x_batch, t_batch)}, accuracy: {network.accuracy(x_batch, t_batch)}')


if __name__ == '__main__':
    start()
