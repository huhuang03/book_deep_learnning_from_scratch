import numpy as np

from util.mnist import load_mnist
from .two_layer_net import TwoNetLayer


def start():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    # iters_num = 10000
    iters_num = 1643
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rage = 0.1

    network = TwoNetLayer(input_size=784, hidden_size=50, output_size=10)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = network.gradient(x_batch, t_batch)
        network.layers['Affine1'].W -= learning_rage * grad['W1']
        network.layers['Affine1'].b -= learning_rage * grad['b1']
        network.layers['Affine2'].W -= learning_rage * grad['W2']
        network.layers['Affine2'].W -= learning_rage * grad['b2']
        loss = network.lose(x_batch, t_batch)
        accuracy = network.accuracy(x_batch, t_batch)
        print(f'{i}/{iters_num}: loss: {loss}, accuracy: {accuracy}')

if __name__ == '__main__':
    start()