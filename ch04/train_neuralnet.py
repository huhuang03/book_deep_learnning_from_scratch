import numpy as np
from .constant import *

from util.mnist import load_mnist
from . import two_layer_net
from .cache import Cache, CacheItem

cache = Cache()


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# what's this
train_lose_list = []

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rage = 0.1

cache = Cache.load() or cache
print(f'items: {cache.items}')
cached_items = cache.items

network = two_layer_net.TwoNetLayer(input_size=784, hidden_size=50, output_size=10)
begin_i = -1
for i in range(len(cached_items)):
    if cached_items[i].loss is not None:
        begin_i = i

if begin_i >= 0:
    network.params = cached_items[begin_i].params

print(f"cached len: {begin_i + 1}")

for i in range(iters_num):
    if i < begin_i:
        train_lose_list.append(cached_items[i].loss)
        print(f'cached train_lose_list: {train_lose_list}')
        continue
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.numerical_gradient(x_batch, t_batch)
    cache_item = CacheItem(i, network.params)
    cache_item.grad = grad
    cache.add(cache_item)
    cache.save()

    for key in (W1, B1, W2, B2):
        network.params[key] -= learning_rage * grad[key]

    loss = network.lose(x_batch, t_batch)
    cache_item.loss = loss
    train_lose_list.append(loss)
    print(f"train_lose_list: {train_lose_list}")
    cache.save()