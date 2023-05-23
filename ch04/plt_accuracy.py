from .cache import Cache
from .two_layer_net import TwoNetLayer
from util.mnist import load_mnist

network = TwoNetLayer(0, 0, 0)

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_test.shape)
print(t_test.shape)

# for i in range(0, len(cache.items)):
#     cache_item = cache.items[i]
#     network.params = cache_item.params
#     network.accuracy()
