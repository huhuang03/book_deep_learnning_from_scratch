# import numpy as np
# import sdl_x
# from .constant import *
#
# # how about do common net?
# class TwoNetLayer:
#     def __init__(self, input_size, hidden_size, output_size):
#         """
#         :param input_size:
#         :param hidden_size:
#         :param output_size:
#         """
#         self.params = {
#             W1: np.random.randn(input_size, hidden_size),
#             B1: np.zeros(hidden_size),
#             W2: np.random.randn(hidden_size, output_size),
#             B2: np.random.randn(output_size)
#         }
#
#     def predict(self, x):
#         # a1 z1
#         a1 = np.dot(x, self.params[W1]) + self.params[B1]
#         # print('a1: ', a1)
#         z1 = sdl_x.sigmoid(a1)
#         # print('z1: ', z1)
#         a2 = np.dot(z1, self.params[W2]) + self.params[B2]
#         z2 = sdl_x.softmax(a2)
#         return z2
#
#     def lose(self, x, t):
#         # 是不是这个lose不对？
#         # print(f'W1: {self.params[W1]}')
#         y = self.predict(x)
#         # print(f"y: {y}")
#         return sdl_x.cross_entropy_error(y, t)
#
#     def accuracy(self, x, t):
#         """
#         why have and you?
#         :param x:
#         :param t:
#         :return:
#         """
#         y = self.predict(x)
#         y = np.argmax(y, axis=1)
#         t = np.argmax(t, axis=1)
#         accuracy = np.sum((y == t) / float(x.shape[0]))
#         return accuracy
#
#     def numerical_gradient(self, x, t):
#         loss_w = lambda w: self.lose(x, t)
#         grad = {}
#         for key in (W1, B1, W2, B2):
#             grad[key] = sdl_x.numerical_gradient(loss_w, self.params[key])
#         return grad
#
# # import sys
# #
# # import numpy as np
# # import psutil
# #
# # from util.mnist import load_mnist
# # # from . import train_recorder
# # from . import two_layer_net
# # from .constant import *
# #
# #
# # def try_set_low_priority():
# #     current_p = psutil.Process()
# #     if sys.platform == 'win32':
# #         current_p.nice(0)
# #     else:
# #         print('is linux')
# #         current_p.nice(-20)
# #
# #
# # def start():
# #     (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# #
# #     # what's this
# #     train_lose_list = []
# #
# #     iters_num = 10000
# #     train_size = x_train.shape[0]
# #     batch_size = 100
# #     learning_rage = 0.1
# #
# #     network = two_layer_net.TwoNetLayer(input_size=784, hidden_size=50, output_size=10)
# #
# #     for i in range(iters_num):
# #         record = train_recorder.find_by_index(i)
# #         if record is not None:
# #             network.params = {W1: record['w1'], B1: record['b1'], W2: record['w2'], B2: record['b2']}
# #             continue
# #
# #         batch_mask = np.random.choice(train_size, batch_size)
# #         x_batch = x_train[batch_mask]
# #         t_batch = t_train[batch_mask]
# #
# #         grad = network.numerical_gradient(x_batch, t_batch)
# #         for key in (W1, B1, W2, B2):
# #             network.params[key] -= learning_rage * grad[key]
# #
# #         record: train_recorder.TrainRecord = {
# #             'index': i,
# #             'w1': network.params[W1],
# #             'b1': network.params[B1],
# #             'w2': network.params[W2],
# #             'b2': network.params[B2],
# #             'loss': network.lose(x_batch, t_batch),
# #             'accuracy': network.accuracy(x_batch, t_batch)
# #         }
# #         train_recorder.insert(record)
# #         print(f'{i}/{iters_num}: loss: {record["loss"]}, accuracy: {record["accuracy"]}')
# #
# #
# # if __name__ == '__main__':
# #     start()
