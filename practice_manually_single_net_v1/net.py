import numpy as np
import sdl_x


class ManuallySingleNet:
    def __init__(self):
        self.w: np.ndarray = np.random.randn(28 * 28, 10)
        self.b: np.ndarray = np.random.randn(1, 10)
        self.dw = None
        self.db = None

    def load(self, w, b):
        self.w = w
        self.b = b

    def forward(self, x: np.ndarray):
        a = x @ self.w + self.b
        y = sdl_x.softmax(a)
        return y

    def accuracy(self, x, t) -> float:
        y = self.forward(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum((y == t) / float(x.shape[0]))
        return accuracy

    def loss(self, x, t) -> float:
        y = self.forward(x)
        return sdl_x.cross_entropy_error(y, t)

    def gradient(self, x, t):
        def f(_):
            return self.loss(x, t)

        self.dw = sdl_x.numerical_gradient(f, self.w)
        self.db = sdl_x.numerical_gradient(f, self.b)
