import math
import numpy as np

from common.gradient import numerical_gradient

def fx(x):
    return (math.pow(x[0], 2) + math.pow(x[1], 2)) / 2

def iter_gradient(fn, x, step, times):
    for _ in np.arange(times):
        g = numerical_gradient(fn, x)
        x -= g * step
    print(x)


iter_gradient(fx, np.array([5., 3.]), 0.02, 1000)