import numpy as np

def numerical_gradient(fn, x):
    """
    calc gradient manually
    :param fn:
    :param x:
    :return:
    """
    h = 1e-4
    rst = np.zeros_like(x)
    for i in np.arange(x.size):
        stub = x[i]
        x[i] = stub + h
        y2 = fn(x)
        x[i] = stub - h
        y1 = fn(x)
        rst[i] = (y2 - y1) / (2*h)
        x[i] = stub
    return np.array(rst)
