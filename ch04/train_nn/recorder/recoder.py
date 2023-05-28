import numpy as np
from typing import TypedDict

class TrainRecord(TypedDict):
    index: int
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    loss: float
    accuracy: float
