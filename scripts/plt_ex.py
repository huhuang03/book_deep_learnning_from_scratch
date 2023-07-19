import matplotlib.pyplot as plt
import numpy as np


x = np.arange(-5, 5, 0.1)
y = np.exp(x)
plt.plot(x, y, label="sin", linestyle="--")
plt.legend()
plt.show()
