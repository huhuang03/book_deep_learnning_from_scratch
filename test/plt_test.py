import matplotlib.pyplot as plt
import time


def plt_lost(lost):
    x = range(0, len(lost))
    y = lost
    plt.plot(x, y)
    plt.show()


lost = []
# set timeout?
for i in range(0, 3):
    lost.append(i)
    plt_lost(lost)
    time.sleep(1)