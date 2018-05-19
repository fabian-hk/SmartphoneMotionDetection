import matplotlib.pyplot as plt
import math


def learning_rate(x):
    return math.exp(-(x * 0.3)) * 0.001


if __name__ == "__main__":
    i = 0.0
    a = []
    b = []
    while i < 21:
        a.append(i)
        b.append(learning_rate(i))
        i += 0.01

    plt.plot(a, b)
    plt.show()
