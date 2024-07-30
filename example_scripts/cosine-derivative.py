import numpy as np
import matplotlib.pyplot as plt
from pseudospectral import DiracOperator, Derivative1D


def cosine(x):
    return np.cos(x)

if __name__ == "__main__":
    d = DiracOperator(Derivative1D(10, 10))
    x = np.arange(10)
    plt.plot(x, -np.sin(x))
    plt.plot(x, np.abs(d.apply_to(x)))
    plt.show()
    # print()