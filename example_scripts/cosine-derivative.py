import numpy as np
import matplotlib.pyplot as plt
from pseudospectral import DiracOperator, Derivative1D


def cosine(x):
    return np.cos(x)


if __name__ == "__main__":
    d = DiracOperator(Derivative1D(10, 10))
    x = np.arange(10)
    plt.plot(x, np.cos(2 * np.pi * x / 10), label="cos(x)")
    plt.plot(x, 2* np.pi/10 * -np.sin(2 * np.pi * x / 10), label="-sin(x)")
    plt.plot(x, d.apply_to(np.cos(2 * np.pi * x / 10)), label="d/dx sin(x)")
    plt.legend()
    plt.show()
