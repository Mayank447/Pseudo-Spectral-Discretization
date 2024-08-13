import numpy as np
import matplotlib.pyplot as plt
from pseudospectral import DiracOperator, Derivative1D


def function(fn, x):
    return fn(x)


def derivative(der, x):
    return der(x)


if __name__ == "__main__":
    num_lattice_points = 100
    theta = 0.5
    L = 2 * np.pi * theta
    d = DiracOperator(Derivative1D(num_lattice_points, L, theta))

    x = np.linspace(0, L, num_lattice_points, endpoint=False)
    y = function(np.sin, x)
    y_ = derivative(np.cos, x)
    z = d.apply_to(y)

    plt.plot(x, y, label="sin(x)")
    plt.plot(x, y_, label="cos(x)")
    plt.plot(x, np.real(z), label="d/dx sin(x)", linestyle="--")
    
    assert np.allclose(np.imag(z), 0)
    plt.legend()
    plt.show()
