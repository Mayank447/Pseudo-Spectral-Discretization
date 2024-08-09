import numpy as np
import matplotlib.pyplot as plt
from pseudospectral import DiracOperator, Derivative1D


if __name__ == "__main__":
    num_lattice_points = 100
    L = 2 * np.pi
    d = DiracOperator(Derivative1D(num_lattice_points, L))
    x = np.linspace(0, L, num_lattice_points, endpoint=False)
    y = np.cos(x)
    plt.plot(x, y, label="cos(x)")
    plt.plot(x, -np.sin(x), label="-sin(x)")
    plt.plot(x, np.real(d.apply_to(y)), label="d/dx cos(x)", linestyle="--")
    plt.legend()
    plt.show()
