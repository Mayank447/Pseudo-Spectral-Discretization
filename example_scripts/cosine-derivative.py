import numpy as np
import matplotlib.pyplot as plt
from pseudospectral import DiracOperator, Derivative1D


if __name__ == "__main__":
    num_lattice_points = 101
    L = 2 * np.pi
    d = DiracOperator(Derivative1D(num_lattice_points, L))
    x = np.linspace(0, L, num_lattice_points, endpoint=False)
    y = np.cos(x)
    z = d.apply_to(y)
    
    plt.plot(x, y, label="cos(x)")
    plt.plot(x, -np.sin(x), label="-sin(x)")
    plt.plot(x, np.real(z), label="d/dx cos(x)", linestyle="--")

    # Any imaginary contribution is numerical noise:
    assert np.allclose(np.imag(z), 0)
    plt.legend()
    plt.show()
