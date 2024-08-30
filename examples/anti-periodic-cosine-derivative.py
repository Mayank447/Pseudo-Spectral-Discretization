import numpy as np
import matplotlib.pyplot as plt
from pseudospectral import DiracOperator, Derivative1D


if __name__ == "__main__":
    total_num_lattice_points = 100
    theta = 0.5
    L = 2 * np.pi * theta
    spectrum = Derivative1D(total_num_lattice_points, L, theta=theta)
    d = DiracOperator(spectrum)
    (x,) = spectrum.lattice()
    y = np.cos(x)
    z = d.apply_to(y)

    plt.plot(x, y, label=r"$\cos(x)$")
    plt.plot(x, -np.sin(x), label=r"$-\sin(x)$")
    plt.plot(
        x, np.real(z), label=r"$\frac{\mathrm{d}}{\mathrm{d}x} \cos(x)$", linestyle="--"
    )

    # Any imaginary contribution is numerical noise:
    assert np.allclose(np.imag(z), 0)
    plt.legend()
    plt.show()
