import timeit
import numpy as np
import matplotlib.pyplot as plt
from pseudospectral import FreeFermion2D


def create_spectrum(n):
    num_points = 2 * [n]
    # In order to ease the benchmarking of different version with different interfaces, we try our way through the different interfaces:
    try:
        # The current interface:
        return FreeFermion2D(num_points, L=num_points)
    except TypeError:
        # A previous interface:
        return FreeFermion2D(*num_points, *num_points)


def total_num_of_dof(spectrum):
    try:
        return spectrum.total_num_of_dof
    except AttributeError:
        return spectrum.total_num_lattice_points


num_input_vectors = 1000

timings = {}
for linear_num_lattice_points in [16, 32, 64, 128, 256]:
    spectrum = create_spectrum(linear_num_lattice_points)
    shape = (num_input_vectors, total_num_of_dof(spectrum))
    timings[linear_num_lattice_points] = np.mean(
        timeit.repeat(
            'spectrum.transform(input_vectors, "real","spectral")',
            setup="input_vectors=np.random.rand(*shape)",
            number=1,
            repeat=2,
            globals=globals(),
        )
    )

print("Timings\n---------")
for size, timing in timings.items():
    print(f"L=N={size}: {round(timing,3)}s")


plt.plot(timings.keys(), timings.values())
plt.show()
