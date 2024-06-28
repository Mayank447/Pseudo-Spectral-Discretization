#!/usr/bin/env python3

import numpy as np


class DiracOperator:

    def __init__(self, n_time, nu, n_landau):
        self.n_time = n_time
        self.nu = nu
        self.n_landau = n_landau

    def apply_to(self, spectral_coefficients, input_space, output_space):
        return self.eigenvalue(0) * spectral_coefficients

    def eigenvalue(self, index):
        return np.pi / self.n_time

    def eigenfunction(self, index, output_space):
        pass

    def lattice(self, output_space):
        pass

    def transform(self, values, input_space, output_space):
        pass
