#!/usr/bin/env python3
import numpy as np


def naive_implementation_of(Spectrum):
    class NaiveImplementationOf(Spectrum):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def scalar_product(self, lhs, rhs, input_basis="real"):
            final_shape = lhs.shape[:-1] + rhs.shape[:-1]
            lhs = lhs.reshape(-1, 1, self.total_num_of_dof)
            rhs = rhs.reshape(1, -1, self.total_num_of_dof)
            return np.inner(self.transform(lhs, input_basis=input_basis, output_basis="spectral"), self.transform(rhs, input_basis=input_basis, output_basis="spectral")).reshape(final_shape)

    return NaiveImplementationOf
