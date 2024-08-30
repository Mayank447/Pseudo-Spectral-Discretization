#!/usr/bin/env python3
import numpy as np


def naive_implementation_of(Spectrum):
    """
    Return a Spectrum class that implements naive transformation and scalar product.

    Given that the Spectrum class provides eigenfunctions, eigenvalues and a lattice,
    the naive implementations of transform and scalar_product are added. These are
    O(N^2) dense matrix operations that you don't want to use in production for
    reasons of performance and numerical stability. But they can be used to provide
    stubs for early development and for testing purposes.
    """

    class NaiveImplementationOf(Spectrum):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._compute_transforms()

        def _compute_transforms(self):
            self.inverse_transform = (
                self.eigenfunction(np.arange(self.total_num_of_dof))(*self.lattice())
                .reshape(self.total_num_of_dof, self.total_num_of_dof)
                .T
            )
            self.forward_transform = np.linalg.inv(self.inverse_transform)

        def transform(self, input_vector, input_basis, output_basis):
            if input_basis == output_basis in ["real", "spectral"]:
                return input_vector
            elif input_basis == "spectral" and output_basis == "real":
                return np.einsum("ij,...j->...i", self.inverse_transform, input_vector)
            elif input_basis == "real" and output_basis == "spectral":
                return np.einsum("ij,...j->...i", self.forward_transform, input_vector)
            else:
                message = (
                    "Unsupported space transformation from "
                    f"{input_basis} to {output_basis}."
                )
                raise ValueError(message)

        def scalar_product(self, lhs, rhs, input_basis="real"):
            final_shape = lhs.shape[:-1] + rhs.shape[:-1]
            lhs = lhs.reshape(-1, 1, self.total_num_of_dof)
            rhs = rhs.reshape(1, -1, self.total_num_of_dof)
            return np.inner(
                self.transform(lhs, input_basis=input_basis, output_basis="spectral"),
                self.transform(rhs, input_basis=input_basis, output_basis="spectral"),
            ).reshape(final_shape)

    return NaiveImplementationOf
