#!/usr/bin/env python3
from copy import deepcopy
import numpy as np


class ElementwiseSpectralMultiplication:
    """
    This spectrum represents elementwise multiplication of the spectral coefficients.
    In a way, it generalises the specialised spectra. For example, using
    spectrum.eigenvalues as field_values reproduces the original spectrum.
    """

    def __new__(cls, spectrum, given_field_values=1):
        # Inheriting from ElementwiseSpectralMultiplication in order to make it
        # identifiable as a derived spectrum.
        # `isinstance(inst, ElementwiseSpectralMultiplicationt)` would return
        # `False` otherwise.`
        class TailoredElementwiseSpectralMultiplication(
            type(spectrum), ElementwiseSpectralMultiplication
        ):
            field_values = given_field_values

            # Revert the non-trivial effects of inheriting from
            # ElementwiseSpectralMultiplication.
            def __new__(cls, *args, **kwargs):
                return super().__new__(cls, *args, **kwargs)

            # We'll not initialise it this way and need to avoid wrong constructor
            # calls.
            def __init__(self, *args, **kwargs):
                pass

            @property
            def eigenvalues(self):
                return self.field_values * np.ones_like(super().eigenvalues)

        new_self = deepcopy(spectrum)
        new_self.__class__ = TailoredElementwiseSpectralMultiplication
        return new_self


class ElementwiseRealMultiplication:
    """
    This spectrum represents elementwise multiplication of the real-space coefficients.
    Choosing all field_values = 1 yields the identity and thereby the same operator
    as ElementwiseSpectralMultiplication. But for any deviation from that the two
    spectra are different.

    CAUTION: The current implementation is only tested for quadrature lattices. There
    might be subtle inconsistencies otherwise.
    """

    @staticmethod
    def _compute_decomposition(spectrum):
        spectral_coefficients = spectrum.transform(
            np.eye(spectrum.total_num_of_dof), "real", "spectral"
        )
        norms = np.sqrt(
            np.diag(
                spectrum.scalar_product(
                    spectral_coefficients,
                    spectral_coefficients,
                    "spectral",
                )
            )
        )
        return spectral_coefficients / norms, norms

    def __new__(cls, spectrum, given_field_values=1):
        # Inheriting from ElementwiseRealMultiplication in order to make it
        # identifiable as a derived spectrum.
        # `isinstance(inst, ElementwiseRealMultiplicationt)` would return
        # `False` otherwise.`
        class TailoredElementwiseRealMultiplication(
            type(spectrum), ElementwiseRealMultiplication
        ):
            field_values = given_field_values
            spectral_coefficients, norms = cls._compute_decomposition(spectrum)

            # Revert the non-trivial effects of inheriting from
            # ElementwiseRealMultiplication.
            def __new__(cls, *args, **kwargs):
                return super().__new__(cls, *args, **kwargs)

            # We'll not initialise it this way and need to avoid wrong constructor
            # calls.
            def __init__(self, *args, **kwargs):
                pass

            @property
            def eigenvalues(self):
                return self.field_values * np.ones_like(super().eigenvalues)

            def eigenfunction(self, index):
                # This is important: Our eigenfunctions are not at all trivial. They are
                # general elements from our vector space that just happen to vanish at
                # all lattice points. It follows that we need to consider an appropriate
                # linear combination of the original eigenfunctions.

                # super() needs to be called inside of the class. We can't shove it into
                # the lambda because it would refer to the wrong context there.
                eigenfunction = super().eigenfunction

                return lambda *x: np.einsum(
                    "ij,j...->i...",
                    self.spectral_coefficients[index, :],
                    eigenfunction(np.arange(self.total_num_of_dof))(*x),
                )

            def transform(self, input_vector, input_basis="real", output_basis="real"):
                if input_basis == output_basis in ["real", "spectral"]:
                    return input_vector

                elif input_basis == "real" and output_basis == "spectral":
                    # We are rescaling the vectors to unit vectors (with respect to the
                    # standard scalar product).
                    return input_vector * self.norms

                elif input_basis == "spectral" and output_basis == "real":
                    return input_vector / self.norms

                else:
                    message = (
                        "Unsupported space transformation from "
                        f"{input_basis} to {output_basis}."
                    )
                    raise ValueError(message)

            def scalar_product(self, lhs, rhs, input_basis="real"):
                return super().scalar_product(lhs, rhs, input_basis)

        new_self = deepcopy(spectrum)
        new_self.__class__ = TailoredElementwiseRealMultiplication
        return new_self
