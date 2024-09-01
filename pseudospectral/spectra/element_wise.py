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
        class TailoredElementwiseSpectralMultiplication(type(spectrum)):
            field_values = given_field_values

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

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
    """

    def __new__(cls, spectrum, given_field_values=1):
        class TailoredElementwiseRealMultiplication(type(spectrum)):
            field_values = given_field_values

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.spectral_coefficients = super().transform(
                    np.eye(super().total_num_of_dof), "real", "spectral"
                )

            @property
            def eigenvalues(self):
                return self.field_values * np.ones_like(super().eigenvalues)

            def eigenfunction(self, index):
                # This is important: Our eigenfunctions are not at all trivial. They are
                # general elements from our vector space that just happen to vanish at
                # all lattice points. It follows that we need to consider an appropriate
                # linear combination of the original eigenfunctions.
                return lambda *x: np.sum(
                    super().eigenfunction(self.spectral_coefficients[:, index])(*x),
                    axis=0,
                )

            def transform(self, coefficients, input_basis="real", output_basis="real"):
                # We're already diagonal, so our spectral basis is the same as the real
                # one.
                return coefficients

        new_self = deepcopy(spectrum)
        new_self.__class__ = TailoredElementwiseRealMultiplication
        return new_self
