#!/usr/bin/env python3
from copy import deepcopy
import numpy as np


class ElementwiseSpectralMultiplication:
    """
    This spectrum represents elementwise multiplication of the spectral coefficients.
    In a way, it generalises the specialised spectra. For example, using
    spectrum.eigenvalues as field_values reproduces the original spectrum.
    """

    def __new__(cls, spectrum, field_values=1):
        class TailoredElementwiseSpectralMultiplication(type(spectrum)):
            @property
            def eigenvalues(self):
                return field_values * np.ones_like(super().eigenvalues)

        new_self = deepcopy(spectrum)
        new_self.__class__ = TailoredElementwiseSpectralMultiplication
        return new_self
