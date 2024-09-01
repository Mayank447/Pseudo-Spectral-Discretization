#!/usr/bin/env python3
from copy import deepcopy
import numpy as np


class ElementwiseSpectralMultiplication:
    """
    This spectrum represents the identity operator in the vector space of the given
    spectrum.
    """

    def __new__(cls, spectrum):
        class TailoredIdentity(type(spectrum)):
            @property
            def eigenvalues(self):
                return np.ones_like(super().eigenvalues)

        new_self = deepcopy(spectrum)
        new_self.__class__ = TailoredIdentity
        return new_self
