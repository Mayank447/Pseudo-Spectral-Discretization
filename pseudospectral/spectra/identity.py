#!/usr/bin/env python3
import numpy as np


class Identity:
    """
    This spectrum represents the identity operator in the vector space of the given
    spectrum.
    """

    def __init__(self, spectrum):
        self.eigenfunction = spectrum.eigenfunction
        self.transform = spectrum.transform
        self.scalar_product = spectrum.scalar_product
        self.eigenvalues = np.ones_like(spectrum.eigenvalues)
