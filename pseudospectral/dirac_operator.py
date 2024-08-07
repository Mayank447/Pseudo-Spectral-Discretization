#!/usr/bin/env python3

import numpy as np

class DiracOperator:
    """
    Dirac operator class to represent the Dirac operator in the pseudospectral method.
    Args: Spectrum object
    """

    def __init__(self, spectrum):
        """
        Initialize the Dirac operator with the given spectrum.
        """
        self.spectrum = spectrum


    def apply_to(self, coefficients, input_space='real', output_space='real'):
        """
        Apply the Dirac operator to the given spectral coefficients as per the given input and output spaces.
        """

        if (input_space in ['real', 'spectral']) and (output_space in ['real', 'spectral']):
            temp_1 = self.spectrum.transform(coefficients, input_space, 'spectral')
            temp_2 = self.spectrum.eigenvalues * temp_1
            temp_3 = self.spectrum.transform(temp_2, 'spectral', output_space)
            return temp_3
        
        else:
            raise ValueError(f"Unsupported space transformation from {input_space} to {output_space}.")