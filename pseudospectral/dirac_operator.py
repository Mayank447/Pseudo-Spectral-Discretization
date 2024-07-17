#!/usr/bin/env python3

import numpy as np
from spectrum import Derivative1D

class DiracOperator:
    """
    Dirac operator class to represent the Dirac operator in the pseudospectral method.
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
            temp_2 = np.diag(self.spectrum.eigenvalues) @ temp_1
            temp_3 = self.spectrum.transform(temp_2, 'spectral', output_space)
            return temp_3
        
        else:
            raise ValueError("Unsupported space transformation.")


    def lattice(self, output_space):
        """
        Return the lattice of the Dirac operator as per the given output space.
        """
        if output_space == 'real':
            return np.linspace(0, self.beta, self.n_real, endpoint=False)
        elif output_space == 'spectral':
            return 2 * np.pi * (np.arange(self.n_real) + 0.5) / self.beta
        else:
            raise ValueError("Unsupported output space.")
        

if __name__ == "__main__":
    spectrum = Derivative1D(10)
    operator = DiracOperator(spectrum)
    result = operator.apply_to([1,2,3,4,5,6,7,8,9,10], 'real', 'real')
    print(result)