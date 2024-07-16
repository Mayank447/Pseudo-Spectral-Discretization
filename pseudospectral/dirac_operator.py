#!/usr/bin/env python3

import numpy as np

class DiracOperator:
    """
    Dirac operator class to represent the Dirac operator in the pseudospectral method.
    """

    def __init__(self, n_time, nu, n_landau):
        self.n_time = n_time
        self.nu = nu
        self.n_landau = n_landau
        self.beta = 1.0  # Assuming beta is 1 for simplicity, can be parameterized


    def apply_to(self, spectral_coefficients, input_space, output_space):
        """
        Apply the Dirac operator to the given spectral coefficients as per the given input and output spaces.
        """
        if input_space == 'time' and output_space == 'frequency':
            transformed_coefficients = self.transform(spectral_coefficients, 'time', 'frequency')
            return self.eigenvalue(0) * transformed_coefficients
        elif input_space == 'frequency' and output_space == 'time':
            transformed_coefficients = self.transform(spectral_coefficients, 'frequency', 'time')
            return self.eigenvalue(0) * transformed_coefficients
        else:
            raise ValueError("Unsupported space transformation.")


    def eigenvalue(self, index):
        """
        Return the eigenvalue of the Dirac operator at the given index.
        """
        omega_m = 2 * np.pi * (index + 0.5) / self.beta  # Matsubara frequencies
        return np.sqrt(omega_m**2 + self.nu**2)


    def eigenfunction(self, index, output_space):
        """
        Return the eigenfunction of the Dirac operator at the given index as per the given space.
        """
        if output_space == 'time':
            return np.sin((index + 1) * np.pi * np.linspace(0, 1, self.n_time))
        elif output_space == 'frequency':
            return np.exp(1j * (index + 0.5) * 2 * np.pi * np.arange(self.n_time) / self.beta)
        else:
            raise ValueError("Unsupported output space.")


    def lattice(self, output_space):
        """
        Return the lattice of the Dirac operator as per the given output space.
        """
        if output_space == 'time':
            return np.linspace(0, self.beta, self.n_time, endpoint=False)
        elif output_space == 'frequency':
            return 2 * np.pi * (np.arange(self.n_time) + 0.5) / self.beta
        else:
            raise ValueError("Unsupported output space.")


    def transform(self, values, input_space, output_space):
        """
        Transform the given values from the input space to the output space.
        """
        if input_space == 'time' and output_space == 'frequency':
            # Perform the discrete Fourier transform to go from time to frequency space
            return np.fft.fft(values) / np.sqrt(self.n_time)
        
        elif input_space == 'frequency' and output_space == 'time':
            # Perform the inverse discrete Fourier transform to go from frequency to time space
            return np.fft.ifft(values) * np.sqrt(self.n_time)
        
        else:
            raise ValueError("Unsupported space transformation.")