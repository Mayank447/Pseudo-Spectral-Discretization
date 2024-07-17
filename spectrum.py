import numpy as np

class Derivative1D:
    def __init__(self, num_lattice_points):
        self.num_lattice_point = num_lattice_points
        self.L = 1
        self.eigenvalues = self._eigenvalue()

    def transform(self, input_vector, input_space, output_space):
        if input_space == 'real' and output_space == 'real':
            return input_vector

        # Perform the discrete Fast Fourier transform to go from real to spectral space
        elif input_space == 'real' and output_space == 'spectral':
            return np.fft.fft(input_vector) / np.sqrt(self.n_real)
        
        elif input_space == 'spectral' and output_space == 'spectral':
            return input_vector
        
        elif input_space == 'spectral' and output_space == 'real':
            # Perform the inverse discrete Fast Fourier transform to go from spectral to real space
            return np.fft.ifft(input_vector) * np.sqrt(self.n_real)
        
        else:
            raise ValueError("Unsupported space transformation.")
        
    
    def _eigenvalues(self):
        return 1j * 2 * np.pi * np.arange(self.num_lattice_points) / self.L
    
    def _eigenfunction(self, index):
        return lambda x: np.exp(self.eigenvalues[index] * x)