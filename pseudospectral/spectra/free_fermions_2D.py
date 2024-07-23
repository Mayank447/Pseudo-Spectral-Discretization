import numpy as np

class FreeFermions2D:
    """
    Spectrum class to represent the spectrum (eigenfunctions, eigenvalues) of the 2D free fermions operator.
    Operator = (\sigma_z \partial_{t} + \sigma_x \partial_{x}) + (m * \Identity) + (\mu * \sigma_z)
                , where the sigmas are the Pauli matrices.
    
    The operator is discretized on p_0 2D lattice with n_t lattice points in the time axis and n_x lattice points in the x axis.
    We assume periodic boundary conditions in both directions with lengths L_t and L_x respectively.            

    Args:
        mu: Chemical potential (Fermi energy)
        m: mass parameter
        L_t: length of the system in the time axis
        L_x: length of the system in the x axis 
        n_t: number of lattice points in the time axis (even)
        n_x: number of lattice points in the x axis (odd)
    """
    
    def __init__(self, mu, m, L_t, L_x, n_t, n_x):
        self.mu = mu
        self.m = m
        self.L_t = L_t
        self.L_x = L_x
        self.n_x = n_t
        self.n_y = n_x
        self._eigenvalues = self._eigenvalues()

    def _eigenvalues(self, index):
        """
        Private function to return the eigenvalues of the 2D free fermions operator.
        """
        p_0, p_1 = index
        return np.sqrt(self.mu**2 + (2 * self.mu * 1j * p_1) - (p_0**2 + p_1**2 + + self.m**2))
    

    def eigenfunctions(self, index):
        """
        Function to return the eigenfunctions of the 2D free fermions operator.
        """
        p_0, p_1 = index
        return lambda x, t: np.exp(1j * ((p_0 - self.mu) * t) + (p_1 * x))
    

    def transform(self, input_vector, input_space, output_space):
        """
        Function to transform the input vector according to the 2D free fermions operator between real and spectral spaces.
        """
        
        if input_space == 'real' and output_space == 'real':
            return input_vector
        
        elif input_space == 'spectral' and output_space == 'spectral':      
            return input_vector
        
        elif input_space == 'real' and output_space == 'spectral':
            return np.fft.fft2(input_vector) / (self.n_t * self.n_x)

        elif input_space == 'spectral' and output_space == 'real':
            return np.fft.ifft2(input_vector) * (self.n_t * self.n_x)

        else:
            raise ValueError("Unsupported space transformation.") 


if __name__ == "__main__":
    n_t = [-(N-1)/2, ..., -1/2 , 1/2 , ..., (N-1)/2]
    n_x = [-(N-1)/2, ..., -1, 0, 1, ..., (N-1)/2]