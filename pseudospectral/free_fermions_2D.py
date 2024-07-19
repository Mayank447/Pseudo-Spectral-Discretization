import numpy as np

class FreeFermions2D:
    """
    Spectrum class to represent the spectrum (eigenfunctions, eigenvalues) of the 2D free fermions operator.
    Operator = (\sigma_z \partial_{t} + \sigma_x \partial_{x}) + (m * \Identity) + (\mu * \sigma_z)
                , where the sigmas are the Pauli matrices.
    
    The operator is discretized on a 2D lattice with n_t lattice points in the time axis and n_x lattice points in the x axis.
    We assume periodic boundary conditions in both directions with lengths L_t and L_x respectively.            

    Args:
        mu: Chemical potential (Fermi energy)
        m: mass parameter
        L_t: length of the system in the time axis
        L_x: length of the system in the x axis
        n_t: number of lattice points in the time axis
        n_x: number of lattice points in the x axis
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
        a, b = index
        return np.sqrt(self.mu**2 + (2 * self.mu * 1j * b) - (a**2 + b**2 + + self.m**2))
    

    def eigenfunctions(self, index):
        """
        Function to return the eigenfunctions of the 2D free fermions operator.
        """
        a, b = index
        return lambda x, t: np.exp(1j * (a * t) + (b * x))
    

    def transform(self, input_vector, input_space, output_space):
        """
        Function to transform the input vector according to the 2D free fermions operator between real and spectral spaces.
        """
        
        if input_space == 'real' and output_space == 'real':
            return input_vector
        
        elif input_space == 'spectral' and output_space == 'spectral':      
            return input_vector
        
        elif input_space == 'real' and output_space == 'spectral':
            return 0

        elif input_space == 'spectral' and output_space == 'real':
            return 0

        else:
            raise ValueError("Unsupported space transformation.")   