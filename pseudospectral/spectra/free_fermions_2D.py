import numpy as np
import scipy

def cartisean_product(array_1, array_2):
    return np.array([[x,y] for x in array_1 for y in array_2])
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

    # array_t = [-(N-1)/2, ..., -1/2 , 1/2 , ..., (N-1)/2]
    # array_x = [-(N-1)/2, ..., -1, 0, 1, ..., (N-1)/2]
    """
    
    def __init__(self, mu, m, L_t, L_x, n_t, n_x):
        self.mu = mu
        self.m = m
        self.L_t = L_t
        self.L_x = L_x
        self.n_t = n_t
        self.n_x = n_x
        
        self._array_t = np.linspace(-(n_t-1)/2, (n_t-1)/2, n_t)
        self._array_x = np.linspace(-(n_x-1)/2, (n_x-1)/2, n_x)
        sign = [1, -1]
        self.eigenvalues = self._eigenvalues(np.meshgrid(self._array_t, self._array_x), sign)

    def _eigenvalues(self, index, sign):
        """
        Private function to return the eigenvalues of the 2D free fermions operator.
        """
        p_0, p_1 = index[0], index[1]
        p_0 = p_0 - self.mu
        return np.kron(sign, (1j * (2 * np.pi) * np.sqrt(p_0**2 + p_1**2)).flatten()) + self.m
    

    def eigenfunctions(self, index, sign):
        """
        Function to return the eigenfunctions of the 2D free fermions operator.
        """
        p_0, p_1 = index
        return lambda x, t: np.exp(1j * (2 * np.pi) * (self.n_t/self.L_t * (p_0 * t) + (self.n_x - 1)/self.L_x * p_1 * x)) * np.array([p_1, np.sqrt((p_0 - self.mu)**2 + p_1**2) + sign * (p_0 - self.mu)])
    

    def transform(self, input_vector, input_space, output_space):
        """
        Function to transform the input vector according to the 2D free fermions operator between real and spectral spaces.
        """
        
        if input_space == 'real' and output_space == 'real':
            return input_vector
        
        elif input_space == 'spectral' and output_space == 'spectral':      
            return input_vector
        
        elif input_space == 'real' and output_space == 'spectral':
            # Split the input vector into f(continuous upper vector component) and g(continuous lower vector component) 
            f, g = np.split(input_vector, 2)

            # Premultiplication of t since the boundary conditions are anti periodic and reshaping them to 2D
            f = np.repeat(np.exp(-1j * np.pi * self._array_t / self.L_t), self.n_x) * f
            g = np.repeat(np.exp(-1j * np.pi * self._array_t / self.L_t), self.n_x) * g
            f = f.reshape(self.n_t, self.n_x)
            g = g.reshape(self.n_t, self.n_x)

            # Perform the 2D discrete Fast Fourier transform to go from real to spectral space on both halves which are in discrete space after reshaping them
            f = scipy.fft.fft2(f)/(self.n_t * self.n_x)
            g = scipy.fft.fft2(g)/(self.n_t * self.n_x)
            
            # Reflatten and then return concatenate the two halves
            return np.join([f.flatten(), g.flatten()], axis=0)


        elif input_space == 'spectral' and output_space == 'real':
            # Split the input vector into f(continuous upper vector component) and g(continuous lower vector component) 
            f, g = np.split(input_vector, 2)

            # Premultiplication of t since the boundary conditions are anti periodic and reshaping them to 2D
            f = np.repeat(np.exp(1j * np.pi * self._array_t / self.L_t), self.n_x) * f
            g = np.repeat(np.exp(1j * np.pi * self._array_t / self.L_t), self.n_x) * g
            f = f.reshape(self.n_t, self.n_x)
            g = g.reshape(self.n_t, self.n_x)

            # Perform the 2D discrete Fast Fourier transform to go from real to spectral space on both halves which are in discrete space after reshaping them
            f = scipy.fft.ifft2(f)/(self.n_t * self.n_x)
            g = scipy.fft.ifft2(g)/(self.n_t * self.n_x)
            
            # Reflatten and then return concatenate the two halves
            return np.join([f.flatten(), g.flatten()], axis=0)

        else:
            raise ValueError(f"Unsupported space transformation from {input_space} to {output_space}.") 


if __name__ == "__main__":
    fermion = FreeFermions2D(0, 0, 4, 5, 4, 5)
    print(fermion.eigenvalues)