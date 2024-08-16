import numpy as np
import scipy.fft

I2PI = 1j * 2 * np.pi

class Derivative1D:
    """
    Class to represent the eigenfunctions, eigenvalues of a 1D Derivative operator
    on a finite interval with periodic boundary conditions.

    Args:
        num_lattice_points: Number of lattice points in the 1D domain
        L: Length of the periodic 1D domain
        theta: Real number in [0,1] as per the boundary condition (e.g. 0 for periodic, 0.5 for anti-periodic)
    """

    def __init__(self, num_lattice_points, L=1, theta=0):
        self.num_lattice_points = num_lattice_points
        self.L = L
        self.a = L / num_lattice_points
        self.theta = theta
        self.eigenvalues = self._eigenvalues()

    def _eigenvalues(self):
        """
        Private function to return the eigenvalues of the 1D derivative operator
        i.e. ik for the k-th eigenfunction exp(ikx) and k = 2*pi*m/L
        """
        return I2PI * (np.fft.fftfreq(self.num_lattice_points, d=self.a) + self.theta/self.L) 

    def eigenfunction(self, index: np.ndarray):
        """
        Function to return the eigenfunctions of the 1D derivative operator i.e. exp(ikx)
        """
        index = np.asarray(index)

        if (index >= self.num_lattice_points).any() or (index < -self.num_lattice_points).any():
            raise ValueError("Index out of bounds for the eigenfunction.")

        else:
            return lambda x: np.exp(self.eigenvalues[index] * x) / np.sqrt(self.L)

    def transform(self, input_vector, input_basis, output_basis):
        """
        Transform the input vector according to 1D derivative operator
        from the given input space to the given output space.
        """
        if input_basis == output_basis in ["real", "spectral"]:
            return input_vector

        # Perform the discrete Fast Fourier transform to go from real to spectral space
        elif input_basis == "real" and output_basis == "spectral":
            premultiplier =  np.exp(-I2PI * (self.theta/self.L) * self.lattice(output_basis="real"))
            return premultiplier * scipy.fft.fft(input_vector, norm="ortho") * np.sqrt(self.a)

        # Perform the inverse discrete Fast Fourier transform to go from spectral to real space
        elif input_basis == "spectral" and output_basis == "real":
            inv_premultiplier = np.exp(I2PI * (self.theta/self.L) * self.lattice(output_basis="spectral"))
            return scipy.fft.ifft(inv_premultiplier * input_vector, norm="ortho") / np.sqrt(self.a)

        else:
            raise ValueError(f"Unsupported space transformation from {input_basis} to {output_basis}.")


    def lattice(self, output_basis="real"):
        """
        Return the lattice of the Dirac operator as per the given output space.

        Parameters:
        output_basis (str): The space for which to generate the lattice.
                            Must be either 'real' or 'space'.

        Returns:
        np.ndarray: The lattice points in the specified output space.

        Raises:
        ValueError: If the output space is not 'real' or 'space'.
        """
        if output_basis == "real":
            return np.linspace(0, self.L, self.num_lattice_points, endpoint=False)

        elif output_basis == "spectral":
            return self.eigenvalues

        else:
            raise ValueError("Unsupported output space.")


    def scalar_product(self, lhs, rhs, input_basis="real"):
        """
        Compute <lhs, rhs> both being represented as coefficients in the `input_basis`.
        If multi-dimensional input is given,
        the last dimension gives the individual vector's entries
        while the first dimensions (all others) are interpreted as enumerating the multiple vectors.
        """
        # For this case the quadrature (and thereby the scalar product) is trivial.
        # Also, it's the same for both spaces.
        #
        # The unexpected ordering takes care of the indexing convention mentioned in the docstring:
        # We are considering lhs, rhs as row vectors in a matrix (or higher-dimensional object).
        # For this reason, we must commute them with respect to the normal interpretation as column vectors
        # (that's why rhs @ lhs and not lhs @ rhs).
        # Furthermore, the @ operator interpretes multi-dimensional objects as "stacks of matrices"
        # and accordingly acts on the LAST index of the first but the SECOND TO LAST index of the second.
        if input_basis == "real":
            return rhs @ lhs.transpose().conjugate() * self.a
        elif input_basis == "spectral":
            return rhs @ lhs.transpose().conjugate()
        else:
            raise ValueError(f"Unsupported input space - {input_basis}.")
