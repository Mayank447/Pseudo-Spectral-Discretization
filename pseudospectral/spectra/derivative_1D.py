import numpy as np
import scipy.fft


class Derivative1D:
    """
    Class to represent the eigenfunctions, eigenvalues of a 1D Derivative operator
    with finite and periodic boundary conditions.

    Args:
        num_lattice_points: Number of lattice points in the 1D domain
        L: Length of the periodic 1D domain
    """

    def __init__(self, num_lattice_points, L=1):
        self.num_lattice_points = num_lattice_points
        self.L = L
        self.eigenvalues = self._eigenvalues()

    def eigenfunction(self, index):
        """
        Private function to return the eigenfunctions of the 1D derivative operator i.e. exp(ikx)
        """
        return lambda x: np.exp(self.eigenvalues[index] * x)

    def _eigenvalues(self):
        """
        Private function to return the eigenvalues of the 1D derivative operator
        i.e. ik for the k-th eigenfunction exp(ikx) and k = 2*pi*m/L
        """
        return 1j * 2 * np.pi * np.arange(self.num_lattice_points) / self.L

    def transform(self, input_vector, input_basis, output_basis):
        """
        Transform the input vector according to 1D derivative operator
        from the given input space to the given output space.
        """
        if input_basis == "real" and output_basis == "real":
            return input_vector

        elif input_basis == "spectral" and output_basis == "spectral":
            return input_vector

        # Perform the discrete Fast Fourier transform to go from real to spectral space
        elif input_basis == "real" and output_basis == "spectral":
            return scipy.fft.fft(input_vector) / np.sqrt(self.num_lattice_points)

        # Perform the inverse discrete Fast Fourier transform to go from spectral to real space
        elif input_basis == "spectral" and output_basis == "real":
            return scipy.fft.ifft(input_vector) * np.sqrt(self.num_lattice_points)

        else:
            raise ValueError(
                f"Unsupported space transformation from {input_basis} to {output_basis}."
            )

    def lattice(self, output_basis):
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
            return 2 * np.pi * np.arange(self.num_lattice_points) / self.L

        else:
            raise ValueError("Unsupported output space.")
