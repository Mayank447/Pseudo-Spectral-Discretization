import numpy as np
import scipy

I2PI = 1j * 2 * np.pi


class FreeFermions2D:
    """
    Spectrum class to represent the spectrum (eigenfunctions, eigenvalues) of the 2D free fermions operator.
    Operator = (\sigma_z \partial_{t} + \sigma_x \partial_{x}) + (m * \Identity) - (\mu * \sigma_z)
                , where the sigmas are the Pauli matrices.

    The operator is discretized on p_t 2D lattice with n_t lattice points in the time axis and n_x lattice points in the x axis.
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
    sign = +-1
    """

    def __init__(self, n_t, n_x, L_t=1, L_x=1, mu=0, m=0):
        self.mu = mu
        self.m = m
        self.L_t = L_t
        self.L_x = L_x
        self.n_t = n_t
        self.n_x = n_x

        # self._array_t = np.linspace(-(n_t - 1) / 2, (n_t - 1) / 2, n_t)
        self._array_t = scipy.fft.fftfreq(n_t, d=1 / L_t) - 0.5
        self._array_x = scipy.fft.fftfreq(n_x, d=1 / L_x)
        self.meshgrid = np.meshgrid(self._array_t, self._array_x)
        self.eigenvalues = self._eigenvalues(self.meshgrid[0], self.meshgrid[1])

        self.p_t = I2PI * (self.meshgrid[0] - 0.5) / self.L_t
        self.p_x = I2PI * np.pi * self.meshgrid[1] / self.L_x
        self.p_t_mu = self.p_t - self.mu

        sq = self.p_t_mu**2 + self.p_x**2
        sqrt = np.sqrt(sq)
        normalization = 1 / np.sqrt(2 * sq - 2 * self.p_t * sqrt)

        self._eta_0 = normalization * self.p_x
        self._eta_1 = normalization * -self.p_t

    def _eigenvalues(self, p_t, p_x):
        """
        Private function to return the eigenvalues of the 2D free fermions operator.
        """
        p_t = p_t - self.mu
        return (
            1j * (2 * np.pi) * np.kron([1, -1], (np.sqrt(p_t**2 + p_x**2))).flatten()
            + self.m
        )

    def eigenfunctions(self, index, sign):
        """
        Function to return the eigenfunctions of the 2D free fermions operator.
        """
        p_t, p_x = index
        p_t_mu = p_t - self.mu
        normalization = np.sqrt(
            2 * (p_t_mu**2 + p_x**2) - 2 * p_t_mu * np.sqrt(p_t_mu**2 + p_x**2)
        )

        return (
            lambda x, t: np.exp(
                1j
                * (2 * np.pi)
                * (
                    self.n_t / self.L_t * (p_t * t)
                    + (self.n_x - 1) / self.L_x * p_x * x
                )
            )
            * normalization
            * np.array([p_x, sign * np.sqrt(p_t_mu**2 + p_x**2) - p_t_mu])
        )

    def _real_to_spectral(self, real_vector, eta):
        """
        Private function to transform a real vector to spectral space.
        """
        # Premultiplication of t since the boundary conditions are anti periodic and reshaping them to 2D
        real_vector = (
            np.repeat(np.exp(-1j * np.pi * self._array_t / self.L_t), self.n_x)
            * real_vector
        )

        # Perform the 2D discrete Fast Fourier transform to go from real to spectral space on both halves which are in discrete space after reshaping them
        real_vector = real_vector.reshape(self.n_t, self.n_x)
        real_vector = scipy.fft.fft2(real_vector, norm="ortho")

        return real_vector

    def transform(self, input_vector, input_basis, output_basis):
        """
        Function to transform the input vector according to the 2D free fermions operator between real and spectral spaces.
        """

        if input_basis == output_basis in ["real", "spectral"]:
            return input_vector

        elif input_basis == "real" and output_basis == "spectral":
            # Split the input vector into f(continuous upper vector component) and g(continuous lower vector component)
            f, g = np.split(input_vector, 2)

            # Transform the two halves to spectral space
            f = self._real_to_spectral(f)
            g = self._real_to_spectral(g)

            # Multiply by respective eta scaler field and then flatten the 2D array
            f = self._eta_0 * f

            # Reflatten and then return concatenate the two halves
            return np.join([f.flatten(), g.flatten()], axis=0)

        elif input_basis == "spectral" and output_basis == "real":
            # Split the input vector into f(continuous upper vector component) and g(continuous lower vector component)
            f, g = np.split(input_vector, 2)

            # Premultiplication of t since the boundary conditions are anti periodic and reshaping them to 2D
            f = np.repeat(np.exp(1j * np.pi * self._array_t / self.L_t), self.n_x) * f
            g = np.repeat(np.exp(1j * np.pi * self._array_t / self.L_t), self.n_x) * g
            f = f.reshape(self.n_t, self.n_x)
            g = g.reshape(self.n_t, self.n_x)

            # Perform the 2D discrete Fast Fourier transform to go from real to spectral space on both halves which are in discrete space after reshaping them
            f = scipy.fft.ifft2(f, norm="ortho")
            g = scipy.fft.ifft2(g, norm="ortho")

            # Reflatten and then return concatenate the two halves
            return np.join([f.flatten(), g.flatten()], axis=0)

        else:
            raise ValueError(
                f"Unsupported space transformation from {input_basis} to {output_basis}."
            )


if __name__ == "__main__":
    fermion = FreeFermions2D(0, 0, 4, 5, 4, 5)
    # print(fermion.eigenvalues)
    # print(fermion.eigenfunctions([1, 1], 1)(0,0))
