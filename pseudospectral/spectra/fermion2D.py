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

    def __init__(self, n_t, n_x, L_t=1, L_x=1, mu=0, m=0, theta_t=0.5, theta_x=0):
        self.mu = mu
        self.m = m
        self.theta_t = theta_t
        self.theta_x = theta_x
        self.L_t = L_t
        self.L_x = L_x
        self.n_t = n_t
        self.n_x = n_x
        self.a_t = L_t/n_t
        self.a_x = L_x/n_x

        self._lattice_t = scipy.fft.fftfreq(n_t, d=self.a_t)
        self._lattice_x = scipy.fft.fftfreq(n_x, d=self.a_x)
        T, X = np.meshgrid(self._lattice_t, self._lattice_x)

        self.p_t = I2PI * (T - (self.theta_t/self.L_t))
        self.p_x = I2PI * (X - (self.theta_x/self.L_x))
        self.p_t_mu = self.p_t - self.mu
        
        self.sqrt = np.sqrt(self.p_t_mu**2 + self.p_x**2)
        self.eigenvalues = self._eigenvalues()


    def _eigenvalues(self):
        """
        Private function to return the eigenvalues of the 2D free fermions operator.
        """
        return (
            (I2PI * np.kron(self.sqrt.flatten(), [1, -1])) + self.m
        )

    def eigenfunctions(self, index, sign):
        """
        Function to return the eigenfunctions of the 2D free fermions operator.
        """
        p_t = self.p_t[index[0]]
        p_x = self.p_x[index[1]]
        p_t_mu = p_t - self.mu

        normalization = np.sqrt(
            2 * np.sqrt(p_t_mu**2 + p_x**2) * (np.sqrt(p_t_mu**2 + p_x**2) - sign * p_t_mu)
        )
        array = np.array(
            [p_x, sign * np.sqrt(p_t_mu**2 + p_x**2) - p_t_mu]
        ) / normalization

        return (
            lambda x, t: np.exp(
                p_t * t +  p_x * x
            ) 
            / np.sqrt(self.n_t * self.n_x)
            * array
        )


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
            f = np.repeat(np.exp(1j * np.pi * self._lattice_t / self.L_t), self.n_x) * f
            g = np.repeat(np.exp(1j * np.pi * self._lattice_t / self.L_t), self.n_x) * g
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


    def _real_to_spectral(self, real_vector, eta):
        """
        Private function to transform a real vector to spectral space.
        """
        # Premultiplication of t since the boundary conditions are anti periodic and reshaping them to 2D
        real_vector = (
            np.repeat(np.exp(-1j * np.pi * self._lattice_t / self.L_t), self.n_x)
            * real_vector
        )

        # Perform the 2D discrete Fast Fourier transform to go from real to spectral space on both halves which are in discrete space after reshaping them
        real_vector = real_vector.reshape(self.n_t, self.n_x)
        real_vector = scipy.fft.fft2(real_vector, norm="ortho")

        return real_vector



if __name__ == "__main__":
    fermion = FreeFermions2D(0, 0, 4, 5, 4, 5)
    # print(fermion.eigenvalues)
    # print(fermion.eigenfunctions([1, 1], 1)(0,0))
