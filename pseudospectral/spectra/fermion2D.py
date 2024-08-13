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
        X, T = np.meshgrid(self._lattice_x, self._lattice_t)
        X = X.flatten()
        T = T.flatten()

        self.p_t = I2PI * (T + (self.theta_t/self.L_t))
        self.p_x = I2PI * (X + (self.theta_x/self.L_x))
        self.p_t_mu = self.p_t - self.mu

        self.sqrt = np.sqrt(self.p_t_mu**2 + self.p_x**2)
        self.eigenvalues = self._eigenvalues()

        # Normalized eigenvector for ((p_t, p_x), (p_x, -p_t)) matrix as 4 scalar function of (p_x, p_t)
        self._norm_1 = np.sqrt(2 * self.sqrt * (self.sqrt - self.p_t_mu))
        self._norm_2 = np.sqrt(2 * self.sqrt * (self.sqrt + self.p_t_mu))
        self._norm_1[self.p_x == 0] = self.p_t_mu[self.p_x == 0]
        self._norm_2[self.p_x == 0] = self.p_t_mu[self.p_x == 0]

        self._eta_11 = self.p_x/self._norm_1
        self._eta_21 = (self.sqrt - self.p_t_mu)/self._norm_1
        self._eta_12 = self.p_x/self._norm_2
        self._eta_22 = (-self.sqrt - self.p_t_mu)/self._norm_2


    def _eigenvalues(self):
        """
        Private function to return the eigenvalues of the 2D free fermions operator.
        """
        return (
            (np.kron(self.sqrt, [1, -1])) + self.m
        )


    def eigenfunctions(self, index, sign):
        """
        Function to return the eigenfunctions of the 2D free fermions operator.
        Note: p_t, p_x used in the function are scalars and not arrays.
        
        Args:
            index: 2D index of the eigenfunction
            sign: +-1 for the two eigenvectors

        Returns:
            lambda function (t,x) of the eigenfunction at the specified index

        Note: t,x passed to the returned lambda function must be flattened out meshgrid.
        e.g. x, t = np.meshgrid(array_x, array_t)
             t = t.flatten()
             x = x.flatten()
        """

        offset = self.n_x * index[0]
        p_t = self.p_t[offset]
        p_x = self.p_x[offset + index[1]]
        p_t_mu = p_t - self.mu

        normalization = None
        array = np.eye(2)[int(0.5 * (sign-1))]

        if p_x != 0:
            normalization = np.sqrt(
                2 * np.sqrt(p_t_mu**2 + p_x**2) 
                * (np.sqrt(p_t_mu**2 + p_x**2) - sign * p_t_mu)
            )

            array = np.array([p_x, 
                              sign * np.sqrt(p_t_mu**2 + p_x**2) - p_t_mu
                              ]) / normalization

        return (
            lambda t, x: np.kron(
                np.exp(
                    p_t * t +  p_x * x
                )
                / np.sqrt(self.n_t * self.n_x)
            , array)
        )


    def transform(self, input_vector, input_basis, output_basis):
        """
        Function to transform the input vector according to the 2D free fermions operator between real and spectral spaces.
        """

        if input_basis == output_basis in ["real", "spectral"]:
            return input_vector

        elif input_basis == "real" and output_basis == "spectral":
            # Split the input vector into f(even index elements) and g(odd index elements)
            f, g = input_vector[0::2], input_vector[1::2]

            # Transform the two halves to spectral space
            f = self._real_to_spectral(f, self._eta_11, self._eta_21)
            g = self._real_to_spectral(g, self._eta_12, self._eta_22)

            # Reflatten and then return the array with elements alternatively concatenated
            return np.ravel([f.flatten(), g.flatten()],'F')


        elif input_basis == "spectral" and output_basis == "real":
            # Split the input vector into f(even index elements) and g(odd index elements)
            f, g = input_vector[0::2], input_vector[1::2]

            # Transform the two halves to spectral space
            f = self._spectral_to_real(f, self._eta_11, self._eta_12)
            g = self._spectral_to_real(g, self._eta_21, self._eta_22)

            # Reflatten and then return the array with elements alternatively concatenated
            return np.ravel([f.flatten(), g.flatten()],'F')

        else:
            raise ValueError(
                f"Unsupported space transformation from {input_basis} to {output_basis}."
            )


    def _real_to_spectral(self, real_vector, eta_1, eta_2):
        """
        Private function to transform a vector from real space to spectral space.
        """
        # Premultiplication factor in variable t since the boundary conditions are anti periodic
        premultiplier = np.exp(-I2PI * self.theta_t * np.arange(self.n_t)/self.L_t)
        real_vector = (
            premultiplier[:, np.newaxis] * real_vector.reshape(-1, self.n_t)
        )

        # Premultiplication factor in variable x since the boundary conditions may not be periodic
        real_vector = (
            real_vector *
            np.repeat(np.exp(-I2PI * self.theta_x * np.arange(self.n_x)/self.L_x), self.n_t)
        )

        # Reshaping to 2D and performing the 2D discrete Fast Fourier transform to go from real to spectral space
        real_vector = real_vector.reshape(self.n_t, self.n_x)
        real_vector = scipy.fft.fft2(real_vector, norm="ortho").flatten()

        # Post multiplication by block diagonalized eigenvector matrix transpose
        real_vector = eta_1 * real_vector + eta_2 * real_vector
        return real_vector
    

    def _spectal_to_real(self, spectral_vector, eta_1, eta_2):
        """
        Private function to transform a vector in spectral space to real space
        """

        # Reshaping to 2D and performing the 2D discrete Inverse Fast Fourier transform to go from spectral to real space
        spectral_vector = spectral_vector.reshape(self.n_t, self.n_x)
        spectral_vector = scipy.fft.ifft2(spectral_vector, norm="ortho")

        # Premultiplication in variable t since the boundary conditions are anti periodic
        spectral_vector = (
            spectral_vector *
            np.repeat(np.exp(I2PI * self.theta_t * np.arange(self.n_t)/self.L_t), self.n_x)
        )

        # Premultiplication in variable x since the boundary conditions may not be periodic
        spectral_vector = (
            spectral_vector *
            np.repeat(np.exp(I2PI * self.theta_x * np.arange(self.n_x)/self.L_x), self.n_t)
        )

        # Post multiplication by block diagonalized eigenvector matrix
        spectral_vector = eta_1 * spectral_vector + eta_2 * spectral_vector
        return spectral_vector
    

    def _operate(f,g, p_t, p_x):
        return (p_t * f + p_x * g, p_x * f - p_t * g) 

    def direct_apply(self, vector):
        for i in range(self.n_t * self.n_x):
            vector[i, i+1] = self._operate(vector[i], vector[i+1], self.p_t_mu[i], self.p_x[i])
        return vector


if __name__ == "__main__":
    fermion = FreeFermions2D(n_t=7, n_x=5, L_t=1, L_x=1, mu=0, m=0, theta_t=0.5, theta_x=0)
    
    x, t = np.meshgrid(
        np.linspace(0, fermion.L_x, fermion.n_x, endpoint=False), 
        np.linspace(0, fermion.L_t, fermion.n_t, endpoint=False)
    )
    t = t.flatten()
    x = x.flatten()
    print("t", t)
    print("x", x)
    e1 = fermion.eigenfunctions([0, 0], 1)(t,x)
    print(e1)
    print(np.linalg.norm(e1))