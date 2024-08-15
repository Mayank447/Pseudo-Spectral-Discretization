import numpy as np
import scipy

I2PI = 1j * 2 * np.pi


class FreeFermion2D:
    r"""
    Spectrum class to represent the spectrum (eigenfunction, eigenvalues) of the 2D free fermions operator.
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
        self.vector_length = 2 * n_t * n_x

        self._freq_t = scipy.fft.fftfreq(n_t, d=self.a_t)
        self._freq_x = scipy.fft.fftfreq(n_x, d=self.a_x)
        X, T = np.meshgrid(self._freq_x, self._freq_t)
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
        self._norm_1[self.p_x == 0] = 1
        self._norm_2[self.p_x == 0] = 1

        self._eta_11 = self.p_x/self._norm_1
        self._eta_21 = (self.sqrt - self.p_t_mu)/self._norm_1
        self._eta_12 = self.p_x/self._norm_2
        self._eta_22 = (-self.sqrt - self.p_t_mu)/self._norm_2
        self._eta_11[self.p_x == 0] = 1 
        # print(self._eta_21[self.p_x==0])
        self._eta_12[self.p_x == 0] = 0 
        self._eta_21[self.p_x == 0] = 0 
        self._eta_22[self.p_x == 0] = 1

    def _eigenvalues(self):
        """
        Private function to return the list of eigenvalues (the diagonal of the eigenvalue matrix)
        of the 2D Free Fermion operator 
        
        Args:
            None
            
        Returns:
            numpy.ndarray: 1D array of eigenvalues
        """
        return (
            (np.kron(self.sqrt, [1, -1])) + self.m
        )

    def eigenfunction(self, index):
        """
        Function to return the eigenfunction of the 2D free fermions operator.
        Note: p_t, p_x used in the function are scalars and not arrays.
        
        Args:
            index: 2D index of the eigenfunction (index_t, index_x)
            sign: +-1 for the two eigenvectors

        Returns:
            lambda function (t,x) of the eigenfunction at the specified index

        Note: t,x passed to the returned lambda function must be flattened out meshgrid.
        e.g. x, t = np.meshgrid(array_x, array_t)
             t = t.flatten()
             x = x.flatten()
        """
        
        index = np.atleast_1d(index)

        if (index >= 2 * self.n_t * self.n_x).any() or (index < -2 * self.n_t * self.n_x).any():
            raise ValueError(f"Index {index} out of bounds.")
        
        p_t = self.p_t[index//2]
        p_x = self.p_x[index//2]
        p_t_mu = p_t - self.mu

        normalization = np.ones(len(index))
        sign = np.array(1 - 2*(index % 2))
        eta = np.eye(2)[(0.5 * (sign-1)).astype(int)]
        
        sq = np.sqrt(p_t_mu**2 + p_x**2)
        normalization =  np.sqrt(
            2 * sq 
            * (sq - (sign * p_t_mu))
        )

        mask = (p_x == 0)
        normalization[mask] = 1
        
        eta = np.array([p_x/ normalization, 
                        (sign * sq - p_t_mu)/ normalization
                    ]).transpose()
        eta[np.logical_and(mask, sign==1)] = np.array([1, 0])
        eta[np.logical_and(mask, sign==-1)] = np.array([0, 1])

        return (
            lambda t, x: self._return_eigenfunction(t, x, index, eta, p_t, p_x)
        )
    
    def _return_eigenfunction(self, t, x, index, eta, p_t, p_x):
        """
        Return function when eigenfucntion method is called.
        """

        #This part {np.kron(p_t, t)} can be done more efficiently since some values of p_x, p_t repeat
        exp = np.exp(
                (np.kron(p_t, t) +  np.kron(p_x, x)) / np.sqrt(self.L_t * self.L_x)
            ).reshape(len(index), -1)
        
        # Initialize the return array
        ret = [0]*len(index) 
        for i in range(len(index)):
             # Kronecker product between each spinor array and corresponding exponential part
            ret[i] = np.kron(exp[i], eta[i])
        
        return np.array(ret)


    def transform(self, input_vector, input_basis, output_basis):
        """
        Function to transform the input vector according to the 2D free fermions operator between real and spectral spaces.
        
        Args:
            input_vector: input vector to be transformed
            input_basis: basis of the input vector (real/spectral)
            output_basis: basis of the output vector (real/spectral)

        Returns:
            transformed vector in the specified output basis

        Raises:
            ValueError: if the input_basis or output_basis is not supported.    

        Note: The input vector must be a 1D array of size 2 * n_t * n_x. (Check for that)
        """

        if input_basis == output_basis in ["real", "spectral"]:
            return input_vector

        elif input_basis == "real" and output_basis == "spectral":
            # Split the input vector into f(even index elements) and g(odd index elements)
            f, g = input_vector[0::2], input_vector[1::2]

            # Transform the two halves to spectral space
            f = self._real_to_spectral(f)
            g = self._real_to_spectral(g)

            # Post multiplication by block diagonalized eigenvector matrix transpose
            f = np.ravel([self._eta_11 * f, self._eta_12 * f], 'F')
            g = np.ravel([self._eta_21 * g, self._eta_22 * g], 'F')
            return f + g
        

        elif input_basis == "spectral" and output_basis == "real":
            
            # Block diagonal multiplication of eigenvector matrix
            f = self._eta_11 * input_vector[::2] + self._eta_12 * input_vector[1::2]
            g = self._eta_21 * input_vector[::2] + self._eta_22 * input_vector[1::2]

            # Transform the two halves to spectral space
            f = self._spectral_to_real(f)
            g = self._spectral_to_real(g)

            # Reflatten and then return the array with elements alternatively concatenated
            return np.ravel([f, g],'F')

        else:
            raise ValueError(
                f"Unsupported space transformation from {input_basis} to {output_basis}."
            )

    def _real_to_spectral(self, coeff):
        """
        Private function to transform a vector from real space to spectral space.
        Use by the public transform function.

        Args:
            coeff (numpy.ndarray): input vector in real space
            [Note: The input vector must be a 1D array of size n_t * n_x.]
        
        Returns:
            vector in spectral space
        """
        
        # Premultiplication factor in variable t, x since the boundary conditions may not be periodic
        premultiplier_t = np.exp(-I2PI * (self.theta_t/self.L_t) * np.arange(self.n_t))
        premultiplier_x = np.exp(-I2PI * (self.theta_x/self.L_x) * np.arange(self.n_x))
        coeff = (
            premultiplier_t[ :, np.newaxis] * coeff.reshape(self.n_t, self.n_x)
        )
        coeff = (
            premultiplier_x[np.newaxis, : ] * coeff
        )

        # Performing the 2D discrete Fast Fourier transform to go from real to spectral space
        coeff = coeff.reshape(self.n_t, self.n_x)
        coeff = scipy.fft.fft2(coeff, norm="ortho") * np.sqrt(self.a_t * self.a_x)
        return coeff.flatten()
    

    def _spectral_to_real(self, coeff):
        """
        Private function to transform a vector in spectral space to real space
        """

        # Reshaping to 2D and performing the 2D discrete Inverse FFT to go from spectral to real space
        coeff = coeff.reshape(self.n_t, self.n_x)
        coeff = scipy.fft.ifft2(coeff, norm="ortho") * np.sqrt(self.a_t * self.a_x)

        # Reversing premultiplication in variable t, x since the boundary conditions may not be periodic
        inv_premultiplier_t = np.exp(I2PI * (self.theta_t/self.L_t) * np.arange(self.n_t))
        inv_premultiplier_x = np.exp(I2PI * (self.theta_x/self.L_x) * np.arange(self.n_x))
        coeff = (
            inv_premultiplier_t[ :, np.newaxis] * coeff.reshape(self.n_t, -1)
        )
        coeff = (
            inv_premultiplier_x[np.newaxis, : ] * coeff
        )

        return coeff.flatten()

    def _operate(f,g, p_t, p_x):
        return (p_t * f + p_x * g, p_x * f - p_t * g) 

    def direct_apply(self, vector):
        for i in range(self.n_t * self.n_x):
            vector[i, i+1] = self._operate(vector[i], vector[i+1], self.p_t_mu[i], self.p_x[i])
        return vector
    
    def scalar_product(self, lhs, rhs, input_basis="real"):
        """
        Function to compute the scalar product of two vectors in the specified basis.
        
        Args:
            lhs: left hand side vector
            rhs: right hand side vector
            input_basis: basis of the input vectors (real/spectral)
        """
        if input_basis == "real":
            return lhs @ rhs.transpose(-1,-2).conjugate() * self.a_t * self.a_x
        
        elif input_basis == "spectral":
            return np.sum(lhs.conjugate * rhs)
        
        else:
            raise ValueError(f"Unsupported input space - {input_basis}.")


    def lattice(self, output_basis='real'):
        """
        Function to return the lattice points in the specified output basis.
        
        Args:
            output_basis: basis of the output lattice points (real/spectral)
        
        Returns:
            (numpy.ndarray): lattice points in the specified output basis
        
        Raises:
        ValueError: If the output space is not 'real' or 'space'.
        """

        if output_basis == "real":
            t, x = np.meshgrid(
                np.linspace(0, self.L_t, self.n_t, endpoint=False), 
                np.linspace(0, self.L_x, self.n_x, endpoint=False), indexing="ij"
            )
            return t.flatten(), x.flatten()
        
        elif output_basis == "spectral":
            return self.eigenvalues
        else:
            raise ValueError(f"Unsupported output space - {output_basis}.")


if __name__ == "__main__":
    fermion = FreeFermion2D(n_t=7, n_x=5, L_t=1, L_x=1, mu=0, m=0, theta_t=0.5, theta_x=0)
    
    x, t = np.meshgrid(
        np.linspace(0, fermion.L_x, fermion.n_x, endpoint=False), 
        np.linspace(0, fermion.L_t, fermion.n_t, endpoint=False)
    )
    t = t.flatten()
    x = x.flatten()
    e1 = fermion.eigenfunction([1, 3], -1)(t,x)
    s1 = fermion.transform(e1, "real", "spectral")
    e1_ = fermion.transform(s1, "spectral", "real")
    assert np.isclose(e1, e1_).all()
    print(np.linalg.norm(e1_ - e1))