#!/usr/bin/env python3
import numpy as np

I2PI = 1j * 2 * np.pi


class FreeFermion2D:
    r"""
    Spectrum class to represent the spectrum (eigenfunction, eigenvalues) of the 2D free fermions operator.
    Operator = (\sigma_z \partial_{t} + \sigma_x \partial_{x}) + (m * \Identity) - (\mu * \sigma_z)
                , where the sigmas are the Pauli matrices.

    The operator is discretized on a 2D lattice with n_t lattice points in the time axis and n_x lattice points in the x axis.
    We assume periodic boundary conditions in both directions with lengths L_t and L_x respectively.

    Args:
        mu: Chemical potential (Fermi energy)
        m: mass parameter
        L_t: length of the system in the time axis
        L_x: length of the system in the x axis
        n_t: number of lattice points in the time axis (odd)
        n_x: number of lattice points in the x axis (odd)
    """

    def __init__(self, n_t, n_x, L_t=1, L_x=1, mu=0, m=0):
        self._initialise_members([n_t, n_x], [L_t, L_x], mu, m)
        self._compute_grids()

        self.p_t_mu = self.p_t - self.mu
        self.norm_p = 1.0j * np.linalg.norm([self.p_t_mu, self.p_x], axis=0)

        self._solve_spectral_problem_in_spinor_space()

    def _solve_spectral_problem_in_spinor_space(self):
        # Normalized eigenvector for ((p_t, p_x), (p_x, -p_t)) matrix as 4 scalar function of (p_x, p_t)
        _norm_1 = np.sqrt(2 * self.norm_p * (self.norm_p - self.p_t_mu))
        _norm_2 = np.sqrt(2 * self.norm_p * (self.norm_p + self.p_t_mu))
        _norm_1[self.p_x == 0] = 1
        _norm_2[self.p_x == 0] = 1

        self.eta = np.moveaxis(np.asarray([[self.p_x / _norm_1, self.p_x / _norm_2], [(self.norm_p - self.p_t_mu) / _norm_1, (-self.norm_p - self.p_t_mu) / _norm_2]]), -1, 0)
        self.eta[self.p_x == 0, 0, 0] = 1
        # primnt(self._eta_21[self.p_x==0]) [Review this line in the future]
        self.eta[self.p_x == 0, 0, 1] = 0
        self.eta[self.p_x == 0, 1, 0] = 0
        self.eta[self.p_x == 0, 1, 1] = 1

    def _initialise_members(self, num_points, L, mu, m):
        self.mu = mu
        self.m = m
        self.num_points = np.asarray(num_points)
        self.L = np.asarray(L)
        self.a = self.L / self.num_points
        self.dof_spinor = 2
        self.total_num_lattice_points = self.dof_spinor * self.num_points.prod()

    def _compute_grids(self):
        self.x = np.array(np.meshgrid(*(np.linspace(0, L, n, endpoint=False) for L, n in zip(self.L, self.num_points))))
        self.p = I2PI * np.array(np.meshgrid(*(np.fft.fftfreq(n, a) for n, a in zip(self.num_points, self.a))))

    @property
    def p_t(self):
        return self.p[0].transpose().reshape(-1)

    @property
    def p_x(self):
        return self.p[1].transpose().reshape(-1)

    @property
    def L_t(self):
        return self.L[0]

    @property
    def L_x(self):
        return self.L[1]

    @property
    def n_t(self):
        return self.num_points[0]

    @property
    def n_x(self):
        return self.num_points[1]

    @property
    def a_t(self):
        return self.a[0]

    @property
    def a_x(self):
        return self.a[1]

    @property
    def dimension(self):
        """
        The dimension of the spectrum.
        """
        return 2

    @property
    def eigenvalues(self):
        """
        Private function to return the list of eigenvalues (the diagonal of the eigenvalue matrix)
        of the 2D Free Fermion operator

        Args:
            None

        Returns:
            numpy.ndarray: 1D array of eigenvalues
        """
        return (np.kron(self.norm_p, [1, -1])) + self.m

    def eigenfunction(self, index):
        """
        Function to return the eigenfunction of the 2D free fermions operator.

        Args:
            index: array of indices of the eigenfunctions to be returned

        Returns:
            lambda function (t,x) of the eigenfunction at the specified index

        Note: t,x passed to the returned lambda function must be scalars or 1D arrays, e.g., flattened out meshgrid.
            x, t = np.meshgrid(array_x, array_t)
             t = t.flatten()
             x = x.flatten()
        """

        index = np.atleast_1d(index)

        if (index >= self.total_num_lattice_points).any() or (index < -self.total_num_lattice_points).any():
            raise ValueError(f"Index {index} out of bounds.")

        sign = 1 - 2 * (index % 2)
        p_t = self.p_t[index // 2]  # Since two eigenvalues exist due to spinor structure
        p_x = self.p_x[index // 2]
        p_t_mu = self.p_t_mu[index // 2]
        norm_p = self.norm_p[index // 2]

        normalization = np.sqrt(2 * norm_p * (norm_p - (sign * p_t_mu)))
        mask = p_x == 0
        normalization[mask] = 1
        normalization = normalization[:, np.newaxis]

        # Normalized eigenvector for the ((p_t, p_x), (p_x, -p_t)) matrix
        eta = np.array([p_x, sign * norm_p - p_t_mu]).transpose() / normalization
        eta[mask] = np.eye(2)[index[mask] % 2]

        return lambda t, x: self._return_eigenfunction(t, x, len(index), eta, p_t, p_x)

    def _return_eigenfunction(self, t, x, num_eigenfunction, eta, p_t, p_x):
        """
        Return function when eigenfucntion method is called.
        """
        # This part {np.kron(p_t, t)} can be done more efficiently since some values of p_x, p_t repeat
        exp_component = np.exp(np.kron(p_t, t) + np.kron(p_x, x)).reshape(num_eigenfunction, -1) / np.sqrt(self.L_t * self.L_x)

        # Initialize the return array of length equal to the number of eigenfunctions indices to be returned
        ret = np.zeros((num_eigenfunction, self.total_num_lattice_points), dtype=np.complex128)

        # Kronecker product between each spinor array and corresponding exponential part
        for i in range(num_eigenfunction):
            ret[i] = np.kron(exp_component[i], eta[i])

        return ret

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
        """
        input_vector = np.asarray(input_vector)

        if input_basis == output_basis in ["real", "spectral"]:
            return input_vector

        elif input_basis == "real" and output_basis == "spectral":
            # Split the input vector into f(even index elements) and g(odd index elements)
            input_in_momentum_space = np.fft.fft2(input_vector.reshape(-1, *self.num_points, self.dof_spinor), axes=-2 - np.arange(self.dimension), norm="ortho") * np.sqrt(self.a_t * self.a_x)
            return np.einsum("ikj,lik->lij", self.eta, input_in_momentum_space.reshape(-1, np.prod(self.num_points), self.dof_spinor)).reshape(*input_vector.shape)

        elif input_basis == "spectral" and output_basis == "real":
            # Block diagonal multiplication of eigenvector matrix
            input_in_uniform_spinor_basis = np.einsum("ijk,lik->lij", self.eta, input_vector.reshape(-1, np.prod(self.num_points), self.dof_spinor))
            return (np.fft.ifft2(input_in_uniform_spinor_basis.reshape(-1, *self.num_points, self.dof_spinor), axes=-2 - np.arange(self.dimension), norm="ortho") / np.sqrt(self.a_t * self.a_x)).reshape(*input_vector.shape)

        else:
            raise ValueError(f"Unsupported space transformation from {input_basis} to {output_basis}.")

    def scalar_product(self, lhs, rhs, input_basis="real"):
        """
        Function to compute the scalar product of two vectors in the specified basis.

        Args:
            lhs: left hand side vector
            rhs: right hand side vector
            input_basis: basis of the input vectors (real/spectral)
        """
        if input_basis == "real":
            return rhs @ lhs.transpose().conjugate() * self.a_t * self.a_x

        elif input_basis == "spectral":
            return rhs @ lhs.transpose().conjugate()

        else:
            raise ValueError(f"Unsupported input space - {input_basis}.")

    def lattice(self, output_basis="real"):
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
            t, x = np.meshgrid(np.linspace(0, self.L_t, self.n_t, endpoint=False), np.linspace(0, self.L_x, self.n_x, endpoint=False), indexing="ij")
            return t.flatten(), x.flatten()

        elif output_basis == "spectral":
            return (self.p_t, self.p_x)
        else:
            raise ValueError(f"Unsupported output space - {output_basis}.")
