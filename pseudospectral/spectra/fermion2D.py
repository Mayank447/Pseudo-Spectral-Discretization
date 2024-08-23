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
        self.x = np.array(np.meshgrid(*(np.linspace(0, L, n, endpoint=False) for L, n in zip(self.L, self.num_points)), indexing="ij"))
        self.p = I2PI * np.array(np.meshgrid(*(np.fft.fftfreq(n, a) for n, a in zip(self.num_points, self.a)), indexing="ij"))

    @property
    def p_t(self):
        return self.p[0].reshape(-1)

    @property
    def p_x(self):
        return self.p[1].reshape(-1)

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

        index = np.asarray(index)

        if (index >= self.total_num_lattice_points).any() or (index < -self.total_num_lattice_points).any():
            raise ValueError(f"Index {index} out of bounds.")

        # Since two eigenvalues exist due to spinor structure
        spacetime_index = index // self.dof_spinor

        return lambda t, x: np.einsum(
            "j...,jk->j...k",
            np.exp(np.einsum("ij,i...->j...", self.p.reshape(self.dimension, -1)[:, spacetime_index], np.asarray([t, x]))).reshape(*index.shape, *x.shape) / np.sqrt(np.prod(self.L)),
            self.eta.reshape(-1, self.dof_spinor, self.dof_spinor)[spacetime_index, :, index % self.dof_spinor],
        )

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
            input_in_momentum_space = np.fft.fft2(input_vector.reshape(-1, *self.num_points, self.dof_spinor), axes=-2 - np.arange(self.dimension), norm="ortho") * np.sqrt(np.prod(self.a))
            return np.einsum("ikj,lik->lij", self.eta, input_in_momentum_space.reshape(-1, np.prod(self.num_points), self.dof_spinor)).reshape(*input_vector.shape)

        elif input_basis == "spectral" and output_basis == "real":
            # Block diagonal multiplication of eigenvector matrix
            input_in_uniform_spinor_basis = np.einsum("ijk,lik->lij", self.eta, input_vector.reshape(-1, np.prod(self.num_points), self.dof_spinor))
            return (np.fft.ifft2(input_in_uniform_spinor_basis.reshape(-1, *self.num_points, self.dof_spinor), axes=-2 - np.arange(self.dimension), norm="ortho") / np.sqrt(np.prod(self.a))).reshape(*input_vector.shape)

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
            return rhs @ lhs.transpose().conjugate() * np.prod(self.a)

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
            return self.x.reshape(self.dimension, -1)

        elif output_basis == "spectral":
            return self.p.reshape(self.dimension, -1)

        else:
            raise ValueError(f"Unsupported output space - {output_basis}.")
