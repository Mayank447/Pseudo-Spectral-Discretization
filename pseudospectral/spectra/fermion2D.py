#!/usr/bin/env python3
import numpy as np

I2PI = 1j * 2 * np.pi
PAULI_MATRICES = np.array([[[1, 0], [0, 1]], [[1, 0], [0, -1]], [[0, 1], [1, 0]], [[0, 1.0j], [-1.0j, 0]]])
IDENTITY = PAULI_MATRICES[0]


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

    def __init__(self, num_points, L=None, mu=0, m=0):
        self._initialise_members(num_points, L, mu, m)
        self._setup_spinor_structure()
        self._compute_grids()
        self._solve_spectral_problem_in_spinor_space()

    def _initialise_members(self, num_points, L, mu, m):
        self.mu = mu
        self.m = m
        self.num_points = np.asarray(num_points)
        self.L = np.asarray(L) if L is not None else self.num_points
        self.a = self.L / self.num_points
        self.spacetime_dimension = len(self.num_points)

    def _compute_grids(self):
        self.x = np.array(np.meshgrid(*(np.linspace(0, L, n, endpoint=False) for L, n in zip(self.L, self.num_points)), indexing="ij"))
        self.p = I2PI * np.array(np.meshgrid(*(np.fft.fftfreq(n, a) for n, a in zip(self.num_points, self.a)), indexing="ij"))

    def _setup_spinor_structure(self):
        if self.spacetime_dimension not in [2, 3]:
            raise NotImplementedError(f"We do not have gamma matrices for dimension {self.spacetime_dimension}.")
        self.gamma = PAULI_MATRICES[1 : self.spacetime_dimension + 1]
        self.dof_spinor = self.gamma.shape[-1]
        self.total_num_of_dof = self.dof_spinor * self.num_points.prod()

    def _solve_spectral_problem_in_spinor_space(self):
        mu_vectorfield = self.mu * np.eye(self.spacetime_dimension)[0].reshape(-1, *(np.ones(self.spacetime_dimension, dtype=int)))
        matrix_in_momentum_space = (
            np.sum(self.gamma[..., *(self.spacetime_dimension * [np.newaxis])] * (self.p - mu_vectorfield)[:, np.newaxis, np.newaxis, ...], axis=0) + self.m * IDENTITY[..., *(self.spacetime_dimension * [np.newaxis])]
        ).transpose(np.roll(np.arange(self.spacetime_dimension + 2), 2))
        self.eigenvalues, self.eta = np.linalg.eig(matrix_in_momentum_space)
        self.eta = self.eta.reshape(-1, self.dof_spinor, self.dof_spinor)
        self.eigenvalues = self.eigenvalues.reshape(-1)

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

        if (index >= self.total_num_of_dof).any() or (index < -self.total_num_of_dof).any():
            raise ValueError(f"Index {index} out of bounds.")

        # Since two eigenvalues exist due to spinor structure
        spacetime_index = index // self.dof_spinor

        return lambda t, x: np.einsum(
            "j...,jk->j...k",
            np.exp(np.einsum("ij,i...->j...", self.p.reshape(self.spacetime_dimension, -1)[:, spacetime_index], np.asarray([t, x]))).reshape(*index.shape, *x.shape) / np.sqrt(np.prod(self.L)),
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
        if (found := input_vector.shape[-1]) != (expected := self.total_num_of_dof):
            raise ValueError(f"Wrongly shaped input_vector: Expected {expected}, found {found} as size of last axis!")

        if input_basis == output_basis in ["real", "spectral"]:
            return input_vector

        elif input_basis == "real" and output_basis == "spectral":
            # Split the input vector into f(even index elements) and g(odd index elements)
            input_in_momentum_space = np.fft.fftn(input_vector.reshape(-1, *self.num_points, self.dof_spinor), axes=1 + np.arange(self.spacetime_dimension), norm="ortho") * np.sqrt(np.prod(self.a))
            return np.einsum("ikj,lik->lij", self.eta, input_in_momentum_space.reshape(-1, np.prod(self.num_points), self.dof_spinor)).reshape(*input_vector.shape)

        elif input_basis == "spectral" and output_basis == "real":
            # Block diagonal multiplication of eigenvector matrix
            input_in_uniform_spinor_basis = np.einsum("ijk,lik->lij", self.eta, input_vector.reshape(-1, np.prod(self.num_points), self.dof_spinor))
            return (np.fft.ifftn(input_in_uniform_spinor_basis.reshape(-1, *self.num_points, self.dof_spinor), axes=1 + np.arange(self.spacetime_dimension), norm="ortho") / np.sqrt(np.prod(self.a))).reshape(*input_vector.shape)

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
            return self.x.reshape(self.spacetime_dimension, -1)

        elif output_basis == "spectral":
            return self.p.reshape(self.spacetime_dimension, -1)

        else:
            raise ValueError(f"Unsupported output space - {output_basis}.")
