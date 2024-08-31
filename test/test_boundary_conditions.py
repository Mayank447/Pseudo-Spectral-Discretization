import numpy as np


def test_eigenfunctions_obey_boundary_conditions(spectrum):
    eigenfunctions = spectrum.eigenfunction(np.arange(spectrum.total_num_of_dof))
    lattice = np.array(spectrum.lattice())
    for dim in range(spectrum.spacetime_dimension):
        shifted_lattice = np.array(lattice) + spectrum.L.reshape(-1)[dim] * np.eye(
            spectrum.spacetime_dimension
        )[dim].reshape(spectrum.spacetime_dimension, 1)
        assert np.allclose(
            np.exp(2.0j * np.pi * spectrum.theta).reshape(-1)[dim]
            * eigenfunctions(*lattice).reshape(spectrum.total_num_of_dof, -1),
            eigenfunctions(*shifted_lattice).reshape(spectrum.total_num_of_dof, -1),
        )


def test_inverse_transform_results_in_eigenfunction(spectrum):
    spectral_representation = np.eye(spectrum.total_num_of_dof)
    real_space_representation = spectrum.transform(
        spectral_representation, "spectral", "real"
    )
    eigenfunction = spectrum.eigenfunction(np.arange(spectrum.total_num_of_dof))
    assert np.allclose(
        eigenfunction(*spectrum.lattice()).reshape(spectrum.total_num_of_dof, -1),
        real_space_representation,
    )


def test_transform_with_correct_boundary_conditions(spectrum):
    eigenfunctions = spectrum.eigenfunction(np.arange(spectrum.total_num_of_dof))
    assert np.allclose(
        spectrum.transform(
            eigenfunctions(*spectrum.lattice()).reshape(spectrum.total_num_of_dof, -1),
            "real",
            "spectral",
        ),
        np.eye(spectrum.total_num_of_dof),
    )
