from pseudospectral import Derivative1D, FreeFermion2D
import pytest
import numpy as np

SPECTRA = [
    {"type": Derivative1D, "config": {"total_num_lattice_points": 3}},
    {"type": Derivative1D, "config": {"total_num_lattice_points": 101}},
    {"type": Derivative1D, "config": {"total_num_lattice_points": 3, "L": 3}},
    {"type": Derivative1D, "config": {"total_num_lattice_points": 101, "L": 42}},
    {"type": Derivative1D, "config": {"total_num_lattice_points": 3, "theta": 0.1}},
    {"type": Derivative1D, "config": {"total_num_lattice_points": 101, "theta": 0.3}},
    {
        "type": Derivative1D,
        "config": {"total_num_lattice_points": 3, "L": 3, "theta": 0.5},
    },
    {
        "type": Derivative1D,
        "config": {"total_num_lattice_points": 101, "L": 42, "theta": 0.8},
    },
    {
        "type": FreeFermion2D,
        "config": {
            "num_points": [3, 3],
            "L": [1, 1],
            "theta": [0.0, 0.0],
            "mu": 0,
            "m": 0,
        },
    },
]


@pytest.fixture(params=SPECTRA)
def spectrum(request):
    return request.param["type"](**request.param["config"])


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
