from pseudospectral.spectra.derivative1D import Derivative1D
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
]


@pytest.fixture(params=SPECTRA)
def spectrum(request):
    return request.param["type"](**request.param["config"])


def test_eigenfunctions_obey_boundary_conditions(spectrum):
    eigenfunctions = spectrum.eigenfunction(
        np.arange(spectrum.total_num_lattice_points)
    )
    assert np.allclose(
        np.exp(2.0j * np.pi * spectrum.theta) * eigenfunctions(*spectrum.lattice()),
        eigenfunctions(spectrum.lattice()[0] + spectrum.L),
    )


def test_inverse_transform_results_in_eigenfunction(spectrum):
    spectral_representation = np.eye(spectrum.total_num_lattice_points)
    real_space_representation = spectrum.transform(
        spectral_representation, "spectral", "real"
    )
    eigenfunction = spectrum.eigenfunction(np.arange(spectrum.total_num_lattice_points))
    assert np.allclose(eigenfunction(*spectrum.lattice()), real_space_representation)


def test_transform_with_correct_boundary_conditions(spectrum):
    eigenfunctions = spectrum.eigenfunction(
        np.arange(spectrum.total_num_lattice_points)
    )
    assert np.allclose(
        spectrum.transform(eigenfunctions(*spectrum.lattice()), "real", "spectral"),
        np.eye(spectrum.total_num_lattice_points),
    )
