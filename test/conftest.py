#!/usr/bin/env python3

from pseudospectral import Derivative1D, FreeFermion2D, naive_implementation_of
import pytest
import numpy as np

SPECTRA = [
    {"type": Derivative1D, "config": {"total_num_lattice_points": 3}},
    {"type": Derivative1D, "config": {"total_num_lattice_points": 101}},
    {"type": Derivative1D, "config": {"total_num_lattice_points": 3, "L": 3}},
    {"type": Derivative1D, "config": {"total_num_lattice_points": 101, "L": 42}},
    {
        "type": FreeFermion2D,
        "config": {"num_points": [3, 3], "L": [1, 1], "mu": 0, "m": 0},
    },
    {
        "type": FreeFermion2D,
        "config": {"num_points": [3, 70], "L": [2, 7], "mu": 0, "m": 0},
    },
    {
        "type": FreeFermion2D,
        "config": {
            "num_points": [3, 3],
            "L": [1, 1],
            "mu": 0,
            "m": 0,
            "theta": [0.5, 0.0],
        },
    },
    {
        "type": FreeFermion2D,
        "config": {
            "num_points": [3, 70],
            "L": [2, 7],
            "mu": 0,
            "m": 0,
            "theta": [0.3, 0.8],
        },
    },
]

SPECTRA += [spec | {"type": naive_implementation_of(spec["type"])} for spec in SPECTRA]


# Pytest settings
def pytest_addoption(parser):
    parser.addoption(
        "--repeat",
        default=1,
        type=int,
        metavar="repeat",
        help="Run each test the specified number of times",
    )


def pytest_collection_modifyitems(session, config, items):
    count = config.option.repeat
    items[:] = items * count  # add each test multiple times


# Common Fixtures for PyTests
@pytest.fixture(params=SPECTRA)
def spectrum(request):
    return request.param["type"](**request.param["config"])


@pytest.fixture
def arbitrary_index_single_eigenfunction(spectrum):
    """
    Python fixture to initialize the arbitrary index for the single eigenfunction test.
    """
    return np.random.randint(spectrum.total_num_of_dof, size=1)


@pytest.fixture
def arbitrary_index_multiple_eigenfunctions(
    spectrum,
):
    """
    Python fixture to initialize the arbitrary index for the two eigenfunctions test.
    """
    return np.random.choice(
        spectrum.total_num_of_dof,
        size=1 + np.random.randint(spectrum.total_num_of_dof - 1),
        replace=False,
    )


@pytest.fixture
def arbitrary_single_coefficient():
    """
    Python fixture to initialize the single coefficient for the tests.
    """
    return np.random.randn()


@pytest.fixture
def sample_points(spectrum, output_basis="real"):
    """
    Python fixture to initialize the sample points for the superposition test.
    """
    return spectrum.lattice(output_basis=output_basis)
