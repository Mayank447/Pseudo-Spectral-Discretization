#!/usr/bin/env python3

from pseudospectral import DiracOperator
import numpy as np
import pytest


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


########################################## FIXTURES ##########################################
@pytest.fixture
def arbitrary_index_single_eigenfunction(index=1):
    """
    Python fixture to initialize the arbitrary index for the single eigenfunction test.
    """
    return np.array(index)


@pytest.fixture
def arbitrary_index_two_eigenfunctions():
    """
    Python fixture to initialize the arbitrary index for the two eigenfunctions test.
    """
    return np.array([(x, y) for x in range(4) for y in range(4)])


############################################ TEST FUNCTION ############################################
@pytest.mark.parametrize("arbitrary_index", np.arange(4))
def test_application_to_a_single_eigenfunction(arbitrary_index, spectrum):
    """
    Python test function to test the application of the Dirac operator to a single eigenfunction in real space.
    """
    operator = DiracOperator(spectrum)

    # Make sample points a fixture later
    sample_points = np.linspace(
        0, spectrum.L, spectrum.num_lattice_points, endpoint=False
    )
    eigenfunction = operator.spectrum.eigenfunction(arbitrary_index)(sample_points)

    result = operator.apply_to(eigenfunction, input_basis="real", output_basis="real")

    expected = eigenfunction * operator.spectrum.eigenvalues[arbitrary_index]
    assert np.isclose(result, expected).all()


@pytest.mark.parametrize(
    "arbitrary_index", np.array([[x, y] for x in range(4) for y in range(4)])
)
def test_application_to_superposition_of_two_eigenfunctions(spectrum, arbitrary_index):
    """
    Python test function to test the application of the Dirac operator to a superposition of two eigenfunctions.
    """
    operator = DiracOperator(spectrum)

    sample_points = np.linspace(
        0, spectrum.L, spectrum.num_lattice_points, endpoint=False
    )
    eigenfunction_1 = spectrum.eigenfunction(arbitrary_index[0])(sample_points)
    eigenfunction_2 = spectrum.eigenfunction(arbitrary_index[1])(sample_points)
    expected = (
        np.column_stack((eigenfunction_1, eigenfunction_2))
        @ spectrum.eigenvalues[arbitrary_index]
    )

    result = operator.apply_to(
        eigenfunction_1 + eigenfunction_2, input_basis="real", output_basis="real"
    )
    assert np.isclose(result, expected).all()


def test_lattice_real_basis(spectrum):
    """
    Python test function to test the lattice method of the Spectrum class in the real space.
    """

    lattice = spectrum.lattice(output_basis="real")
    excepted = np.linspace(0, spectrum.L, spectrum.num_lattice_points, endpoint=False)
    assert np.equal(lattice, excepted).all()
