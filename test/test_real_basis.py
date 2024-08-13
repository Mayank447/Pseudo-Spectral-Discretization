#!/usr/bin/env python3

from pseudospectral import DiracOperator
import numpy as np
import pytest


num_single_eigenfunction_testrun = 10
num_eigenfunctions_superposition_testrun = 10

## Some Fixtures like Spectrum, arbitrary_index_single_eigenfunction, arbitrary_single_coefficient, arbitrary_index_multiple_eigenfunctions
## are defined in the conftest.py file.and imported in all the test files automatically.


########################################## HELPER_FUNCTIONS ##########################################
def arbitrary_multiple_coefficients(length=1):
    """
    Python function to initialize a numpy array of the given length 
    with arbitrary coefficients sampled from a normal distribution for the tests.
    """
    return np.random.randn(length)


############################################ TEST FUNCTION ############################################
@pytest.mark.parametrize("arbitrary_index_single_eigenfunction", range(num_single_eigenfunction_testrun), indirect=True)
def test_application_to_a_single_eigenfunction(spectrum, arbitrary_index_single_eigenfunction, arbitrary_single_coefficient):
    """
    Python test function to test the application of the Dirac operator to a single eigenfunction in real space.
    """
    operator = DiracOperator(spectrum)
    sample_points = np.linspace(0, spectrum.L, spectrum.num_lattice_points, endpoint=False)
    eigenfunction = arbitrary_single_coefficient * spectrum.eigenfunction(arbitrary_index_single_eigenfunction)(sample_points)

    result = operator.apply_to(eigenfunction, input_basis="real", output_basis="real")

    expected = eigenfunction * operator.spectrum.eigenvalues[arbitrary_index_single_eigenfunction]
    assert np.isclose(result, expected).all()


@pytest.mark.parametrize("arbitrary_index_multiple_eigenfunctions", range(num_eigenfunctions_superposition_testrun), indirect=True)
def test_application_to_superposition_of_eigenfunctions(spectrum, arbitrary_index_multiple_eigenfunctions):
    """
    Python test function to test the application of the Dirac operator to a superposition of two eigenfunctions.
    """
    operator = DiracOperator(spectrum)
    arbitrary_coefficients = arbitrary_multiple_coefficients(len(arbitrary_index_multiple_eigenfunctions))

    sample_points = np.linspace(0, spectrum.L, spectrum.num_lattice_points, endpoint=False)
    superposition = (
        arbitrary_coefficients * 
        spectrum.eigenfunction(arbitrary_index_multiple_eigenfunctions)(sample_points.reshape(-1, 1))
    )
    expected = superposition @ spectrum.eigenvalues[arbitrary_index_multiple_eigenfunctions]

    result = operator.apply_to(np.sum(superposition, axis=1), input_basis="real", output_basis="real")
    assert np.isclose(result, expected).all()


def test_lattice_real_basis(spectrum):
    """
    Python test function to test the lattice method of the Spectrum class in the real space.
    """

    lattice = spectrum.lattice(output_basis="real")
    excepted = np.linspace(0, spectrum.L, spectrum.num_lattice_points, endpoint=False)
    assert np.equal(lattice, excepted).all()
