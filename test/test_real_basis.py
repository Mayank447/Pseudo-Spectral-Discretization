#!/usr/bin/env python3

from pseudospectral import DiracOperator
import numpy as np


num_single_eigenfunction_testrun = 10
num_eigenfunctions_superposition_testrun = 10

# Some Fixtures like
#   Spectrum,
#   arbitrary_index_single_eigenfunction,
#   arbitrary_single_coefficient,
#   arbitrary_index_multiple_eigenfunctions
# are defined in the conftest.py file.and imported in all the test files automatically.


# HELPER_FUNCTIONS
def arbitrary_multiple_coefficients(length=1):
    """
    Python function to initialize a numpy array of the given length
    with arbitrary coefficients sampled from a normal distribution for the tests.
    """
    return np.random.randn(length)


# TEST FUNCTION
def test_application_to_a_single_eigenfunction(
    spectrum,
    arbitrary_index_single_eigenfunction,
    arbitrary_single_coefficient,
    sample_points,
):
    """
    Python test function to test the application of the Dirac operator to a single
    eigenfunction in real space.
    """
    operator = DiracOperator(spectrum)
    eigenfunction = arbitrary_single_coefficient * spectrum.eigenfunction(
        arbitrary_index_single_eigenfunction
    )(*sample_points).reshape(-1)

    result = operator.apply_to(eigenfunction, input_basis="real", output_basis="real")

    expected = (
        eigenfunction
        * operator.spectrum.eigenvalues[arbitrary_index_single_eigenfunction]
    )
    assert np.allclose(result, expected)


def test_application_to_superposition_of_eigenfunctions(
    spectrum, arbitrary_index_multiple_eigenfunctions, sample_points
):
    """
    Python test function to test the application of the Dirac operator to a
    superposition of two eigenfunctions.
    """
    operator = DiracOperator(spectrum)
    arbitrary_coefficients = arbitrary_multiple_coefficients(
        len(arbitrary_index_multiple_eigenfunctions)
    )

    superposition = arbitrary_coefficients[:, np.newaxis] * spectrum.eigenfunction(
        arbitrary_index_multiple_eigenfunctions
    )(*sample_points).reshape(*arbitrary_index_multiple_eigenfunctions.shape, -1)

    expected = np.sum(
        spectrum.eigenvalues[arbitrary_index_multiple_eigenfunctions][:, np.newaxis]
        * superposition,
        axis=0,
    )

    result = operator.apply_to(
        np.sum(superposition, axis=0), input_basis="real", output_basis="real"
    )
    assert np.allclose(result, expected)
