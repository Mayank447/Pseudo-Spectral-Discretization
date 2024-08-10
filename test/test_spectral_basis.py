#!/usr/bin/env python3

from pseudospectral import DiracOperator
import numpy as np
import pytest
import scipy

num_single_eigenfunction_testrun = 10
num_eigenfunctions_superposition_testrun = 10

########################################## HELPER_FUNCTIONS ##########################################
def arbitrary_multiple_coefficients(length=1):
    """
    Python function to initialize a numpy array of the given length 
    with arbitrary coefficients sampled from a normal distribution for the tests.
    """
    return np.random.randn(length)


########################################### TEST FUNCTION ############################################
@pytest.mark.parametrize("arbitrary_index_single_eigenfunction", range(num_single_eigenfunction_testrun), indirect=True)
def test_application_to_a_single_eigenfunction(spectrum, arbitrary_index_single_eigenfunction, arbitrary_single_coefficient):
    """
    Python test function to test the application of the Dirac operator to a single eigenfunction in real space.
    """
    operator = DiracOperator(spectrum)

    eigenfunction = arbitrary_single_coefficient * np.eye(spectrum.num_lattice_points)[arbitrary_index_single_eigenfunction, :]
    expected = eigenfunction * spectrum.eigenvalues[arbitrary_index_single_eigenfunction]

    result = operator.apply_to(eigenfunction, input_basis="spectral", output_basis="spectral")
    assert np.isclose(result, expected).all()


@pytest.mark.parametrize("arbitrary_index_multiple_eigenfunctions", range(num_eigenfunctions_superposition_testrun), indirect=True)
def test_application_to_superposition_of_multiple_eigenfunctions(spectrum, arbitrary_index_multiple_eigenfunctions):
    """
    Python test function to test the application of the Dirac operator to a superposition of two eigenfunctions.
    """
    operator = DiracOperator(spectrum)
    arbitrary_coefficients = arbitrary_multiple_coefficients(len(arbitrary_index_multiple_eigenfunctions))

    eigenfunctions = np.eye(spectrum.num_lattice_points)[arbitrary_index_multiple_eigenfunctions].transpose() * arbitrary_coefficients
    expected = eigenfunctions @ spectrum.eigenvalues[arbitrary_index_multiple_eigenfunctions]

    result = operator.apply_to(
        np.sum(eigenfunctions, axis=1),
        input_basis="spectral",
        output_basis="spectral",
    )
    assert np.isclose(result, expected).all()


def test_lattice_spectral_basis(spectrum):
    """
    Python test function to test the lattice of the Dirac operator in spectral space.
    """
    result = spectrum.lattice(output_basis="spectral")
    expected = 2j * np.pi * (scipy.fft.fftfreq(spectrum.num_lattice_points, d=spectrum.a) - spectrum.theta / spectrum.L)
    assert np.equal(result, expected).all()
