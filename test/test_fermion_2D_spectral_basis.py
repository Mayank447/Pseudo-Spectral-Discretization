#!/usr/bin/env python3

from pseudospectral import DiracOperator
import numpy as np
import pytest

## Some Fixtures like spectrum_fermion2D, arbitrary_index_single_eigenfunction, arbitrary_single_coefficient, arbitrary_index_multiple_eigenfunctions
## are defined in the conftest.py file.and imported in all the test files automatically.

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
def test_application_to_a_single_eigenfunction(spectrum_fermion2D, arbitrary_index_single_eigenfunction_fermion2D, arbitrary_single_coefficient):
    """
    Python test function to test the application of the Dirac operator to a single eigenfunction in real space.
    """
    operator = DiracOperator(spectrum_fermion2D)
    sign = 1
    index_t, index_x, sign = arbitrary_index_single_eigenfunction_fermion2D
    index = 2 * (spectrum_fermion2D.n_x * index_t + index_x) + int(0.5*(1-sign))

    eigenfunction = arbitrary_single_coefficient * np.eye(spectrum_fermion2D.num_lattice_points)[index, :]
    expected = eigenfunction * spectrum_fermion2D.eigenvalues[index]

    result = operator.apply_to(eigenfunction, input_basis="spectral", output_basis="spectral")
    assert np.isclose(result, expected).all()


def test_lattice_spectral_basis(spectrum_fermion2D):
    """
    Python test function to test the lattice of the Dirac operator in spectral space.
    """
    result = spectrum_fermion2D.lattice(output_basis="spectral")
    expected = (np.kron(spectrum_fermion2D.sqrt, [1, -1])) + spectrum_fermion2D.m
    assert np.equal(result, expected).all()