#!/usr/bin/env python3

from pseudospectral import DiracOperator, Derivative1D
import numpy as np
import pytest

########################################## FIXTURES ##########################################
@pytest.fixture
def spectrum(L=4, n=4):
    """
    Python fixture to initialize the spectrum for the tests.
    """
    return Derivative1D(L=L, num_lattice_points=n)


@pytest.fixture
def arbitrary_index_single_eigenfunction(index=1):
    """
    Python fixture to initialize the arbitrary index for the single eigenfunction test.
    """
    return index


@pytest.fixture
def arbitrary_index_two_eigenfunctions():
    """
    Python fixture to initialize the arbitrary index for the two eigenfunctions test.
    """
    return np.array([1,2])


########################################### TEST FUNCTION ############################################
def test_application_to_a_single_eigenfunction(spectrum, arbitrary_index_single_eigenfunction):
    """
    Python test function to test the application of the Dirac operator to a single eigenfunction in real space.
    """
    operator = DiracOperator(spectrum)
    
    eigenfunction = np.eye(spectrum.num_lattice_points)[arbitrary_index_single_eigenfunction, :]
    expected = eigenfunction * spectrum.eigenvalues[arbitrary_index_single_eigenfunction]
    
    result = operator.apply_to(eigenfunction, input_space="spectral", output_space="spectral")
    assert np.isclose(result, expected).all()



def test_application_to_superposition_of_two_eigenfunctions(spectrum, arbitrary_index_two_eigenfunctions):
    """
    Python test function to test the application of the Dirac operator to a superposition of two eigenfunctions.
    """
    operator = DiracOperator(spectrum)
    
    eigenfunction_1, eigenfunction_2 = np.eye(spectrum.num_lattice_points)[arbitrary_index_two_eigenfunctions, :]
    expected = eigenfunction_1 * operator.spectrum.eigenvalues[arbitrary_index_two_eigenfunctions[0]] + eigenfunction_2 * operator.spectrum.eigenvalues[arbitrary_index_two_eigenfunctions[1]]

    result = operator.apply_to(eigenfunction_1 + eigenfunction_2, input_space="spectral", output_space="spectral")
    assert np.isclose(result, expected).all()


def test_lattice_spectral_space(spectrum):
    """
    Python test function to test the lattice of the Dirac operator in spectral space.
    """
    result = spectrum.lattice(output_space="spectral")
    expected = 2 * np.pi * np.arange(spectrum.num_lattice_points) / spectrum.L
    assert np.equal(result, expected).all()