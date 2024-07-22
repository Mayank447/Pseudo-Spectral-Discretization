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



############################################ TEST FUNCTION ############################################
def test_application_to_a_single_eigenfunction(spectrum, arbitrary_index_single_eigenfunction):
    """
    Python test function to test the application of the Dirac operator to a single eigenfunction in real space.
    """
    operator = DiracOperator(spectrum)
    
    # Make sample points a fixture later
    sample_points = np.linspace(0, spectrum.L, spectrum.num_lattice_points, endpoint=False)
    eigenfunction = operator.spectrum.eigenfunction(arbitrary_index_single_eigenfunction)(sample_points)
    result = operator.apply_to(eigenfunction, input_space="real", output_space="real")
    
    expected = eigenfunction * operator.spectrum.eigenvalues[arbitrary_index_single_eigenfunction]
    assert np.isclose(result, expected).all()



def test_application_to_superposition_of_two_eigenfunctions(spectrum, arbitrary_index_two_eigenfunctions):
    """
    Python test function to test the application of the Dirac operator to a superposition of two eigenfunctions.
    """
    operator = DiracOperator(spectrum)
    
    sample_points = np.linspace(0, spectrum.L, spectrum.num_lattice_points, endpoint=False)
    eigenfunction_1 = spectrum.eigenfunction(arbitrary_index_two_eigenfunctions[0])(sample_points)
    eigenfunction_2 = spectrum.eigenfunction(arbitrary_index_two_eigenfunctions[1])(sample_points)
    expected = np.column_stack((eigenfunction_1, eigenfunction_2)) @ spectrum.eigenvalues[arbitrary_index_two_eigenfunctions]

    result = operator.apply_to(eigenfunction_1 + eigenfunction_2, input_space="real", output_space="real")
    assert np.isclose(result, expected).all()



# def test_lattice():
#     """
#     Python test function to test the lattice method of the Dirac Operator class in the real space.
#     """
#     operator = DiracOperator(Derivative1D(10))
    
#     # Check exact lattice values
#     time_lattice = operator.lattice(output_space="real")
#     spectral_lattice = operator.lattice(output_space="spectral")
    
#     expected_time_lattice = np.linspace(0, operator.beta, operator.n_time, endpoint=False)
#     expected_spectral_lattice = 2 * np.pi * (np.arange(operator.n_time) + 0.5) / operator.beta
    
#     assert np.allclose(time_lattice, expected_time_lattice)
#     assert np.allclose(spectral_lattice, expected_spectral_lattice)
    
#     # Potentially more properties to check
#     assert len(time_lattice) == operator.n_time
#     assert len(spectral_lattice) == operator.n_time