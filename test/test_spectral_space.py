#!/usr/bin/env python3

from pseudospectral import DiracOperator, Derivative1D
import numpy as np
import pytest


def test_application_to_a_single_eigenfunction(L=4, n=5):
    """
    Python test function to test the application of the Dirac operator to a single eigenfunction in real space.
    """
    arbitrary_index = 1 # Index of the eigenfunction to be tested
    operator = DiracOperator(Derivative1D(num_lattice_points=n, L=L))
    
    eigenfunction = np.zeros(L)
    eigenfunction[arbitrary_index] = 1
    expected = eigenfunction * operator.spectrum.eigenvalues[arbitrary_index]
    print("Eigenfunction: ", eigenfunction, '\n')
    print("Eigenvalue: ", operator.spectrum.eigenvalues[arbitrary_index], '\n')
    print("Expected: ", expected, '\n')
    
    result = operator.apply_to(eigenfunction, input_space="spectral", output_space="spectral")
    assert np.allclose(np.sum(np.absolute(result -expected)), 0.05)



def test_application_to_superposition_of_two_eigenfunctions(L=1, n=5):
    """
    Python test function to test the application of the Dirac operator to a superposition of two eigenfunctions.
    """
    arbitrary_index = [1,2] # Index of the eigenfunction to be tested
    operator = DiracOperator(Derivative1D(num_lattice_points=n, L=L))
    
    eigenfunction_1, eigenfunction_2 = np.zeros(L), np.zeros(L)
    eigenfunction_1[arbitrary_index[0]], eigenfunction_2[arbitrary_index[1]] = 1, 1
    expected = eigenfunction_1 * operator.spectrum.eigenvalues[arbitrary_index[0]] + eigenfunction_2 * operator.spectrum.eigenvalues[arbitrary_index[1]]
    
    result = operator.apply_to(eigenfunction_1 + eigenfunction_2, input_space="spectral", output_space="spectral")
    assert np.allclose(np.sum(np.absolute(result -expected)), 0.05)



# def test_reports_correct_eigenvalue():
#     n_time = 4
#     operator = DiracOperator(n_time=n_time, nu=4, n_landau=4)
#     arbitrary_index = (0, 0, 0)
#     expected_eigenvalue = np.pi / n_time
#     assert np.isclose(operator.eigenvalue(arbitrary_index), expected_eigenvalue)


# Parameterized test for a table of eigenvalues
# @pytest.mark.parametrize("index, expected_eigenvalue", [
#     ([0, 0, 0], np.pi / 4),
#     ([1, 0, 0], np.pi / 4),
#     ([2, 0, 0], np.pi / 4),
#     ([3, 0, 0], np.pi / 4),
# ])
# def test_eigenvalues(index, expected_eigenvalue):
#     operator = DiracOperator(n_time=4, nu=4, n_landau=4)
#     assert np.isclose(operator.eigenvalue(index), expected_eigenvalue)


# write a table of eigenvalues and test all of them

# use approximative equality from pytests


# you can deduplicate test code via @pytest.parametrize() (or something like that...)
