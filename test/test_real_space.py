#!/usr/bin/env python3

from pseudospectral import DiracOperator, Derivative1D
import numpy as np
import pytest

def test_application_to_a_single_eigenfunction(L=4, n=4):
    """
    Python test function to test the application of the Dirac operator to a single eigenfunction in real space.
    """
    arbitrary_index = 1 # Index of the eigenfunction to be tested
    operator = DiracOperator(Derivative1D(num_lattice_points=n, L=L))
    
    eigenfunction = operator.spectrum.eigenfunction(arbitrary_index)(np.arange(L))
    expected = eigenfunction * operator.spectrum.eigenvalues[arbitrary_index]
    result = operator.apply_to(eigenfunction, input_space="real", output_space="real")
    
    # print("Expected: ", expected, '\n')
    # print("Result: ", result, '\n')
    assert np.isclose(result, expected).all()



def test_application_to_superposition_of_two_eigenfunctions(L=5, n=5):
    """
    Python test function to test the application of the Dirac operator to a superposition of two eigenfunctions.
    """
    arbitrary_index = [1,2] # Index of the eigenfunction to be tested
    operator = DiracOperator(Derivative1D(num_lattice_points = n, L = L))
    
    eigenfunction_1 = operator.spectrum.eigenfunction(arbitrary_index[0])(np.arange(L))
    eigenfunction_2 = operator.spectrum.eigenfunction(arbitrary_index[1])(np.arange(L))
    expected = eigenfunction_1 * operator.spectrum.eigenvalues[arbitrary_index[0]] + eigenfunction_2 * operator.spectrum.eigenvalues[arbitrary_index[1]]
    
    result = operator.apply_to(eigenfunction_1 + eigenfunction_2, input_space="real", output_space="real")
    # print("Expected: ", expected, '\n')
    # print("Result: ", result, '\n')
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


# A simpler operator: Free Dirac operator
# @pytest.mark.parametrize("index, expected_eigenvalue", [
#     ([0, 0, 0], np.pi / 4),
#     ([1, 0, 0], np.pi / 4),
#     ([2, 0, 0], np.pi / 4),
#     ([3, 0, 0], np.pi / 4),
# ])
# def test_free_dirac_operator_eigenvalues(index, expected_eigenvalue):
#     operator = DiracOperator(n_time=4, nu=4, n_landau=4)
#     assert np.isclose(operator.eigenvalue(index), expected_eigenvalue)


# A simpler operator: Free Dirac operator == plain first derivative
#
# D = gamma^mu partial_mu =
#     (
#           partial_0                   partial_1 + i partial_2
#           partial_1 - i partial_2        -partial_0
#     )
#
# Eigenfunctions are productions of plane waves in each direction
# In a finite box with (anti-periodic boundary conditions), eigenvalues are linearly spaced with an offset depending on the boundary conditions.

# def main():
#     # test_application_to_a_single_eigenfunction(L=4, n=4)
#     test_application_to_superposition_of_two_eigenfunctions(L=5, n=5)

# if __name__ == "__main__":
#     main()