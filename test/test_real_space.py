#!/usr/bin/env python3

from pseudospectral import DiracOperator
import numpy as np
import pytest

def test_application_to_a_single_eigenfunction():
    """
    Python test function to test the application of the Dirac operator to a single eigenfunction in real space.
    """
    operator = DiracOperator(n_time=4, nu=4, n_landau=4)
    arbitrary_index = [2, 3, 1]
    
    # Create a lattice in real space
    lattice = operator.lattice(output_space="real space")
    function_values = operator.eigenfunction(
        arbitrary_index, output_space="real space", coordinates=lattice
    )
    
    result = operator.apply_to(
        function_values, input_space="real space", output_space="real space"
    )
    expected = operator.eigenvalue(arbitrary_index) * function_values
    assert np.allclose(result, expected)



def test_application_to_another_single_eigenfunction():
    """
    Python test function to test the application of the Dirac operator to another single eigenfunction in real space.
    """
    operator = DiracOperator(n_time=4, nu=4, n_landau=4)
    arbitrary_index = [1, 2, 0]
    
    lattice = operator.lattice(output_space="real space")
    function_values = operator.eigenfunction(
        arbitrary_index, output_space="real space", coordinates=lattice
    )
    
    result = operator.apply_to(
        function_values, input_space="real space", output_space="real space"
    )
    expected = operator.eigenvalue(arbitrary_index) * function_values
    assert np.allclose(result, expected)



def test_application_to_superposition_of_two_eigenfunctions():
    """
    Python test function to test the application of the Dirac operator to a superposition of two eigenfunctions.
    """
    operator = DiracOperator(n_time=4, nu=4, n_landau=4)
    index1 = [0, 1, 2]
    index2 = [2, 2, 3]
    
    lattice = operator.lattice(output_space="real space")
    function_values1 = operator.eigenfunction(
        index1, output_space="real space", coordinates=lattice
    )
    function_values2 = operator.eigenfunction(
        index2, output_space="real space", coordinates=lattice
    )
    
    function_values = function_values1 + function_values2
    
    result = operator.apply_to(
        function_values, input_space="real space", output_space="real space"
    )
    expected = operator.eigenvalue(index1) * function_values1 + operator.eigenvalue(index2) * function_values2
    assert np.allclose(result, expected)



def test_eigenfunction():
    """
    Python test function to test the eigenfunction method of the Dirac Operator class in the real space.
    """
    operator = DiracOperator(n_time=4, nu=4, n_landau=4)
    index1 = [0, 0, 0]
    index2 = [1, 0, 0]
    
    lattice = operator.lattice(output_space="real space")
    function_values1 = operator.eigenfunction(
        index1, output_space="real space", coordinates=lattice
    )
    function_values2 = operator.eigenfunction(
        index2, output_space="real space", coordinates=lattice
    )
    
    # Check exact function values for known eigenfunctions (example values)
    assert np.allclose(function_values1, np.exp(1j * np.dot(lattice, index1)))
    assert np.allclose(function_values2, np.exp(1j * np.dot(lattice, index2)))
    
    # Check orthogonality
    inner_product = np.vdot(function_values1, function_values2)
    assert np.isclose(inner_product, 0)



def test_lattice():
    """
    Python test function to test the lattice method of the Dirac Operator class in the real space.
    """
    operator = DiracOperator(n_time=4, nu=4, n_landau=4)
    
    # Check exact lattice values
    time_lattice = operator.lattice(output_space="time")
    frequency_lattice = operator.lattice(output_space="frequency")
    
    expected_time_lattice = np.linspace(0, operator.beta, operator.n_time, endpoint=False)
    expected_frequency_lattice = 2 * np.pi * (np.arange(operator.n_time) + 0.5) / operator.beta
    
    assert np.allclose(time_lattice, expected_time_lattice)
    assert np.allclose(frequency_lattice, expected_frequency_lattice)
    
    # Potentially more properties to check
    assert len(time_lattice) == operator.n_time
    assert len(frequency_lattice) == operator.n_time


# A simpler operator: Free Dirac operator
@pytest.mark.parametrize("index, expected_eigenvalue", [
    ([0, 0, 0], np.pi / 4),
    ([1, 0, 0], np.pi / 4),
    ([2, 0, 0], np.pi / 4),
    ([3, 0, 0], np.pi / 4),
])
def test_free_dirac_operator_eigenvalues(index, expected_eigenvalue):
    operator = DiracOperator(n_time=4, nu=4, n_landau=4)
    assert np.isclose(operator.eigenvalue(index), expected_eigenvalue)


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
