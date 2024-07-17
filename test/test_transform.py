from pseudospectral import DiracOperator
import numpy as np
import pytest


def test_transforms_from_spectral_to_real_space():
    """
    Python test function to test the transformation of spectral coefficients with a single component from spectral space to real space.
    """
    operator = DiracOperator(n_real=4, nu=4, n_landau=4)
    arbitrary_index = [2, 3, 1]
    
    # Create spectral coefficients with a single component
    spectral_coefficients = np.zeros([operator.n_real, operator.nu, operator.n_landau])
    spectral_coefficients[arbitrary_index] = 1.0
    
    # Transform from spectral space to real space
    function_values = operator.transform(
        spectral_coefficients, input_space="spectral", output_space="real"
    )
    
    # Generate expected eigenfunction in real space
    lattice = operator.lattice(output_space="real")
    expected_function_values = operator.eigenfunction(
        arbitrary_index, output_space="real", coordinates=lattice
    )
    
    assert np.allclose(function_values, expected_function_values)



def test_transforms_multiple_components_from_spectral_to_real_space():
    """
    Python test function to test the transformation of spectral coefficients with multiple components from spectral space to real space.
    """
    operator = DiracOperator(n_real=4, nu=4, n_landau=4)
    
    # Create spectral coefficients with multiple components
    arbitrary_index1 = [2, 3, 1]
    arbitrary_index2 = [1, 0, 2]
    spectral_coefficients = np.zeros([operator.n_real, operator.nu, operator.n_landau])
    spectral_coefficients[tuple(arbitrary_index1)] = 1.0
    spectral_coefficients[tuple(arbitrary_index2)] = 2.0
    
    # Transform from spectral space to real space
    function_values = operator.transform(
        spectral_coefficients, input_space="spectral", output_space="real"
    )
    
    # Generate expected function values in real space
    lattice = operator.lattice(output_space="real")
    expected_function_values1 = operator.eigenfunction(
        arbitrary_index1, output_space="real", coordinates=lattice
    )
    expected_function_values2 = operator.eigenfunction(
        arbitrary_index2, output_space="real", coordinates=lattice
    )
    expected_function_values = expected_function_values1 + 2.0 * expected_function_values2
    
    assert np.allclose(function_values, expected_function_values)


# Other way around left
def main():
    test_transforms_from_spectral_to_real_space()

if __name__ == "__main__":
    main()


