from pseudospectral import Derivative1D
import numpy as np
import pytest


def test_transforms_from_real_to_spectral_space(L=4, n=4):
    """
    Python test function to test the transformation of an eigenvector from real space to spectral space.
    """
    spectrum = Derivative1D(L, n)
    arbitrary_index = 1
    sample_points = np.linspace(0,L,n,endpoint=False)
    eigenfunction = spectrum.eigenfunction(arbitrary_index)(sample_points)
    result = spectrum.transform(eigenfunction, input_space="real", output_space="spectral")
    
    # Create spectral coefficients with a single component
    expected = np.eye(L)[arbitrary_index,:]
    print("Expected:", expected)
    assert np.isclose(expected, result).all()



def test_transforms_multiple_components_from_real_to_spectral_space(L=4, n=4):
    """
    Python test function to test the transformation of real coefficients with multiple components from real space to spectral space.
    """
    # Create spectral coefficients with multiple components
    spectrum = Derivative1D(L, n)
    arbitrary_index = [1,3]
    arbitrary_coefficients = [1.0, 2.0]
    sample_points = np.linspace(0,L,n,endpoint=False)

    eigenfunction_1 = arbitrary_coefficients[0] * spectrum.eigenfunction(arbitrary_index[0])(sample_points)
    eigenfunction_2 = arbitrary_coefficients[1] * spectrum.eigenfunction(arbitrary_index[1])(sample_points)
    result = spectrum.transform(eigenfunction_1 + eigenfunction_2, input_space="real", output_space="spectral")
    
    # Create spectral coefficients with a single component
    expected = np.zeros(L)
    expected[arbitrary_index] = arbitrary_coefficients
    assert np.isclose(expected, result).all()



def test_transforms_from_spectral_to_real_space(L=4, n=4):
    """
    Python test function to test the transformation of spectral coefficients with a single component from spectral space to real space.
    """
    spectrum = Derivative1D(L, n)
    arbitrary_index = 2
    
    # Create spectral coefficients with a single component
    spectral_coefficients = np.eye(L)[arbitrary_index,:]
    
    # Transform from spectral space to real space
    result = spectrum.transform(
        spectral_coefficients, input_space="spectral", output_space="real"
    )
    
    # Generate expected eigenfunction in real space
    sample_points = np.linspace(0,L,n,endpoint=False)
    expected = spectrum.eigenfunction(arbitrary_index)(sample_points)
    assert np.isclose(expected, result).all()



def test_transforms_multiple_components_from_spectral_to_real_space(L=4, n=4):
    """
    Python test function to test the transformation of spectral coefficients with multiple components from spectral space to real space.
    """
    spectrum = Derivative1D(L, n)
    
    # Create spectral coefficients with multiple components
    arbitrary_index = [2, 3]
    arbitrary_coefficents = [1.0, 2.0]
    spectral_coefficients = np.zeros(L)
    spectral_coefficients[arbitrary_index] = arbitrary_coefficents
    
    # Transform from spectral space to real space
    result = spectrum.transform(
        spectral_coefficients, input_space="spectral", output_space="real"
    )
    
    # Generate expected function values in real space
    sample_points = np.linspace(0,L,n,endpoint=False)
    e1 = spectrum.eigenfunction(arbitrary_index[0])(sample_points)
    e2 = spectrum.eigenfunction(arbitrary_index[1])(sample_points)
    expected = arbitrary_coefficents[0] * e1 + arbitrary_coefficents[1] * e2
    assert np.isclose(expected, result).all()