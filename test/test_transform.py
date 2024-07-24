from pseudospectral import Derivative1D
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
def arbitrary_single_coefficient():
    """
    Python fixture to initialize the single coefficient for the tests.
    """
    return 2.5


@pytest.fixture
def arbitrary_coefficients():
    """
    Python fixture to initialize the arbitrary coefficients for the tests.
    """
    return np.array([1.0, 2.0])


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
def test_transforms_from_real_to_spectral_space(spectrum, arbitrary_index_single_eigenfunction, arbitrary_single_coefficient):
    """
    Python test function to test the transformation of an eigenvector 
    from real space to spectral space.
    """

    sample_points = np.linspace(0, spectrum.L, spectrum.num_lattice_points, endpoint=False)
    eigenfunction = arbitrary_single_coefficient * spectrum.eigenfunction(arbitrary_index_single_eigenfunction)(sample_points)
    
    expected = arbitrary_single_coefficient * np.eye(spectrum.num_lattice_points)[arbitrary_index_single_eigenfunction, :]
    result = spectrum.transform(eigenfunction, input_space="real", output_space="spectral")
    assert np.isclose(expected, result).all()


def test_transforms_multiple_components_from_real_to_spectral_space(spectrum, arbitrary_index_two_eigenfunctions, arbitrary_coefficients):
    """
    Python test function to test the transformation of linear combination
    of eigenvectors from real space to spectral space.
    """

    sample_points = np.linspace(0, spectrum.L, spectrum.num_lattice_points, endpoint=False)

    eigenfunction_1 = arbitrary_coefficients[0] * spectrum.eigenfunction(arbitrary_index_two_eigenfunctions[0])(sample_points)
    eigenfunction_2 = arbitrary_coefficients[1] * spectrum.eigenfunction(arbitrary_index_two_eigenfunctions[1])(sample_points)
    
    expected = np.zeros(spectrum.num_lattice_points)
    expected[arbitrary_index_two_eigenfunctions] = arbitrary_coefficients
    result = spectrum.transform(eigenfunction_1 + eigenfunction_2, input_space="real", output_space="spectral")
    assert np.isclose(expected, result).all()



def test_transforms_from_spectral_to_real_space(spectrum, arbitrary_single_coefficient, arbitrary_index_single_eigenfunction):
    """
    Python test function to test the transformation of spectral coefficients
    with a single component from spectral space to real space.
    """

    spectral_vector = arbitrary_single_coefficient * np.eye(spectrum.num_lattice_points)[arbitrary_index_single_eigenfunction,:]
    result = spectrum.transform(spectral_vector, input_space="spectral", output_space="real")
    
    sample_points = np.linspace(0, spectrum.L, spectrum.num_lattice_points, endpoint=False)
    expected = arbitrary_single_coefficient * spectrum.eigenfunction(arbitrary_index_single_eigenfunction)(sample_points)
    assert np.isclose(expected, result).all()


def test_transforms_multiple_components_from_spectral_to_real_space(spectrum, arbitrary_coefficients, arbitrary_index_two_eigenfunctions):
    """
    Python test function to test the transformation of linear combination of eigenvectors in spectral space with arbitrary coefficients to real space.
    """

    spectral_coefficients = np.zeros(spectrum.num_lattice_points)
    spectral_coefficients[arbitrary_index_two_eigenfunctions] = arbitrary_coefficients
    
    # Transform from spectral space to real space
    result = spectrum.transform(
        spectral_coefficients, input_space="spectral", output_space="real"
    )
    
    # Generate expected function values in real space
    sample_points = np.linspace(0, spectrum.L, spectrum.num_lattice_points, endpoint=False)
    e1 = spectrum.eigenfunction(arbitrary_index_two_eigenfunctions[0])(sample_points)
    e2 = spectrum.eigenfunction(arbitrary_index_two_eigenfunctions[1])(sample_points)
    expected = np.column_stack((e1, e2)) @ arbitrary_coefficients
    assert np.isclose(expected, result).all()