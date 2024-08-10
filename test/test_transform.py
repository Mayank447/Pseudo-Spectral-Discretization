import numpy as np
import pytest

########################################## FIXTURES ##########################################
@pytest.fixture
def arbitrary_single_coefficient():
    """
    Python fixture to initialize the single coefficient for the tests.
    """
    return 1


@pytest.fixture
def arbitrary_multiple_coefficients():
    """
    Python fixture to initialize the arbitrary coefficients for the tests.
    """
    return np.array([1.0, 2.0])


@pytest.fixture
def arbitrary_index_single_eigenfunction(index=1):
    """
    Python fixture to initialize the arbitrary index for the single eigenfunction test.
    """
    return np.array(index)


@pytest.fixture
def arbitrary_index_multiple_eigenfunctions():
    """
    Python fixture to initialize the arbitrary index for the two eigenfunctions test.
    """
    return np.array([1, 2])


############################################ TEST FUNCTION ############################################
def test_transforms_from_real_to_spectral_basis(
    spectrum, arbitrary_index_single_eigenfunction, arbitrary_single_coefficient
):
    """
    Python test function to test the transformation of an eigenvector
    from real space to spectral space.
    """

    sample_points = np.linspace(
        0, spectrum.L, spectrum.num_lattice_points, endpoint=False
    )
    eigenfunction = arbitrary_single_coefficient * spectrum.eigenfunction(arbitrary_index_single_eigenfunction)(sample_points)
    result = spectrum.transform(
        eigenfunction, input_basis="real", output_basis="spectral"
    )

    expected = (
        arbitrary_single_coefficient
        * np.eye(spectrum.num_lattice_points)[arbitrary_index_single_eigenfunction, :]
    )
    assert np.isclose(expected, result).all()


def test_transforms_multiple_components_from_real_to_spectral_basis(
    spectrum, arbitrary_index_multiple_eigenfunctions, arbitrary_multiple_coefficients
):
    """
    Python test function to test the transformation of linear combination
    of eigenvectors from real space to spectral space.
    """

    sample_points = np.linspace(
        0, spectrum.L, spectrum.num_lattice_points, endpoint=False
    )

    eigenfunctions = spectrum.eigenfunction(arbitrary_index_multiple_eigenfunctions)(sample_points.reshape(-1, 1))
    result = spectrum.transform(
        eigenfunctions @ arbitrary_multiple_coefficients, input_basis="real", output_basis="spectral"
    )

    expected = np.zeros(spectrum.num_lattice_points)
    expected[arbitrary_index_multiple_eigenfunctions] = arbitrary_multiple_coefficients
    assert np.isclose(expected, result).all()


def test_transforms_from_spectral_to_real_basis(
    spectrum, arbitrary_single_coefficient, arbitrary_index_single_eigenfunction
):
    """
    Python test function to test the transformation of spectral coefficients
    with a single component from spectral space to real space.
    """

    spectral_vector = (
        arbitrary_single_coefficient
        * np.eye(spectrum.num_lattice_points)[arbitrary_index_single_eigenfunction, :]
    )
    result = spectrum.transform(
        spectral_vector, input_basis="spectral", output_basis="real"
    )

    sample_points = np.linspace(
        0, spectrum.L, spectrum.num_lattice_points, endpoint=False
    )
    expected = arbitrary_single_coefficient * spectrum.eigenfunction(arbitrary_index_single_eigenfunction)(sample_points)
    assert np.isclose(expected, result).all()


def test_transforms_multiple_components_from_spectral_to_real_basis(
    spectrum, arbitrary_multiple_coefficients, arbitrary_index_multiple_eigenfunctions
):
    """
    Python test function to test the transformation of linear combination of eigenvectors
    in spectral space with arbitrary coefficients to real space.
    """

    spectral_coefficients = np.zeros(spectrum.num_lattice_points)
    spectral_coefficients[arbitrary_index_multiple_eigenfunctions] = (
        arbitrary_multiple_coefficients
    )

    # Transform from spectral space to real space
    result = spectrum.transform(
        spectral_coefficients, input_basis="spectral", output_basis="real"
    )

    # Generate expected function values in real space
    sample_points = np.linspace(
        0, spectrum.L, spectrum.num_lattice_points, endpoint=False
    )
    eigenfunctions = spectrum.eigenfunction(arbitrary_index_multiple_eigenfunctions)(sample_points.reshape(-1, 1))
    expected = eigenfunctions @ arbitrary_multiple_coefficients
    assert np.isclose(expected, result).all()
