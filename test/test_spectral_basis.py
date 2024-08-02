#!/usr/bin/env python3

from pseudospectral import DiracOperator
import numpy as np
import pytest


########################################## FIXTURES ##########################################
@pytest.fixture
def arbitrary_index_single_eigenfunction(spectrum):
    """
    Python fixture to initialize the arbitrary index for the single eigenfunction test.
    """
    return np.arange(spectrum.L)


@pytest.fixture
def arbitrary_index_two_eigenfunctions():
    """
    Python fixture to initialize the arbitrary index for the two eigenfunctions test.
    """
    return np.array([1, 2])


########################################### TEST FUNCTION ############################################
@pytest.mark.parametrize("arbitrary_index", np.arange(4))
def test_application_to_a_single_eigenfunction(spectrum, arbitrary_index):
    """
    Python test function to test the application of the Dirac operator to a single eigenfunction in real space.
    """
    operator = DiracOperator(spectrum)

    eigenfunction = np.eye(spectrum.num_lattice_points)[arbitrary_index, :]
    expected = eigenfunction * spectrum.eigenvalues[arbitrary_index]

    result = operator.apply_to(
        eigenfunction, input_basis="spectral", output_basis="spectral"
    )
    assert np.isclose(result, expected).all()


@pytest.mark.parametrize(
    "arbitrary_index", [[x, y] for x in range(4) for y in range(4)]
)
def test_application_to_superposition_of_two_eigenfunctions(spectrum, arbitrary_index):
    """
    Python test function to test the application of the Dirac operator to a superposition of two eigenfunctions.
    """
    operator = DiracOperator(spectrum)

    eigenfunction_1, eigenfunction_2 = np.eye(spectrum.num_lattice_points)[
        arbitrary_index, :
    ]
    print(eigenfunction_1, eigenfunction_2)
    expected = (
        np.column_stack((eigenfunction_1, eigenfunction_2))
        @ spectrum.eigenvalues[arbitrary_index]
    )

    result = operator.apply_to(
        eigenfunction_1 + eigenfunction_2,
        input_basis="spectral",
        output_basis="spectral",
    )
    assert np.isclose(result, expected).all()


def test_lattice_spectral_basis(spectrum):
    """
    Python test function to test the lattice of the Dirac operator in spectral space.
    """
    result = spectrum.lattice(output_basis="spectral")
    expected = 2 * np.pi * np.arange(spectrum.num_lattice_points) / spectrum.L
    assert np.equal(result, expected).all()
