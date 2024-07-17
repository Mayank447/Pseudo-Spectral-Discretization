#!/usr/bin/env python3

from pseudospectral import DiracOperator
import numpy as np
import pytest


def test_application_to_single_spectral_component():
    operator = DiracOperator(n_time=4, nu=4, n_landau=4)
    spectral_coefficients = np.zeros([operator.n_time, operator.nu, operator.n_landau])
    arbitrary_index = (0, 1, 2)
    spectral_coefficients[arbitrary_index] = 1.0
    result = operator.apply_to(
        spectral_coefficients,
        input_space="spectral space",
        output_space="spectral space",
    )
    expected = operator.eigenvalue(arbitrary_index) * spectral_coefficients
    assert np.allclose(result, expected)


def test_application_to_another_single_spectral_component():
    operator = DiracOperator(n_time=4, nu=4, n_landau=4)
    spectral_coefficients = np.zeros([operator.n_time, operator.nu, operator.n_landau])
    arbitrary_index = (0, 2, 0)
    spectral_coefficients[arbitrary_index] = 1.0
    result = operator.apply_to(
        spectral_coefficients,
        input_space="spectral space",
        output_space="spectral space",
    )
    expected = operator.eigenvalue(arbitrary_index) * spectral_coefficients
    assert np.allclose(result, expected)


def test_reports_correct_eigenvalue():
    n_time = 4
    operator = DiracOperator(n_time=n_time, nu=4, n_landau=4)
    arbitrary_index = (0, 0, 0)
    expected_eigenvalue = np.pi / n_time
    assert np.isclose(operator.eigenvalue(arbitrary_index), expected_eigenvalue)


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


def test_application_to_two_spectral_components():
    operator = DiracOperator(n_time=4, nu=4, n_landau=4)
    spectral_coefficients = np.zeros([operator.n_time, operator.nu, operator.n_landau])
    arbitrary_index1 = (0, 2, 0)
    arbitrary_index2 = (1, 1, 1)
    spectral_coefficients[arbitrary_index1] = 1.0
    spectral_coefficients[arbitrary_index2] = 2.0
    result = operator.apply_to(
        spectral_coefficients,
        input_space="spectral space",
        output_space="spectral space",
    )
    expected = np.array(spectral_coefficients)
    expected[arbitrary_index1] *= operator.eigenvalue(arbitrary_index1)
    expected[arbitrary_index2] *= operator.eigenvalue(arbitrary_index2)
    assert np.allclose(result, expected)


# write a table of eigenvalues and test all of them

# use approximative equality from pytests


# you can deduplicate test code via @pytest.parametrize() (or something like that...)
