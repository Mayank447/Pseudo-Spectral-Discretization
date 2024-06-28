#!/usr/bin/env python3

from pseudospectral import DiracOperator
import numpy as np


def test_application_to_single_spectral_component():
    operator = DiracOperator(n_time=4, nu=4, n_landau=4)
    spectral_coefficients = np.zeros([operator.n_time, operator.nu, operator.n_landau])
    arbitrary_index = [0, 1, 2]
    spectral_coefficients[arbitrary_index] = 1.0
    assert (
        operator.apply_to(
            spectral_coefficients,
            input_space="spectral space",
            output_space="spectral space",
        )
        == operator.eigenvalue(arbitrary_index) * spectral_coefficients
    ).all()


def test_application_to_another_single_spectral_component():
    operator = DiracOperator(n_time=4, nu=4, n_landau=4)
    spectral_coefficients = np.zeros([operator.n_time, operator.nu, operator.n_landau])
    arbitrary_index = [0, 2, 0]
    spectral_coefficients[arbitrary_index] = 1.0
    assert (
        operator.apply_to(
            spectral_coefficients,
            input_space="spectral space",
            output_space="spectral space",
        )
        == operator.eigenvalue(arbitrary_index) * spectral_coefficients
    ).all()


def test_reports_correct_eigenvalue():
    n_time = 4
    operator = DiracOperator(n_time=n_time, nu=4, n_landau=4)
    arbitrary_index = [0, 0, 0]
    assert operator.eigenvalue(arbitrary_index) == np.pi / n_time


# write a table of eigenvalues and test all of them
#
# use approximative equality from pytest


def test_application_to_two_spectral_components():
    spectral_coefficients = np.zeros([operator.n_time, operator.nu, operator.n_landau])
    arbitrary_index1 = [0, 2, 0]
    arbitrary_index2 = [1, 1, 1]
    spectral_coefficients[arbitrary_index1] = 1.0
    spectral_coefficients[arbitrary_index2] = 2.0
    expected = np.array(spectral_coefficients)
    expected[arbitrary_index1] *= operator.eigenvalue(arbitrary_index1)
    expected[arbitrary_index2] *= operator.eigenvalue(arbitrary_index2)
    assert (
        operator.apply_to(
            spectral_coefficients,
            input_space="spectral space",
            output_space="spectral space",
        )
        == expected
    ).all()


# you can deduplicate test code via @pytest.parametrize() (or something like that...)
