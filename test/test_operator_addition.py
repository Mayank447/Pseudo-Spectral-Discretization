#!/usr/bin/env python3
import numpy as np

from pseudospectral.dirac_operator import DiracOperator


def test_add_twice(spectrum):
    operator = DiracOperator(spectrum)
    input_vectors = np.eye(spectrum.total_num_of_dof)
    assert np.allclose(
        (operator + operator).apply_to(input_vectors),
        2 * operator.apply_to(input_vectors),
    )


def test_add_three_times(spectrum):
    operator = DiracOperator(spectrum)
    input_vectors = np.eye(spectrum.total_num_of_dof)
    assert np.allclose(
        (operator + operator + operator).apply_to(input_vectors),
        3 * operator.apply_to(input_vectors),
    )
