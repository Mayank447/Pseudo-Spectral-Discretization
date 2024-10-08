#!/usr/bin/env python3
import numpy as np
import pytest

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


def test_enforces_real_basis_for_input(spectrum):
    operator = DiracOperator(spectrum)
    input_vectors = np.eye(spectrum.total_num_of_dof)
    with pytest.raises(ValueError):
        (operator + operator).apply_to(input_vectors, input_basis="spectral")


def test_enforces_real_basis_for_output(spectrum):
    operator = DiracOperator(spectrum)
    input_vectors = np.eye(spectrum.total_num_of_dof)
    with pytest.raises(ValueError):
        (operator + operator).apply_to(input_vectors, output_basis="spectral")
