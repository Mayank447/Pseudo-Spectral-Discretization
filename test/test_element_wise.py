#!/usr/bin/env python3
import numpy as np
import pytest

from pseudospectral import (
    DiracOperator,
    ElementwiseSpectralMultiplication,
    ElementwiseRealMultiplication,
)


@pytest.fixture()
def spectrum_not_elementwise(spectrum):
    if not isinstance(
        spectrum, (ElementwiseSpectralMultiplication, ElementwiseRealMultiplication)
    ):
        return spectrum
    pytest.skip()


@pytest.mark.parametrize(
    "ElementWise", [ElementwiseSpectralMultiplication, ElementwiseRealMultiplication]
)
def test_elementwise_are_instances_of_themselves(spectrum_not_elementwise, ElementWise):
    assert isinstance(ElementWise(spectrum_not_elementwise), ElementWise)


@pytest.mark.parametrize(
    "ElementWise", [ElementwiseSpectralMultiplication, ElementwiseRealMultiplication]
)
def test_identity_is_identity(spectrum_not_elementwise, ElementWise):
    input_vectors = np.eye(spectrum_not_elementwise.total_num_of_dof)
    assert np.allclose(
        DiracOperator(ElementWise(spectrum_not_elementwise, 1)).apply_to(input_vectors),
        input_vectors,
    )


def test_reproduces_operator(spectrum_not_elementwise):
    input_vectors = np.eye(spectrum_not_elementwise.total_num_of_dof)
    field_values = spectrum_not_elementwise.eigenvalues
    assert np.allclose(
        DiracOperator(
            ElementwiseSpectralMultiplication(spectrum_not_elementwise, field_values)
        ).apply_to(input_vectors),
        DiracOperator(spectrum_not_elementwise).apply_to(input_vectors),
    )


def test_works_in_spectral_space(spectrum_not_elementwise):
    input_vectors = np.eye(spectrum_not_elementwise.total_num_of_dof)
    field_values = np.arange(spectrum_not_elementwise.total_num_of_dof)
    assert np.allclose(
        DiracOperator(
            ElementwiseSpectralMultiplication(spectrum_not_elementwise, field_values)
        ).apply_to(input_vectors, "spectral", "spectral"),
        np.diag(field_values),
    )


def test_works_in_real_space(spectrum_not_elementwise):
    input_vectors = np.eye(spectrum_not_elementwise.total_num_of_dof)
    field_values = np.arange(spectrum_not_elementwise.total_num_of_dof)
    assert np.allclose(
        DiracOperator(
            ElementwiseRealMultiplication(spectrum_not_elementwise, field_values)
        ).apply_to(input_vectors, "spectral", "spectral"),
        np.diag(field_values),
    )
