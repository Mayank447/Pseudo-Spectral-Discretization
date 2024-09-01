#!/usr/bin/env python3
import numpy as np

from pseudospectral import ElementwiseSpectralMultiplication, DiracOperator


def test_identity_is_identity(spectrum):
    input_vectors = np.eye(spectrum.total_num_of_dof)
    assert np.allclose(
        DiracOperator(ElementwiseSpectralMultiplication(spectrum)).apply_to(
            input_vectors
        ),
        input_vectors,
    )


def test_reproduces_operator(spectrum):
    input_vectors = np.eye(spectrum.total_num_of_dof)
    field_values = spectrum.eigenvalues
    assert np.allclose(
        DiracOperator(
            ElementwiseSpectralMultiplication(spectrum, field_values)
        ).apply_to(input_vectors),
        DiracOperator(spectrum).apply_to(input_vectors),
    )


def test_works_in_spectral_space(spectrum):
    input_vectors = np.eye(spectrum.total_num_of_dof)
    field_values = np.arange(spectrum.total_num_of_dof)
    assert np.allclose(
        DiracOperator(
            ElementwiseSpectralMultiplication(spectrum, field_values)
        ).apply_to(input_vectors, "spectral", "spectral"),
        np.diag(field_values),
    )
