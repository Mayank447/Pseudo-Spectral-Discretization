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
