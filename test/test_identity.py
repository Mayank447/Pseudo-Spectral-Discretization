#!/usr/bin/env python3
import numpy as np

from pseudospectral.dirac_operator import DiracOperator
from pseudospectral import Identity


def test_identity_is_identity(spectrum):
    input_vectors = np.eye(spectrum.total_num_of_dof)
    assert np.allclose(
        DiracOperator(Identity(spectrum)).apply_to(input_vectors), input_vectors
    )
