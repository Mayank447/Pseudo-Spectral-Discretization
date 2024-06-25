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
    )
