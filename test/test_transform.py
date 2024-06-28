#!/usr/bin/env python3

from pseudospectral import DiracOperator
import numpy as np


def test_transforms_from_spectral_to_real_space():
    operator = DiracOperator(...)
    spectral_coefficients = (
        ...
    )  # again: start with single component and then add more tests with more involved setups
    assert operator.transform(
        spectral_coefficients, input_space="spectral space", output_space="real space"
    ) == operator.eigenfunction(arbitrary_index)


# and of course, the other way round:
expected = np.zeros()
expected[arbitrary_index] = 1.0
assert (
    operator.transform(
        function_values, input_space="real space", output_space="spectral space"
    )
    == expected
)
