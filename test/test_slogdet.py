#!/usr/bin/env python3
import numpy as np
from conftest import SPECTRA
import pytest

from pseudospectral import DiracOperator, Derivative1D, FreeFermion2D, slogdet


def has_exact_zero_modes(spec):
    return np.isclose(spec["type"](**spec["config"]).eigenvalues, 0.0).any()


# exact zero modes are difficult, we'll treat them separately
SPECTRA_WITHOUT_EXACT_ZERO_MODES = [
    # these are fine
    spec
    for spec in SPECTRA
    if not has_exact_zero_modes(spec)
] + [
    # we'll add a small mass to the rest, so they have no exact zero modes anymore
    spec | (spec["config"] | {"m": 0.01})
    for spec in SPECTRA
    if has_exact_zero_modes(spec) and isinstance(spec["type"], FreeFermion2D)
]


@pytest.fixture(params=SPECTRA_WITHOUT_EXACT_ZERO_MODES)
def spectrum_without_exact_zero_mode(request):
    return request.param["type"](**request.param["config"])


def test_slogdet_without_exact_zero_modes(spectrum_without_exact_zero_mode):
    sign, magnitude = slogdet(DiracOperator(spectrum_without_exact_zero_mode))
    eigenvalues = spectrum_without_exact_zero_mode.eigenvalues
    expected_sign = np.prod(eigenvalues / np.abs(eigenvalues))
    expected_magnitude = np.sum(np.log(np.abs(eigenvalues)))
    assert np.isclose(sign, expected_sign)
    assert np.isclose(magnitude, expected_magnitude)


# This needs a bit of care because we must avoid too large rounding errors. So, choosing
# L close to the number of lattice points helps. In general, don't use too large values.
SPECTRA_WITH_EXACT_ZERO_MODES = [
    {"type": Derivative1D, "config": {"total_num_lattice_points": 11, "L": 21}},
    {"type": Derivative1D, "config": {"total_num_lattice_points": 21, "L": 21}},
    {"type": Derivative1D, "config": {"total_num_lattice_points": 31, "L": 21}},
    {
        "type": FreeFermion2D,
        "config": {"num_points": [3, 3], "L": [1, 1], "mu": 0, "m": 0},
    },
    {
        "type": FreeFermion2D,
        # This configuration is already quite borderline. Lifting one 5 to a 7 makes
        # this blow up on my machine.
        "config": {"num_points": [5, 5], "L": [5, 5], "mu": 0, "m": 0},
    },
]


@pytest.fixture(params=SPECTRA_WITH_EXACT_ZERO_MODES)
def spectrum_with_exact_zero_mode(request):
    return request.param["type"](**request.param["config"])


def test_slogdet_with_exact_zero_modes(spectrum_with_exact_zero_mode):
    sign, magnitude = slogdet(DiracOperator(spectrum_with_exact_zero_mode))
    assert np.isclose(sign * np.exp(magnitude), 0.0)
