#!/usr/bin/env python3
import numpy as np


def test_broadcasting_many_dims(spectrum):
    shape = (1, 2, 3, spectrum.total_num_of_dof)
    assert spectrum.transform(np.zeros(shape), "real", "spectral").shape == shape


def test_broadcasting_single_input(spectrum):
    shape = (spectrum.total_num_of_dof,)
    assert spectrum.transform(np.zeros(shape), "real", "spectral").shape == shape
