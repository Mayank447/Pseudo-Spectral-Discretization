#!/usr/bin/env python3
from pseudospectral.spectra.fermion2D import FreeFermion2D
import pytest


@pytest.mark.parametrize("L", [(1, 2), (9, 8), (0.1, 42.0)])
def test_provides_box_lengths_as_array(L):
    assert all(FreeFermion2D(3, 3, L_t=L[0], L_x=L[1]).L == L)


@pytest.mark.parametrize("num_points", [(1, 2), (9, 8), (1, 42)])
def test_provides_number_of_points_as_array(num_points):
    assert all(FreeFermion2D(*num_points).num_points == num_points)
