#!/usr/bin/env python3
from pseudospectral.spectra.fermion2D import FreeFermion2D
import pytest
import numpy as np


@pytest.mark.parametrize("L", [(1, 2), (9, 8), (0.1, 42.0)])
def test_provides_box_lengths_as_array(L):
    assert all(FreeFermion2D(3, 3, L_t=L[0], L_x=L[1]).L == L)


@pytest.mark.parametrize("num_points", [(1, 2), (9, 8), (1, 42)])
def test_provides_number_of_points_as_array(num_points):
    assert all(FreeFermion2D(*num_points).num_points == num_points)


@pytest.mark.parametrize("num_points,L", [((1, 2), (3, 4)), ((9, 8), (0.1, 1000)), ((1, 42), (1, 1))])
def test_provides_lattice_spacing_as_array(num_points, L):
    assert all(FreeFermion2D(*num_points, *L).a == np.array(L) / num_points)
