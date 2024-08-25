#!/usr/bin/env python3
from pseudospectral.spectra.fermion2D import FreeFermion2D
import pytest
import numpy as np


@pytest.mark.parametrize("L", [(1, 2), (9, 8), (0.1, 42.0)])
def test_provides_box_lengths_as_array(L):
    assert all(FreeFermion2D([3, 3], L).L == L)


@pytest.mark.parametrize("num_points", [(1, 2), (9, 8), (1, 42)])
def test_provides_number_of_points_as_array(num_points):
    assert all(FreeFermion2D(num_points).num_points == num_points)


@pytest.mark.parametrize("num_points,L", [((1, 2), (3, 4)), ((9, 8), (0.1, 1000)), ((1, 42), (1, 1))])
def test_provides_lattice_spacing_as_array(num_points, L):
    assert all(FreeFermion2D(num_points, L).a == np.array(L) / num_points)


@pytest.mark.parametrize("num_points", [(1, 2), (9, 8), (1, 42)])
def test_defaults_to_unit_lattice_spacing(num_points):
    assert all(FreeFermion2D(num_points).a == np.ones_like(num_points))


@pytest.mark.parametrize("num_points,L", [((1, 2), (3, 4)), ((9, 8), (0.1, 1000)), ((1, 42), (1, 1))])
def test_exposes_x(num_points, L):
    assert np.allclose(FreeFermion2D(num_points, L).x, np.meshgrid(np.linspace(0, L[0], num_points[0], endpoint=False), np.linspace(0, L[1], num_points[1], endpoint=False), indexing="ij"))


@pytest.mark.parametrize("num_points,L", [((1, 2), (3, 4)), ((9, 8), (0.1, 1000)), ((1, 42), (1, 1))])
def test_exposes_momenta(num_points, L):
    p_t = 2.0j * np.pi * np.fft.fftfreq(num_points[0], L[0] / num_points[0])
    p_x = 2.0j * np.pi * np.fft.fftfreq(num_points[1], L[1] / num_points[1])
    assert np.allclose(FreeFermion2D(num_points, L).p, np.array(np.meshgrid(p_t, p_x, indexing="ij")))
