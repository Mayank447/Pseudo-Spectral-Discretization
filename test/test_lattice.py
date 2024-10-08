#!/usr/bin/env python3
import numpy as np


def test_lattice_real_basis(spectrum):
    """
    Python test function to test the lattice method of the Spectrum class in the real
    space.
    """

    lattice = spectrum.lattice(output_basis="real")
    expected = None

    if spectrum.spacetime_dimension == 1:
        expected = np.linspace(0, spectrum.L, spectrum.total_num_of_dof, endpoint=False)

    elif spectrum.spacetime_dimension == 2:
        t, x = np.meshgrid(
            np.linspace(0, spectrum.L[0], spectrum.num_points[0], endpoint=False),
            np.linspace(0, spectrum.L[1], spectrum.num_points[1], endpoint=False),
            indexing="ij",
        )
        expected = t.flatten(), x.flatten()
    assert np.equal(lattice, expected).all()


def test_lattice_spectral_basis(spectrum):
    """
    Python test function to test the lattice of the Dirac operator in spectral space.
    """
    result = spectrum.lattice(output_basis="spectral")
    expected = None

    if spectrum.spacetime_dimension == 1:
        # We're explicitly using the private data here because the public interface may
        # have been overridden by another spectrum.
        expected = spectrum._eigenvalues

    elif spectrum.spacetime_dimension == 2:
        expected = spectrum.p.reshape(spectrum.spacetime_dimension, -1)

    assert np.equal(result, expected).all()
