#!/usr/bin/env python3

from pseudospectral import DiracOperator
import numpy as np


# everything is pseudocode in here: Fix the details!


def test_application_to_a_single_eigenfunction():
    operator = DiracOperator(...)
    arbitrary_index = [2, 3, 1]
    # for our simplest case that can probably be a regular cubic lattice for starters
    lattice = operator.lattice(output_space="real space")
    function_values = operator.eigenfunction(
        arbitrary_index, output_space="real space", coordinates=lattice
    )
    assert (
        operator.apply_to(
            function_values, input_space="real space", output_space="real space"
        )
        == operator.eigenvalue(arbitrary_index) * function_values
    )


# basically do the same tests as in spectral space here:
# apply_to another arbitrary_index
# apply_to a superposition of two eigenfunctions


def test_eigenfunction():
    assert False and "exact function values in real space"
    assert False and "eigenfunctions are orthogonal"


def test_lattice():
    assert False and "check the exact values because they are fixed for each operator"
    assert False and "potentially more properties"


# A simpler operator: Free Dirac operator == plain first derivative
#
# D = gamma^mu partial_mu =
#     (
#           partial_0                   partial_1 + i partial_2
#           partial_1 - i partial_2        -partial_0
#     )
#
# Eigenfunctions are productions of plane waves in each direction
# In a finite box with (anti-periodic boundary conditions), eigenvalues are linearly spaced with an offset depending on the boundary conditions.
