#!/usr/bin/env python3
import numpy as np


def slogdet(operator):
    """
    Compute the sign and logarithm of the determinant for operators.
    See np.linalg.slogdet for details.
    This is currently a naive implementation falling back to the matrix representation.
    Before use in production, this should be implemented in a smarter way.
    """
    return np.linalg.slogdet(
        operator.apply_to(np.eye(operator.spectrum.total_num_of_dof))
    )
