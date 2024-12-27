#!/usr/bin/env python3
import numpy as np


def slogdet(operator):
    """
    Compute the sign and logarithm of the determinant for operators.
    See np.linalg.slogdet for details.
    This is currently a naive implementation falling back to the matrix representation.
    Before use in production, this should be implemented in a smarter way.
    
    Update: the detX = prod eigenvalues, take log on both sides and this a sum of log
    Assuming no eigenvalue can be zero, otherwise the logdet is not defined in such cases.
    """
    eigval = operator.spectrum.eigenvalues
    sign = np.sum(np.sign(eigval))
    logdet = np.log(np.absolute(eigval))
    
    return sign, logdet