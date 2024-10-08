#!/usr/bin/env python3
import numpy as np


class DiracOperator:
    """
    Dirac operator class to represent the Dirac operator in the pseudospectral method.
    Args: Spectrum object
    """

    def __init__(self, spectrum):
        """
        Initialize the Dirac operator with the given spectrum.
        """
        self.spectrum = spectrum

    def apply_to(self, coefficients, input_basis="real", output_basis="real"):
        """
        Apply the Dirac operator to the given spectral coefficients
        as per the given input and output spaces.
        """

        if (input_basis in ["real", "spectral"]) and (
            output_basis in ["real", "spectral"]
        ):
            temp_1 = self.spectrum.transform(coefficients, input_basis, "spectral")
            temp_2 = self.spectrum.eigenvalues * temp_1
            temp_3 = self.spectrum.transform(temp_2, "spectral", output_basis)
            return temp_3

        else:
            message = (
                "Unsupported space transformation from "
                f"{input_basis} to {output_basis}."
            )
            raise ValueError(message)

    def __add__(self, other):
        return CompositeOperator((self, other), lambda f: np.sum(f, axis=0))


class CompositeOperator:
    def __init__(self, operators, operation):
        self.operators = operators
        self.operation = operation

    def __add__(self, other):
        return CompositeOperator((self, other), lambda f: np.sum(f, axis=0))

    def apply_to(self, coefficients, input_basis="real", output_basis="real"):
        """
        As the multiple operators might not commute, we cannot offer a version in the
        spectral basis because that would be ambiguous.

        This is a general purpose implementation. Special cases should be implemented as
        needed to increase performance.
        """
        if input_basis != "real" or output_basis != "real":
            message = (
                "We cannot know a spectral basis for a composite operator. "
                "Please use a real basis for input and output. "
                f"You gave {input_basis=} and {output_basis=}"
            )
            raise ValueError(message)
        return self.operation(
            [operator.apply_to(coefficients) for operator in self.operators]
        )
