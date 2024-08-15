#!/usr/bin/env python3


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

        if (input_basis in ["real", "spectral"]) and (output_basis in ["real", "spectral"]):
            temp_1 = self.spectrum.transform(coefficients, input_basis, "spectral")
            print("V^T", temp_1)
            temp_2 = self.spectrum.eigenvalues * temp_1
            # print(self.spectrum.eigenvalues)
            # print("ulta V", temp_2)
            temp_3 = self.spectrum.transform(temp_2, "spectral", output_basis)
            print("V", temp_3)
            return temp_3

        else:
            raise ValueError(f"Unsupported space transformation from {input_basis} to {output_basis}.")
