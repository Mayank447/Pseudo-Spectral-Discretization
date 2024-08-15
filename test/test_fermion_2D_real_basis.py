from pseudospectral import DiracOperator, FreeFermion2D
import numpy as np
import pytest


############################################ TEST FUNCTION ############################################
def test_application_to_a_single_eigenfunction(spectrum_fermion2D, 
                                               arbitrary_index_single_eigenfunction_fermion2D, 
                                               arbitrary_single_coefficient
                                               ):
    """
    Python test function to test the application of the Dirac operator to a single eigenfunction in real space.
    """
    operator = DiracOperator(spectrum_fermion2D)
    index = 10

    eigenfunction = (
        spectrum_fermion2D.eigenfunction(index)(*spectrum_fermion2D.lattice(output_basis='real'))
    )
    print("Eigenfunction", eigenfunction)
    result = operator.apply_to(eigenfunction, input_basis="real", output_basis="real")
    assert np.isclose(eigenfunction, result).all()
    print(result)

    expected = eigenfunction * operator.spectrum.eigenvalues[index]
    print(operator.spectrum.eigenvalues[index])
    print(expected)
    assert np.isclose(result, expected).all()