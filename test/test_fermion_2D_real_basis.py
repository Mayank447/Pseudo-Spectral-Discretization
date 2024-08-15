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
    sign = 1
    index_t, index_x, sign = arbitrary_index_single_eigenfunction_fermion2D
    
    print(spectrum_fermion2D.lattice(output_basis='real'))
    eigenfunction = (
        spectrum_fermion2D.eigenfunction((index_t, index_x), sign)(*spectrum_fermion2D.lattice(output_basis='real'))
    )
    # print(spectrum_fermion2D.scalar_product(eigenfunction, eigenfunction))
    print(eigenfunction.shape)
    print(eigenfunction)
    print(index_t, index_x, sign)
    print(2 * (spectrum_fermion2D.n_x * index_t + index_x) + int(0.5*(1-sign)))
    result = operator.apply_to(eigenfunction, input_basis="real", output_basis="real")
    assert np.isclose(eigenfunction, result).all()
    print(result)

    expected = eigenfunction * operator.spectrum.eigenvalues[2 * (spectrum_fermion2D.n_x * index_t + index_x) + int(0.5*(1-sign))]
    print(operator.spectrum.eigenvalues[2 * (spectrum_fermion2D.n_x * index_t + index_x) + int(0.5*(1-sign))])
    print(expected)
    assert np.isclose(result, expected).all()