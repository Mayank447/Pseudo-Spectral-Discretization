from pseudospectral import DiracOperator
import numpy as np
import pytest


########################################## HELPER_FUNCTIONS ##########################################
def arbitrary_multiple_coefficients(length=1):
    """
    Python function to initialize a numpy array of the given length
    with arbitrary coefficients sampled from a normal distribution for the tests.
    """
    return np.random.randn(length)


############################################ TEST FUNCTION ############################################
def test_application_to_a_single_eigenfunction(
        spectrum_fermion2D, 
        arbitrary_index_single_eigenfunction_fermion2D, 
        arbitrary_single_coefficient
    ):
    """
    Python test function to test the application of the Dirac operator to a single eigenfunction in real space.
    """
    operator = DiracOperator(spectrum_fermion2D)
    eigenfunction = (
        arbitrary_single_coefficient * spectrum_fermion2D.eigenfunction(arbitrary_index_single_eigenfunction_fermion2D)(*spectrum_fermion2D.lattice())
    )

    result = operator.apply_to(eigenfunction, input_basis="real", output_basis="real")
    expected = (
        operator.spectrum.eigenvalues[arbitrary_index_single_eigenfunction_fermion2D] 
        * eigenfunction
    )
    assert np.allclose(result, expected)



def test_application_to_superposition_of_eigenfunctions(
    spectrum_fermion2D, 
    arbitrary_index_multiple_eigenfunctions_fermion_2D
):
    """
    Python test function to test the application of the Dirac operator to a superposition of multiple eigenfunctions.
    """
    operator = DiracOperator(spectrum_fermion2D)
    arbitrary_coefficients = arbitrary_multiple_coefficients(len(arbitrary_index_multiple_eigenfunctions_fermion_2D))

    superposition = (
        arbitrary_coefficients[:, np.newaxis] * 
        spectrum_fermion2D.eigenfunction(arbitrary_index_multiple_eigenfunctions_fermion_2D)(*spectrum_fermion2D.lattice())
    )

    expected = np.sum(
        (spectrum_fermion2D.eigenvalues[arbitrary_index_multiple_eigenfunctions_fermion_2D])[:, np.newaxis]
        * superposition, axis=0
    )

    result = operator.apply_to(np.sum(superposition, axis=0), input_basis="real", output_basis="real")
    assert np.allclose(result, expected)


def test_lattice_real_basis(spectrum_fermion2D):
    """
    Python test function to test the lattice method of the Spectrum class in the real space.
    """

    lattice = spectrum_fermion2D.lattice(output_basis="real")
    
    t, x = np.meshgrid(
                np.linspace(0, spectrum_fermion2D.L_t, spectrum_fermion2D.n_t, endpoint=False), 
                np.linspace(0, spectrum_fermion2D.L_x, spectrum_fermion2D.n_x, endpoint=False), 
                indexing="ij"
            )
    excepted = (t.flatten(), x.flatten())
    assert np.equal(lattice, excepted).all()