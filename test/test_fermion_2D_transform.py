import numpy as np

## Some Fixtures like Spectrum, arbitrary_index_single_eigenfunction, arbitrary_single_coefficient, arbitrary_index_multiple_eigenfunctions
## are defined in the conftest.py file.and imported in all the test files automatically.

########################################## HELPER FUNCTIONS ##########################################
def arbitrary_multiple_coefficients(length=1):
    """
    Python function to initialize a numpy array of the given length
    with arbitrary coefficients sampled from a normal distribution for the tests.
    """
    return np.random.randn(length)


############################################ TEST FUNCTION ############################################
def test_transforms_from_real_to_spectral_basis(
    spectrum_fermion2D, 
    arbitrary_index_single_eigenfunction_fermion2D, 
    arbitrary_single_coefficient
):
    """
    Python test function to test the transformation of an eigenvector
    from real space to spectral space.
    """

    eigenfunction = (arbitrary_single_coefficient 
                     * spectrum_fermion2D.eigenfunction(arbitrary_index_single_eigenfunction_fermion2D)(*spectrum_fermion2D.lattice())
                    )
    
    result = spectrum_fermion2D.transform(eigenfunction, input_basis="real", output_basis="spectral")
    expected = (
        arbitrary_single_coefficient 
        * np.eye(spectrum_fermion2D.vector_length)[arbitrary_index_single_eigenfunction_fermion2D, :]
    )
    assert np.allclose(expected, result)



def test_transforms_multiple_components_from_real_to_spectral_basis(
    spectrum_fermion2D, 
    arbitrary_index_multiple_eigenfunctions_fermion_2D
):
    """
    Python test function to test the transformation of linear combination
    of eigenvectors from real space to spectral space.
    """

    arbitrary_coefficients = arbitrary_multiple_coefficients(len(arbitrary_index_multiple_eigenfunctions_fermion_2D))
    eigenfunctions = spectrum_fermion2D.eigenfunction(arbitrary_index_multiple_eigenfunctions_fermion_2D)(*spectrum_fermion2D.lattice())
    
    result = spectrum_fermion2D.transform(
                np.sum(arbitrary_coefficients[:, np.newaxis] * eigenfunctions, axis = 0), 
                input_basis="real", 
                output_basis="spectral"
             )

    expected = np.zeros(spectrum_fermion2D.vector_length)
    expected[arbitrary_index_multiple_eigenfunctions_fermion_2D] = arbitrary_coefficients
    assert np.allclose(expected, result)


def test_transforms_from_spectral_to_real_basis(
    spectrum_fermion2D, 
    arbitrary_single_coefficient, 
    arbitrary_index_single_eigenfunction_fermion2D
):
    """
    Python test function to test the transformation of spectral coefficients
    with a single component from spectral space to real space.
    """

    spectral_vector = arbitrary_single_coefficient * np.eye(spectrum_fermion2D.vector_length)[arbitrary_index_single_eigenfunction_fermion2D, :]
    result = spectrum_fermion2D.transform(
                spectral_vector, 
                input_basis="spectral",
                output_basis="real"
            )   

    expected = arbitrary_single_coefficient * spectrum_fermion2D.eigenfunction(arbitrary_index_single_eigenfunction_fermion2D)(*spectrum_fermion2D.lattice())
    assert np.allclose(expected, result)



def test_transforms_multiple_components_from_spectral_to_real_basis(
    spectrum_fermion2D, 
    arbitrary_index_multiple_eigenfunctions_fermion_2D
):
    """
    Python test function to test the transformation of linear combination of eigenvectors with arbitrary coefficients
    in spectral space to real space.
    """

    arbitrary_coefficients = arbitrary_multiple_coefficients(len(arbitrary_index_multiple_eigenfunctions_fermion_2D))
    spectral_coefficients = np.zeros(spectrum_fermion2D.vector_length)
    spectral_coefficients[arbitrary_index_multiple_eigenfunctions_fermion_2D] = arbitrary_coefficients

    # Transform from spectral space to real space
    result = spectrum_fermion2D.transform(spectral_coefficients, input_basis="spectral", output_basis="real")

    # Generate expected function values in real space
    eigenfunctions = spectrum_fermion2D.eigenfunction(arbitrary_index_multiple_eigenfunctions_fermion_2D)(*spectrum_fermion2D.lattice())
    expected = np.sum(arbitrary_coefficients[:, np.newaxis] * eigenfunctions, axis=0)
    assert np.allclose(expected, result)