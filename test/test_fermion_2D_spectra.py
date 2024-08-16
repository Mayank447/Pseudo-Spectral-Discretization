import numpy as np
import pytest

## Some Fixtures like spectrum_fermion2D, arbitrary_index_single_eigenfunction, arbitrary_single_coefficient, arbitrary_index_multiple_eigenfunctions
## are defined in the conftest.py file.and imported in all the test files automatically.


@pytest.fixture()
def eigenfunctions(spectrum_fermion2D):
    t, x = spectrum_fermion2D.lattice()
    return spectrum_fermion2D.eigenfunction(np.arange(spectrum_fermion2D.vector_length))(t.reshape(-1, 1), x.reshape(-1, 1))


# def test_orthonormality(spectrum_fermion2D, eigenfunctions):
#     print(spectrum_fermion2D.scalar_product(eigenfunctions[0], eigenfunctions[2]))
#     assert np.allclose(
#         spectrum_fermion2D.scalar_product(eigenfunctions, eigenfunctions),
#         np.eye(spectrum_fermion2D.vector_length),
#     )


def test_back_and_forth_transform_is_identity(spectrum_fermion2D, eigenfunctions):
    e = eigenfunctions
    assert np.allclose(
        spectrum_fermion2D.transform(
            spectrum_fermion2D.transform(e, input_basis="real", output_basis="spectral"),
            input_basis="spectral",
            output_basis="real",
        ),
        e,
    )


# def test_unitary_transform(spectrum_fermion2D, eigenfunctions):
#     after_transform = spectrum_fermion2D.transform(eigenfunctions, input_basis="real", output_basis="spectral")

#     # It suffices to test the forward transformation because if the forward
#     # transformation is unitary and the other test ensures that back and forth
#     # transformation is an identity, we can conclude that the inverse
#     # transformation is also unitary.
#     assert np.allclose(
#         spectrum_fermion2D.scalar_product(after_transform, after_transform, input_basis="spectral"),
#         spectrum_fermion2D.scalar_product(eigenfunctions, eigenfunctions),
#     )