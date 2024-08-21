import numpy as np
import pytest

## Some Fixtures like Spectrum, arbitrary_index_single_eigenfunction, arbitrary_single_coefficient, arbitrary_index_multiple_eigenfunctions
## are defined in the conftest.py file.and imported in all the test files automatically.


@pytest.fixture()
def eigenfunctions(spectrum, sample_points):
    return spectrum.eigenfunction(np.arange(spectrum.total_num_lattice_points).reshape(-1, 1))(sample_points)


def test_orthonormality(spectrum, eigenfunctions):
    assert np.allclose(
        spectrum.scalar_product(eigenfunctions, eigenfunctions),
        np.eye(*eigenfunctions.shape),
    )


def test_back_and_forth_transform_is_identity(spectrum, eigenfunctions):
    assert np.allclose(
        spectrum.transform(
            spectrum.transform(eigenfunctions, input_basis="real", output_basis="spectral"),
            input_basis="spectral",
            output_basis="real",
        ),
        eigenfunctions,
    )


def test_unitary_transform(spectrum, eigenfunctions):
    after_transform = spectrum.transform(eigenfunctions, 
                                         input_basis="real", 
                                         output_basis="spectral"
                                        )

    # It suffices to test the forward transformation because if the forward
    # transformation is unitary and the other test ensures that back and forth
    # transformation is an identity, we can conclude that the inverse
    # transformation is also unitary.
    assert np.allclose(
        spectrum.scalar_product(after_transform, after_transform, input_basis="spectral"),
        spectrum.scalar_product(eigenfunctions, eigenfunctions),
    )
