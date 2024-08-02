from pseudospectral import Derivative1D
import numpy as np
import pytest

SPECTRA = [
    {"type": Derivative1D, "config": {"num_lattice_points": 3}},
    {"type": Derivative1D, "config": {"num_lattice_points": 101}},
    {"type": Derivative1D, "config": {"num_lattice_points": 3, "L": 3}},
    {"type": Derivative1D, "config": {"num_lattice_points": 101, "L": 42}},
]


@pytest.fixture(params=SPECTRA)
def spectrum(request):
    return request.param["type"](**request.param["config"])


@pytest.fixture()
def eigenfunctions(spectrum):
    return spectrum.eigenfunction(
        np.arange(spectrum.num_lattice_points).reshape(-1, 1)
    )(spectrum.lattice(output_basis="real"))


def test_orthonormality(spectrum, eigenfunctions):
    """
    Function to test the orthonormality of the eigenfunctions of the 1D Derivative operator.
    """
    print(eigenfunctions, '\n')
    assert np.allclose(
        spectrum.scalar_product(eigenfunctions, eigenfunctions),
        np.eye(*eigenfunctions.shape),
    )


def test_back_and_forth_transform_is_identity(spectrum, eigenfunctions):
    assert np.allclose(
        spectrum.transform(
            spectrum.transform(
                eigenfunctions, input_basis="real", output_basis="spectral"
            ),
            input_basis="spectral",
            output_basis="real",
        ),
        eigenfunctions,
    )


def test_unitary_transform(spectrum, eigenfunctions):
    after_transform = spectrum.transform(
        eigenfunctions, input_basis="real", output_basis="spectral"
    )

    # It suffices to test the forward transformation because if the forward
    # transformation is unitary and the other test ensures that back and forth
    # transformation is an identity, we can conclude that the inverse
    # transformation is also unitary.
    assert np.allclose(
        spectrum.scalar_product(
            after_transform, after_transform, input_basis="spectral"
        ),
        spectrum.scalar_product(eigenfunctions, eigenfunctions),
    )

if __name__ == "__main__":
    spectra = Derivative1D(num_lattice_points=3, L=1)
    print(spectra.eigenvalues)
    ef = spectra.eigenfunction(
        np.arange(spectra.num_lattice_points).reshape(-1, 1)
    )(spectra.lattice(output_basis="real"))
    print(ef)
    # test_orthonormality(spectra, eigenfunctions)
