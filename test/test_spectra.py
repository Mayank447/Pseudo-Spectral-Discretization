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
    )(spectrum.lattice(output_space="real"))


def test_orthonormality(spectrum, eigenfunctions):
    assert np.allclose(
        spectrum.scalar_product(eigenfunctions, eigenfunctions),
        np.eye(*eigenfunctions.shape),
    )


def test_back_and_forth_transform_is_identity(spectrum, eigenfunctions):
    assert np.allclose(
        spectrum.transform(
            spectrum.transform(
                eigenfunctions, input_space="real", output_space="spectral"
            ),
            input_space="spectral",
            output_space="real",
        ),
        eigenfunctions,
    )


def test_unitary_transform(spectrum, eigenfunctions):
    after_transform = spectrum.transform(
        eigenfunctions, input_space="real", output_space="spectral"
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
