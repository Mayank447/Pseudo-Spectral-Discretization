from pseudospectral import Derivative1D
import numpy as np
import pytest

TEST_DATA = [
    (Derivative1D, {"num_lattice_points": 3}),
    (Derivative1D, {"num_lattice_points": 11}),
    (Derivative1D, {"num_lattice_points": 101}),
]


@pytest.mark.parametrize(["Spectrum", "spectral_config"], TEST_DATA)
def test_orthonormality(Spectrum, spectral_config):
    spectrum = Spectrum(**spectral_config)
    eigenfunctions = spectrum.eigenfunction(
        np.arange(spectral_config["num_lattice_points"]).reshape(-1, 1)
    )(spectrum.lattice(output_space="real"))
    assert np.allclose(
        spectrum.scalar_product(eigenfunctions, eigenfunctions),
        np.eye(spectral_config["num_lattice_points"]),
    )


@pytest.mark.parametrize(["Spectrum", "spectral_config"], TEST_DATA)
def test_back_and_forth_transform_is_identity(Spectrum, spectral_config):
    spectrum = Spectrum(**spectral_config)
    eigenfunctions = spectrum.eigenfunction(
        np.arange(spectral_config["num_lattice_points"]).reshape(-1, 1)
    )(spectrum.lattice(output_space="real"))
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
