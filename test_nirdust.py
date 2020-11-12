# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import pathlib

from astropy import units as u

import nirdust as nd

import numpy as np

import pytest

import specutils as su

# ==============================================================================
# CONSTANTS
# ==============================================================================

PATH = os.path.abspath(os.path.dirname(__file__))

TEST_PATH = pathlib.Path(PATH) / "test_data"

# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture(scope="session")
def NGC4945_continuum():
    file_name = TEST_PATH / "cont03.fits"
    spect = nd.read_spectrum(file_name, 0, 0.00188)
    return spect


@pytest.fixture(scope="session")
def NGC4945_external_continuum_400pc():
    file_name = TEST_PATH / "external_spectrum_400pc_N4945.fits"
    spect = nd.read_spectrum(file_name, 0, 0.00188)
    return spect


@pytest.fixture(scope="session")
def NGC4945_external_continuum_200pc():
    file_name = TEST_PATH / "external_spectrum_200pc_N4945.fits"
    spect = nd.read_spectrum(file_name, 0, 0.00188)
    return spect


# ==============================================================================
# TESTS
# ==============================================================================


def test_match(NGC4945_continuum):
    spectrum = NGC4945_continuum
    assert spectrum.spectral_axis.shape == spectrum.flux.shape


def test_header(NGC4945_continuum):
    spectrum = NGC4945_continuum
    assert isinstance(spectrum.header["EXPTIME"], (float, int))
    assert spectrum.header["EXPTIME"] >= 0.0


def test_wav_axis(NGC4945_continuum):
    spectrum = NGC4945_continuum
    assert spectrum.header["CRVAL1"] >= 0.0
    assert spectrum.header["CTYPE1"] == "LINEAR"


def test_calibration():
    with pytest.raises(ValueError):
        path = TEST_PATH / "galaxia_sin_calibrar.fits"
        nd.read_spectrum(path, 0, 0)


def test_redshift_correction(NGC4945_continuum):
    spectrum = NGC4945_continuum
    assert (
        spectrum.spectral_axis[0]
        == (spectrum.header["CRVAL1"] / (1 + 0.00188)) * u.AA
    )


def test_convert_to_frequency(NGC4945_continuum):
    spectrum = NGC4945_continuum
    freq = spectrum._convert_to_frequency().frequency_axis
    np.testing.assert_almost_equal(
        freq.value.mean(), 137313.317328585, decimal=7
    )


def test_getattr(NGC4945_continuum):
    spectrum = NGC4945_continuum
    np.testing.assert_array_equal(spectrum.spec1d.flux, spectrum.flux)


def test_slice(NGC4945_continuum):
    spectrum = NGC4945_continuum
    result = spectrum[20:40]
    np.testing.assert_array_equal(
        result.spec1d.flux, spectrum.spec1d[20:40].flux
    )


def test_cut_edges(NGC4945_continuum):
    spectrum = NGC4945_continuum
    region = su.SpectralRegion(20000 * u.AA, 23000 * u.AA)
    expected = su.manipulation.extract_region(spectrum.spec1d, region)
    result = spectrum.cut_edges(20000, 23000)
    np.testing.assert_array_equal(result.spec1d.flux, expected.flux)


def test_nomrmalization(NGC4945_continuum):
    spectrum = NGC4945_continuum
    normalized_spectrum = spectrum._normalization()
    mean = np.mean(normalized_spectrum.spec1d.flux)
    assert mean == 1.0


def test_nirdustprepare(NGC4945_continuum, NGC4945_external_continuum_400pc):
    spectrum = NGC4945_continuum
    external_spectrum = NGC4945_external_continuum_400pc
    prepared = nd.Nirdustprepare(spectrum, external_spectrum, 19600, 22900)
    cutted_spectrum = spectrum.cut_edges(19600, 22900)
    expected_len = len(cutted_spectrum.flux)
    assert len(prepared.flux) == expected_len


def test_nirdustprepare_second_if(
    NGC4945_continuum, NGC4945_external_continuum_200pc
):
    spectrum = nd.read_spectrum(TEST_PATH / "cont01.fits", 0, 0.00188)
    external_spectrum = NGC4945_external_continuum_200pc
    prepared = nd.Nirdustprepare(spectrum, external_spectrum, 19600, 22900)
    cutted_spectrum = spectrum.cut_edges(19600, 22900)
    expected_len = len(cutted_spectrum.flux)
    assert len(prepared.flux) == expected_len


def test_nirdustprepare_third_if(
    NGC4945_continuum, NGC4945_external_continuum_200pc
):
    spectrum = NGC4945_external_continuum_200pc
    external_spectrum = nd.read_spectrum(TEST_PATH / "cont01.fits", 0, 0.00188)
    prepared = nd.Nirdustprepare(spectrum, external_spectrum, 19600, 22900)
    dif = len(spectrum.spectral_axis) - len(external_spectrum.spectral_axis)
    cutted_spectrum = spectrum.cut_edges(19600, 22900)
    expected_len = len(cutted_spectrum.flux) - dif
    assert len(prepared.spectral_axis) == expected_len
