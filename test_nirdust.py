# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import pathlib

from astropy import units as u
from astropy.modeling import models
from astropy.modeling.models import BlackBody

import nirdust as nd
from nirdust import NirdustSpectrum

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
def NGC4945_continuum_rest_frame():
    file_name = TEST_PATH / "cont03.fits"
    spect = nd.read_spectrum(file_name, 0, 0)
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
    freq = spectrum.convert_to_frequency().frequency_axis
    np.testing.assert_almost_equal(
        freq.value.mean(), 137313317328585.02, decimal=7
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


def test_nomrmalize(NGC4945_continuum):
    spectrum = NGC4945_continuum
    normalized_spectrum = spectrum.normalize()
    mean = np.mean(normalized_spectrum.spec1d.flux)
    assert mean == 1.0


def test_sp_correction(NGC4945_continuum, NGC4945_external_continuum_400pc):
    spectrum = NGC4945_continuum.cut_edges(19600, 22900)
    external_spectrum = NGC4945_external_continuum_400pc.cut_edges(
        19600, 22900
    )
    prepared = nd.sp_correction(spectrum, external_spectrum)
    expected_len = len(spectrum.flux)
    assert len(prepared.flux) == expected_len


def test_sp_correction_second_if(
    NGC4945_continuum, NGC4945_external_continuum_200pc
):
    spectrum = nd.read_spectrum(
        TEST_PATH / "cont01.fits", 0, 0.00188
    ).cut_edges(19600, 22900)
    external_spectrum = NGC4945_external_continuum_200pc.cut_edges(
        19600, 22900
    )
    prepared = nd.sp_correction(spectrum, external_spectrum)
    expected_len = len(spectrum.flux)
    assert len(prepared.flux) == expected_len


def test_sp_correction_third_if(
    NGC4945_continuum, NGC4945_external_continuum_200pc
):
    spectrum = NGC4945_external_continuum_200pc.cut_edges(19600, 22900)
    external_spectrum = nd.read_spectrum(
        TEST_PATH / "cont01.fits", 0, 0.00188
    ).cut_edges(19600, 22900)
    prepared = nd.sp_correction(spectrum, external_spectrum)
    expected_len = len(spectrum.spectral_axis)
    assert len(prepared.spectral_axis) == expected_len


def test_normalized_bb(NGC4945_continuum):
    n_black = nd.normalized_blackbody(T=1200)
    n_inst = n_black(NGC4945_continuum.frequency_axis.value)
    a_blackbody = models.BlackBody(1200 * u.K)
    a_instance = a_blackbody(NGC4945_continuum.frequency_axis)
    expected = a_instance / np.mean(a_instance)
    np.testing.assert_almost_equal(n_inst[200], expected[200].value, decimal=7)


def test_fit_temperature(NGC4945_continuum_rest_frame):
    real_spectrum = NGC4945_continuum_rest_frame
    freq_axis = real_spectrum.convert_to_frequency().frequency_axis
    sinthetic_model = BlackBody(1000 * u.K)
    sinthetic_flux = sinthetic_model(freq_axis)

    dispersion = 3.51714285129581
    first_wave = 18940.578099674
    dispersion_type = "LINEAR  "

    spectrum_length = len(real_spectrum.flux)
    spectral_axis = (
        first_wave + dispersion * np.arange(0, spectrum_length)
    ) * u.AA
    spec1d = su.Spectrum1D(flux=sinthetic_flux, spectral_axis=spectral_axis)
    frequency_axis = spec1d.spectral_axis.to(u.Hz)

    snth_blackbody = NirdustSpectrum(
        header=None,
        z=0,
        spectrum_length=spectrum_length,
        dispersion_key=None,
        first_wavelength=None,
        dispersion_type=dispersion_type,
        spec1d=spec1d,
        frequency_axis=frequency_axis,
    )

    snth_bb_temp = (
        snth_blackbody.normalize()
        .convert_to_frequency()
        .fit_temperature(100)[0]
        .T.value
    )
    np.testing.assert_almost_equal(snth_bb_temp, 1000, decimal=7)
