# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import pathlib
from unittest.mock import patch

from astropy import units as u
from astropy.io import fits
from astropy.modeling import models
from astropy.modeling.models import BlackBody

from matplotlib.testing.decorators import check_figures_equal

import nirdust as nd
from nirdust import NirdustResults
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
    spect = nd.read_spectrum(file_name, 0, z=0.00188)
    return spect


@pytest.fixture(scope="session")
def NGC4945_continuum_rest_frame():
    file_name = TEST_PATH / "cont03.fits"
    spect = nd.read_spectrum(file_name, 0, z=0)
    return spect


@pytest.fixture(scope="session")
def NGC4945_external_continuum_400pc():
    file_name = TEST_PATH / "external_spectrum_400pc_N4945.fits"
    spect = nd.read_spectrum(file_name, 0, z=0.00188)
    return spect


@pytest.fixture(scope="session")
def NGC4945_external_continuum_200pc():
    file_name = TEST_PATH / "external_spectrum_200pc_N4945.fits"
    spect = nd.read_spectrum(file_name, 0, z=0.00188)
    return spect


@pytest.fixture(scope="session")
def NGC4945_external_with_lines_200pc():
    file_name = TEST_PATH / "External_with_lines.fits"
    spect = nd.read_spectrum(file_name, 0, z=0.00188)
    return spect


@pytest.fixture(scope="session")
def NGC4945_nuclear_with_lines():
    file_name = TEST_PATH / "NuclearNGC4945.fits"
    spect = nd.read_spectrum(file_name, 0, z=0.00188)
    return spect


@pytest.fixture(scope="session")
def NGC3998_sp_lower_resolution():
    file_name = TEST_PATH / "ngc3998-sfin.fits"
    spect = nd.read_spectrum(file_name, 1, z=0.00350)
    return spect


@pytest.fixture(scope="session")
def snth_spectrum_1000(NGC4945_continuum_rest_frame):

    real_spectrum = NGC4945_continuum_rest_frame
    freq_axis = real_spectrum.frequency_axis
    sinthetic_model = nd.NormalizedBlackBody(1000)
    sinthetic_flux = sinthetic_model(freq_axis.value)

    mu, sigma = 0, 0.1
    nd_random = np.random.default_rng(50)
    gaussian_noise = nd_random.normal(mu, sigma, len(freq_axis))
    noisy_model = sinthetic_flux * (1 * u.adu + gaussian_noise * u.adu)

    dispersion = 3.51714285129581
    first_wave = 18940.578099674

    spectrum_length = len(real_spectrum.flux)
    spectral_axis = (
        first_wave + dispersion * np.arange(0, spectrum_length)
    ) * u.AA
    frequency_axis = spectral_axis.to(u.Hz, equivalencies=u.spectral())

    return NirdustSpectrum(
        flux=noisy_model,
        frequency_axis=frequency_axis,
        z=0,
    )


# ==============================================================================
# TESTS
# ==============================================================================


def test_read_spectrum():
    # read with no extension and wrong keyword
    file_name = TEST_PATH / "external_spectrum_200pc_N4945.fits"
    obj1 = nd.read_spectrum(file_name)
    obj2 = nd.read_spectrum(file_name, extension=0)
    np.testing.assert_almost_equal(
        obj1.frequency_axis.value, obj2.frequency_axis.value, decimal=10
    )


def test_spectrum_dir():
    file_name = TEST_PATH / "external_spectrum_200pc_N4945.fits"
    obj1 = nd.read_spectrum(file_name)
    result = repr(obj1)
    expected = "NirdustSpectrum(z=0, spectral_length=1751, spectral_range=[18940.65-25095.62] Angstrom)"  # noqa
    assert result == expected


def test_spectrum_repr():
    file_name = TEST_PATH / "external_spectrum_200pc_N4945.fits"
    obj1 = nd.read_spectrum(file_name)
    result = dir(obj1)
    expected = dir(obj1.spec1d_)
    assert not set(expected).difference(result)


def test_infer_science_extension_MEF_multiple_spectrum():
    # fits with multiple extensions
    file_name = TEST_PATH / "external_spectrum_200pc_N4945.fits"
    with fits.open(file_name) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(data=data.copy(), header=header.copy())
    hdu2 = fits.ImageHDU(data=data.copy(), header=None)
    hdu3 = fits.ImageHDU(data=data.copy(), header=header.copy())
    hdul = fits.HDUList([hdu0, hdu1, hdu2, hdu3])

    ext_candidates = nd.infer_fits_science_extension(hdul)
    assert len(ext_candidates) == 2
    np.testing.assert_array_equal(ext_candidates, np.array([1, 3]))


def test_read_spectrum_MEF_single_spectrum():
    # fits with multiple extensions
    file_name = TEST_PATH / "external_spectrum_200pc_N4945.fits"
    with fits.open(file_name) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    # only one header with relevant keywords
    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(data=None)
    hdu2 = fits.ImageHDU(data=data.copy(), header=header.copy())
    hdu3 = fits.ImageHDU(data=None)
    hdul = fits.HDUList([hdu0, hdu1, hdu2, hdu3])

    with patch("astropy.io.fits.open", return_value=hdul):
        obj = nd.read_spectrum("imaginary_file.fits")
        assert isinstance(obj, nd.NirdustSpectrum)


def test_read_spectrum_MEF_multiple_spectrum():
    # fits with multiple extensions
    file_name = TEST_PATH / "external_spectrum_200pc_N4945.fits"
    with fits.open(file_name) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(data=data.copy(), header=header.copy())
    hdu2 = fits.ImageHDU(data=data.copy(), header=header.copy())
    hdu3 = fits.ImageHDU(data=data.copy(), header=header.copy())
    hdul = fits.HDUList([hdu0, hdu1, hdu2, hdu3])

    with patch("astropy.io.fits.open", return_value=hdul):
        with pytest.raises(nd.HeaderKeywordError):
            # Default is extension=None. this tries to detect
            # the data extension, if there are many the error is raised
            nd.read_spectrum("imaginary_file.fits")

        # If the extension is specified it should work ok
        obj = nd.read_spectrum("imaginary_file.fits", extension=2)
        assert isinstance(obj, nd.NirdustSpectrum)


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
        path = TEST_PATH / "no-calibrated_spectrum.fits"
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
    np.testing.assert_array_equal(spectrum.spec1d_.flux, spectrum.flux)


def test_slice(NGC4945_continuum):
    spectrum = NGC4945_continuum
    result = spectrum[20:40]
    np.testing.assert_array_equal(
        result.spec1d_.flux, spectrum.spec1d_[20:40].flux
    )


def test_cut_edges(NGC4945_continuum):
    spectrum = NGC4945_continuum
    region = su.SpectralRegion(20000 * u.AA, 23000 * u.AA)
    expected = su.manipulation.extract_region(spectrum.spec1d_, region)
    result = spectrum.cut_edges(20000, 23000)
    np.testing.assert_array_equal(result.spec1d_.flux, expected.flux)


def test_nomrmalize(NGC4945_continuum):
    spectrum = NGC4945_continuum
    normalized_spectrum = spectrum.normalize()
    mean = np.mean(normalized_spectrum.spec1d_.flux)
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
        TEST_PATH / "cont01.fits", 0, z=0.00188
    ).cut_edges(19600, 22900)
    external_spectrum = NGC4945_external_continuum_200pc.cut_edges(
        19600, 22900
    )
    prepared = nd.sp_correction(spectrum, external_spectrum)
    expected_len = len(spectrum.spectral_axis)
    assert len(prepared.spectral_axis) == expected_len


def test_sp_correction_third_if(
    NGC4945_continuum, NGC4945_external_continuum_200pc
):
    spectrum = NGC4945_external_continuum_200pc.cut_edges(19600, 22900)
    external_spectrum = nd.read_spectrum(
        TEST_PATH / "cont01.fits", 0, z=0.00188
    ).cut_edges(19600, 22900)
    prepared = nd.sp_correction(spectrum, external_spectrum)
    expected_len = len(external_spectrum.spectral_axis)
    assert len(prepared.spectral_axis) == expected_len


def test_NormalizedBlackBody_normalization(NGC4945_continuum):
    """Test NBB normalization by the mean"""
    n_black = nd.NormalizedBlackBody(1200 * u.K)
    n_inst = n_black(NGC4945_continuum.frequency_axis.value)
    a_blackbody = models.BlackBody(1200 * u.K)
    a_instance = a_blackbody(NGC4945_continuum.frequency_axis)
    expected = a_instance / np.mean(a_instance)
    np.testing.assert_almost_equal(n_inst, expected.value, decimal=10)


def test_NormalizedBlackBody_T_proxy():
    """Test consistency of NBB temperature proxy"""
    bb = nd.NormalizedBlackBody(1200 * u.K)
    assert bb.T.value == bb.temperature.value
    assert bb.T.unit == bb.temperature.unit


def test_NormalizedBlackBody_initialization_units():
    """Test NBB T unit at instantiation"""
    bb_with_units = nd.NormalizedBlackBody(1200 * u.K)
    assert bb_with_units.T.unit is u.K

    bb_with_no_units = nd.NormalizedBlackBody(1200)
    assert bb_with_no_units.T.unit is None


def test_NormalizedBlackBody_evaluation_units():
    """Test NBB T and freq units at evaluation"""
    bb_T_with_units = nd.NormalizedBlackBody(1200 * u.K)
    freq_with_units = np.arange(1, 10) * u.Hz
    result_with_units = bb_T_with_units(freq_with_units)
    assert isinstance(
        result_with_units, u.Quantity
    )  # only Quantity if T is Quantity
    assert result_with_units.unit.is_unity()

    bb_T_with_no_units = nd.NormalizedBlackBody(1200)
    freq_with_no_units = np.arange(1, 10)
    result_with_no_units = bb_T_with_no_units(freq_with_no_units)
    assert not isinstance(
        result_with_no_units, u.Quantity
    )  # only Quantity if T is Quantity
    np.testing.assert_almost_equal(
        result_with_no_units, result_with_units.value, decimal=10
    )


@pytest.mark.parametrize("T_kelvin", [500.0, 1000.0, 5000.0])
@pytest.mark.parametrize("noise_tolerance", [(0.0, 4), (0.1, -1), (0.2, -2)])
def test_normalized_blackbody_fitter(T_kelvin, noise_tolerance):

    # noise_tolerance has two values: noise and tolerance
    noise_level, decimal_tolerance = noise_tolerance

    freq = np.linspace(1e14, 2e14, 10000) * u.Hz
    sinthetic_model = BlackBody(T_kelvin * u.K)
    flux = sinthetic_model(freq)

    mu, sigma = 0, noise_level
    nd_random = np.random.RandomState(42)
    gaussian_noise = nd_random.normal(mu, sigma, len(freq))
    noisy_flux = flux * (1 + gaussian_noise)

    normalized_flux = noisy_flux / np.mean(noisy_flux)
    fitted_model, fit_info = nd.normalized_blackbody_fitter(
        freq, normalized_flux, T0=900
    )

    np.testing.assert_almost_equal(
        fitted_model.T.value, T_kelvin, decimal=decimal_tolerance
    )


def test_NirdustResults_temperature():
    nr_inst = NirdustResults(
        20 * u.K,
        "Hyperion",
        2356.89,
        "redblackhole",
        freq_axis=None,
        flux_axis=None,
    )
    assert nr_inst.temperature == 20 * u.K


def test_NirdustResults_info():
    nr_inst = NirdustResults(
        20 * u.K,
        "Hyperion",
        2356.89,
        "redblackhole",
        freq_axis=None,
        flux_axis=None,
    )
    assert nr_inst.info == "Hyperion"


def test_NirdustResults_uncertainty():
    nr_inst = NirdustResults(
        20 * u.K,
        "Hyperion",
        2356.89,
        "redblackhole",
        freq_axis=None,
        flux_axis=None,
    )
    assert nr_inst.uncertainty == 2356.89


def test_NirdustResults_fitted_blackbody():
    nr_inst = NirdustResults(
        20 * u.K,
        "Hyperion",
        2356.89,
        "redblackhole",
        freq_axis=None,
        flux_axis=None,
    )
    assert nr_inst.fitted_blackbody == "redblackhole"


def test_NirdustResults_freq_axis(NGC4945_continuum):
    axis = NGC4945_continuum.frequency_axis
    nr_inst = NirdustResults(
        20 * u.K,
        "Hyperion",
        2356.89,
        "redblackhole",
        freq_axis=axis,
        flux_axis=None,
    )
    assert len(nr_inst.freq_axis) == len(axis)


def test_NirdustResults_flux_axis(NGC4945_continuum):
    fluxx = NGC4945_continuum.flux
    nr_inst = NirdustResults(
        20 * u.K,
        "Hyperion",
        2356.89,
        "redblackhole",
        freq_axis=None,
        flux_axis=fluxx,
    )
    assert len(nr_inst.flux_axis) == len(fluxx)


def test_fit_blackbody(NGC4945_continuum_rest_frame):
    real_spectrum = NGC4945_continuum_rest_frame
    freq_axis = real_spectrum.frequency_axis
    sinthetic_model = BlackBody(1000 * u.K)
    sinthetic_flux = sinthetic_model(freq_axis)

    dispersion = 3.51714285129581
    first_wave = 18940.578099674

    spectrum_length = len(real_spectrum.flux)
    spectral_axis = (
        first_wave + dispersion * np.arange(0, spectrum_length)
    ) * u.AA
    frequency_axis = spectral_axis.to(u.Hz, equivalencies=u.spectral())

    snth_blackbody = NirdustSpectrum(
        flux=sinthetic_flux,
        frequency_axis=frequency_axis,
        z=0,
    )

    snth_bb_temp = (
        snth_blackbody.normalize()
        .convert_to_frequency()
        .fit_blackbody(1200)
        .temperature
    )
    np.testing.assert_almost_equal(snth_bb_temp.value, 1000, decimal=7)

    # test also if fit_backbody can recieve T with units
    snth_bb_temp = (
        snth_blackbody.normalize()
        .convert_to_frequency()
        .fit_blackbody(100 * u.K)
        .temperature
    )
    np.testing.assert_almost_equal(snth_bb_temp.value, 1000, decimal=7)


@check_figures_equal(extensions=["png"])
def test_nplot(fig_test, fig_ref):

    spectrum = (
        nd.read_spectrum(TEST_PATH / "cont03.fits", 0, z=0.00188)
        .cut_edges(19500, 22900)
        .normalize()
    )

    freq_axis = spectrum.frequency_axis
    flux = spectrum.flux

    stella = nd.NormalizedBlackBody(1100)
    instanstella = stella(freq_axis.value)

    fit_results = NirdustResults(
        1100, "Claire Dunphy", 71, stella, freq_axis, flux
    )

    ax_test = fig_test.subplots()
    fit_results.nplot(ax=ax_test)

    ax_ref = fig_ref.subplots()

    ax_ref.plot(freq_axis, flux, color="firebrick", label="continuum")
    ax_ref.plot(freq_axis, instanstella, color="navy", label="model")
    ax_ref.set_xlabel("Frequency [Hz]")
    ax_ref.set_ylabel("Normalized Energy [arbitrary units]")
    ax_ref.legend()


def test_pix2wavelength():
    file_name = TEST_PATH / "external_spectrum_400pc_N4945.fits"

    with fits.open(file_name) as hdul:
        header = hdul[0].header

    pix_0_wav = header["CRVAL1"]
    pix_disp = header["CD1_1"]

    pix_array = np.arange(50.0)
    z = 0.1

    expected = (pix_0_wav + pix_disp * pix_array) / (1 + z)
    result = nd.pix2wavelength(pix_array, header, z)

    np.testing.assert_almost_equal(result, expected, decimal=10)


def test_line_spectrum(NGC4945_continuum_rest_frame):

    sp_axis = NGC4945_continuum_rest_frame.spectral_axis
    g1 = models.Gaussian1D(0.6, 21200, 10)
    g2 = models.Gaussian1D(-0.3, 22000, 15)

    rng = np.random.default_rng(75)

    y = (
        g1(sp_axis.value)
        + g2(sp_axis.value)
        + rng.normal(0.0, 0.01, sp_axis.shape)
    )
    y_tot = (y + 0.0001 * sp_axis.value + 1000) * u.adu

    snth_line_spectrum = NirdustSpectrum(
        flux=y_tot,
        frequency_axis=sp_axis.to(u.Hz),
        z=0,
    )

    expected_positions = (
        np.array(
            [
                [g1.mean - 3 * g1.stddev, g1.mean + 3 * g1.stddev],
                [g2.mean - 3 * g2.stddev, g2.mean + 3 * g2.stddev],
            ]
        )
        * u.Angstrom
    )

    positions = nd.line_spectrum(
        snth_line_spectrum, 23000, 24000, 5, window=80
    )[1]

    np.testing.assert_almost_equal(
        positions.value * 0.001, expected_positions.value * 0.001, decimal=0
    )


def test_number_of_lines(NGC4945_continuum_rest_frame):

    sp_axis = NGC4945_continuum_rest_frame.spectral_axis
    g1 = models.Gaussian1D(0.6, 21200, 10)
    g2 = models.Gaussian1D(-0.3, 22000, 15)

    rng = np.random.default_rng(75)

    y = (
        g1(sp_axis.value)
        + g2(sp_axis.value)
        + rng.normal(0.0, 0.03, sp_axis.shape)
    )
    y_tot = (y + 0.0001 * sp_axis.value + 1000) * u.adu

    snth_line_spectrum = NirdustSpectrum(
        flux=y_tot,
        frequency_axis=sp_axis.to(u.Hz),
        z=0,
    )

    positions = nd.line_spectrum(
        snth_line_spectrum, 23000, 24000, 5, window=80
    )[1]

    assert len(positions[0]) == 2


def test_spectral_dispersion(NGC4945_continuum_rest_frame):

    sp = NGC4945_continuum_rest_frame

    dispersion = sp.spectral_dispersion.value
    expected = sp.header["CD1_1"]

    np.testing.assert_almost_equal(dispersion, expected, decimal=14)


def test_mask_spectrum_1(NGC4945_continuum_rest_frame):

    spectrum = NGC4945_continuum_rest_frame

    with pytest.raises(ValueError):
        spectrum.mask_spectrum(line_intervals=None, mask=None)


def test_mask_spectrum_2(NGC4945_continuum_rest_frame):

    spectrum = NGC4945_continuum_rest_frame
    line_intervals = ((20000, 20050), (21230, 21280)) * u.Angstrom
    mask = (True, True, False, False)

    with pytest.raises(ValueError):
        spectrum.mask_spectrum(line_intervals, mask)


def test_mask_spectrum_3(NGC4945_continuum_rest_frame):

    spectrum = NGC4945_continuum_rest_frame
    line_intervals = ((20000, 20050), (21230, 21280)) * u.Angstrom

    line_indexes = np.searchsorted(spectrum.spectral_axis, line_intervals)
    mask = np.ones(len(spectrum.flux), dtype=bool)

    for i, j in line_indexes:
        mask[i : j + 1] = False  # noqa

    new_flux = spectrum.flux[mask]

    masked_flux = spectrum.mask_spectrum(line_intervals).flux

    np.testing.assert_array_equal(new_flux, masked_flux, verbose=True)


def test_mask_spectrum_4(NGC4945_continuum_rest_frame):

    spectrum = NGC4945_continuum_rest_frame

    mask = np.ones(spectrum.spectral_length, dtype=bool)
    mask[100] = False

    new_flux = spectrum.flux[mask]

    masked_flux = spectrum.mask_spectrum(mask=mask).flux

    np.testing.assert_array_equal(new_flux, masked_flux, verbose=True)


def test_mask_spectrum_5(NGC4945_continuum_rest_frame):

    spectrum = NGC4945_continuum_rest_frame

    mask = np.ones(1200, dtype=bool)
    mask[100] = False

    with pytest.raises(ValueError):
        spectrum.mask_spectrum(mask=mask)


def test_sp_correction_with_mask(
    NGC4945_nuclear_with_lines, NGC4945_external_with_lines_200pc
):

    nuclear_sp = NGC4945_nuclear_with_lines.cut_edges(20000, 22500)
    external_sp = NGC4945_external_with_lines_200pc.cut_edges(20000, 22500)

    w1 = nd.line_spectrum(nuclear_sp, 20800, 21050, 5, window=80)[1]
    w2 = nd.line_spectrum(external_sp, 20800, 21050, 5, window=80)[1]

    clean_nuc_sp = nuclear_sp.mask_spectrum(w1)
    clean_ext_sp = external_sp.mask_spectrum(w2)

    dust = nd.sp_correction(clean_nuc_sp, clean_ext_sp)

    assert len(dust.flux) == 544


def test_spectrum_resampling_downscale():

    rng = np.random.default_rng(75)

    g1 = models.Gaussian1D(0.6, 21200, 10)
    g2 = models.Gaussian1D(-0.3, 22000, 15)

    axis = np.arange(1500, 2500, 1) * u.Angstrom

    y = g1(axis.value) + g2(axis.value) + rng.normal(0.0, 0.03, axis.shape)
    y_tot = (y + 0.0001 * axis.value + 1000) * u.adu

    low_disp_sp = NirdustSpectrum(
        flux=y_tot,
        frequency_axis=axis.to(u.Hz, equivalencies=u.spectral()),
        z=0,
    )

    # same as axis but half the points, hence twice the dispersion
    new_axis = np.arange(1500, 2500, 2) * u.Angstrom
    new_flux = np.ones(len(new_axis)) * u.adu

    high_disp_sp = NirdustSpectrum(
        flux=new_flux,
        frequency_axis=new_axis.to(u.Hz, equivalencies=u.spectral()),
        z=0,
    )

    # check without cleaning nan values
    f_sp, s_sp = nd.spectrum_resampling(
        low_disp_sp, high_disp_sp, scaling="downscale", clean=False
    )

    assert len(f_sp.flux) == len(s_sp.flux)
    assert len(s_sp.flux) == 500

    # check cleaning nan values.
    # we know only 1 nan occurs for these spectrums
    f_sp, s_sp = nd.spectrum_resampling(
        low_disp_sp, high_disp_sp, scaling="downscale", clean=True
    )

    assert len(f_sp.flux) == len(s_sp.flux)
    assert len(s_sp.flux) == 499


def test_spectrum_resampling_upscale():

    g1 = models.Gaussian1D(0.6, 21200, 10)
    g2 = models.Gaussian1D(-0.3, 22000, 15)

    axis = np.arange(1500, 2500, 1) * u.Angstrom

    rng = np.random.default_rng(75)

    y = g1(axis.value) + g2(axis.value) + rng.normal(0.0, 0.03, axis.shape)
    y_tot = (y + 0.0001 * axis.value + 1000) * u.adu

    low_disp_sp = NirdustSpectrum(
        flux=y_tot,
        frequency_axis=axis.to(u.Hz, equivalencies=u.spectral()),
        z=0,
    )

    # same as axis but half the points, hence twice the dispersion
    new_axis = np.arange(1500, 2500, 2) * u.Angstrom
    new_flux = np.ones(len(new_axis)) * u.adu

    high_disp_sp = NirdustSpectrum(
        flux=new_flux,
        frequency_axis=new_axis.to(u.Hz, equivalencies=u.spectral()),
        z=0,
    )

    # check without cleaning nan values
    f_sp, s_sp = nd.spectrum_resampling(
        low_disp_sp, high_disp_sp, scaling="upscale", clean=False
    )

    assert len(f_sp.flux) == len(s_sp.flux)
    assert len(s_sp.flux) == 1000

    # check cleaning nan values.
    # we know only 1 nan occurs for these spectrums
    f_sp, s_sp = nd.spectrum_resampling(
        low_disp_sp, high_disp_sp, scaling="upscale", clean=True
    )

    assert len(f_sp.flux) == len(s_sp.flux)
    assert len(s_sp.flux) == 999


def test_spectrum_resampling_invalid_scaling():
    with pytest.raises(ValueError):
        nd.spectrum_resampling(None, None, scaling="equal")


@pytest.mark.parametrize("true_temp", [500.0, 1000.0, 5000.0])
def test_fit_blackbody_with_resampling(
    NGC4945_continuum_rest_frame,
    NGC3998_sp_lower_resolution,
    true_temp,
):
    real_spectrum = NGC4945_continuum_rest_frame
    freq_axis = real_spectrum.frequency_axis
    sinthetic_model = BlackBody(true_temp * u.K)
    sinthetic_flux = sinthetic_model(freq_axis)

    # the NirdustSpectrum object is instantiated
    snth_blackbody = NirdustSpectrum(
        flux=sinthetic_flux,
        frequency_axis=freq_axis,
        z=0,
    )

    # resampling
    f_sp, s_sp = nd.spectrum_resampling(
        snth_blackbody, NGC3998_sp_lower_resolution
    )

    snth_bb_temp = (
        f_sp.normalize()
        .convert_to_frequency()
        .fit_blackbody(2000.0)
        .temperature
    )

    np.testing.assert_almost_equal(snth_bb_temp.value, true_temp, decimal=1)
