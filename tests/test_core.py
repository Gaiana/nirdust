#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   NirDust Project (https://github.com/Gaiana/nirdust)
# Copyright (c) 2020, 2021 Gaia Gaspar, Jose Alacoria
# License: MIT
#   Full Text: https://github.com/Gaiana/nirdust/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================

from unittest.mock import patch

from astropy import units as u
from astropy.modeling import models
from astropy.modeling.models import BlackBody

from matplotlib.testing.decorators import check_figures_equal

from nirdust import core

import numpy as np

import pytest

import specutils as su

# =============================================================================
# TEST METADATA
# =============================================================================


def test_metadata_creation():
    md = core._NDSpectrumMetadata({"alfa": 1})
    assert md["alfa"] == md.alfa == 1
    assert len(md) == 1


def test_metadata_creation_empty():
    md = core._NDSpectrumMetadata({})
    assert len(md) == 0


def test_metadata_key_notfound():
    md = core._NDSpectrumMetadata({"alfa": 1})
    assert md["alfa"] == md.alfa == 1
    with pytest.raises(KeyError):
        md["bravo"]


def test_metadata_attribute_notfound():
    md = core._NDSpectrumMetadata({"alfa": 1})
    assert md["alfa"] == md.alfa == 1
    with pytest.raises(AttributeError):
        md.bravo


def test_metadata_iter():
    md = core._NDSpectrumMetadata({"alfa": 1})
    assert list(iter(md)) == ["alfa"]


def test_metadata_repr():
    md = core._NDSpectrumMetadata({"alfa": 1})
    assert repr(md) == "metadata({'alfa'})"


def test_metadata_dir():
    md = core._NDSpectrumMetadata({"alfa": 1})
    assert "alfa" in dir(md)


def test_metadata_creation_fitst_header_fits_header(header_of):
    header = header_of("external_spectrum_200pc_N4945.fits")
    md = core._NDSpectrumMetadata(header)
    assert md["AMPDAC21"] == md.AMPDAC21 == 180.0
    assert len(md) == len(header)


def test_metadata_key_notfound_fits_header(header_of):
    header = header_of("external_spectrum_200pc_N4945.fits")
    md = core._NDSpectrumMetadata(header)
    assert md["AMPDAC21"] == md.AMPDAC21 == 180.0
    with pytest.raises(KeyError):
        md["bravo"]


def test_metadata_attribute_notfound_fits_header(header_of):
    header = header_of("external_spectrum_200pc_N4945.fits")
    md = core._NDSpectrumMetadata(header)
    assert md["AMPDAC21"] == md.AMPDAC21 == 180.0
    with pytest.raises(AttributeError):
        md.bravo


def test_metadata_iter_fits_header(header_of):
    header = header_of("external_spectrum_200pc_N4945.fits")
    md = core._NDSpectrumMetadata(header)
    assert list(iter(md)) == list(iter(header))


def test_metadata_repr_fits_header(header_of):
    header = header_of("external_spectrum_200pc_N4945.fits")
    md = core._NDSpectrumMetadata(header)
    assert repr(md) == f"metadata({set(header)})"


def test_metadata_dir_fits_header(header_of):
    header = header_of("external_spectrum_200pc_N4945.fits")
    md = core._NDSpectrumMetadata(header)
    for elem in header:
        assert elem in dir(md)


# =============================================================================
# TESTS
# =============================================================================


def test_spectrum_repr(NGC4945_external_continuum_200pc):
    result = repr(NGC4945_external_continuum_200pc)
    expected = "NirdustSpectrum(z=0.00188, spectral_length=1751, spectral_range=[18905.11-25048.53] Angstrom)"  # noqa
    assert result == expected


def test_spectrum_dir(NGC4945_external_continuum_200pc):
    obj1 = NGC4945_external_continuum_200pc
    result = dir(obj1)
    expected = dir(obj1.spec1d_)
    assert not set(expected).difference(result)


def test_match(NGC4945_continuum):
    spectrum = NGC4945_continuum
    assert spectrum.spectral_axis.shape == spectrum.flux.shape


def test_fits_header(NGC4945_continuum):
    spectrum = NGC4945_continuum
    assert isinstance(spectrum.metadata["EXPTIME"], (float, int))
    assert spectrum.metadata["EXPTIME"] >= 0.0
    assert spectrum.metadata["EXPTIME"] == spectrum.metadata.EXPTIME


def test_wav_axis(NGC4945_continuum):
    spectrum = NGC4945_continuum
    assert spectrum.metadata.CRVAL1 >= 0.0
    assert spectrum.metadata.CTYPE1 == "LINEAR"


def test_redshift_correction(NGC4945_continuum):
    spectrum = NGC4945_continuum
    assert (
        spectrum.spectral_axis[0]
        == (spectrum.metadata.CRVAL1 / (1 + 0.00188)) * u.AA
    )


def test_convert_to_frequency(NGC4945_continuum):
    spectrum = NGC4945_continuum
    freq = spectrum.convert_to_frequency().spectral_axis
    np.testing.assert_almost_equal(
        freq.value.mean(), 137313317328585.02, decimal=7
    )


def test_getattr(NGC4945_continuum):
    spectrum = NGC4945_continuum

    # primero tiramos el dir original a spectrum y a spec1d
    original_dir = dir(super(type(spectrum), spectrum))
    spec1d_dir = dir(spectrum.spec1d_)

    # garantizamos que redshift no este oculto por un atributo de spectum
    # y que si este en spectrum.spect1d_
    assert "redshift" not in original_dir and "redshift" in spec1d_dir

    # ahora si probamos el funcionamiento
    np.testing.assert_array_equal(spectrum.spec1d_.redshift, spectrum.redshift)


def test_slice(NGC4945_continuum):
    spectrum = NGC4945_continuum
    result = spectrum[20:40]
    np.testing.assert_array_equal(
        result.spec1d_.flux, spectrum.spec1d_[20:40].flux
    )


def test_len(NGC4945_continuum):
    sp = NGC4945_continuum
    assert len(sp) == len(sp.spectral_axis)


def test_unit():
    a = np.arange(0, 100, 1) * u.Angstrom
    f = np.arange(0, 200, 2) * u.adu

    sp = core.NirdustSpectrum(a, f)

    assert sp.unit == "adu", "Angstrom"


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


# =============================================================================
# DUST_COMPONENT
# =============================================================================


def test_dust_component(NGC4945_continuum, NGC4945_external_continuum_400pc):
    spectrum = NGC4945_continuum.cut_edges(19600, 22900)
    external_spectrum = NGC4945_external_continuum_400pc.cut_edges(
        19600, 22900
    )
    prepared = core.dust_component(spectrum, external_spectrum)
    expected_len = len(spectrum.flux)
    assert len(prepared.flux) == expected_len


def test_dust_component_different_length(
    NGC4945_continuum, NGC4945_external_continuum_400pc
):
    spectrum = NGC4945_continuum.cut_edges(19600, 22900)
    external_spectrum = NGC4945_external_continuum_400pc.cut_edges(
        19600, 22500
    )

    matched_1, matched_2 = core.match_spectral_axes(
        spectrum, external_spectrum
    )

    prepared = core.dust_component(matched_1, matched_2)

    assert len(prepared.flux) == len(prepared.spectral_axis)


def test_dust_component_with_mask(
    NGC4945_nuclear_with_lines, NGC4945_external_with_lines_200pc
):

    nuclear_sp = NGC4945_nuclear_with_lines.cut_edges(20000, 22500)
    external_sp = NGC4945_external_with_lines_200pc.cut_edges(20000, 22500)

    w1 = core.line_spectrum(nuclear_sp, 20800, 21050, 5, window=80)[1]
    w2 = core.line_spectrum(external_sp, 20800, 21050, 5, window=80)[1]

    clean_nuc_sp = nuclear_sp.mask_spectrum(w1)
    clean_ext_sp = external_sp.mask_spectrum(w2)

    match_nuc, match_ext = core.match_spectral_axes(clean_nuc_sp, clean_ext_sp)

    dust = core.dust_component(match_nuc, match_ext)

    assert len(dust.flux) == 544


# =============================================================================
# NORMALIZEDBLACKBODY
# =============================================================================


def test_NormalizedBlackBody_normalization(NGC4945_continuum):
    """Test NBB normalization by the mean"""
    n_black = core.NormalizedBlackBody(1200 * u.K)
    n_inst = n_black(
        NGC4945_continuum.spectral_axis.to_value(
            u.Hz, equivalencies=u.spectral()
        )
    )
    a_blackbody = models.BlackBody(1200 * u.K)
    a_instance = a_blackbody(
        NGC4945_continuum.spectral_axis.to_value(
            u.Hz, equivalencies=u.spectral()
        )
    )
    expected = a_instance / np.mean(a_instance)
    np.testing.assert_almost_equal(n_inst, expected.value, decimal=10)


def test_NormalizedBlackBody_T_proxy():
    """Test consistency of NBB temperature proxy"""
    bb = core.NormalizedBlackBody(1200 * u.K)
    assert bb.T.value == bb.temperature.value
    assert bb.T.unit == bb.temperature.unit


def test_NormalizedBlackBody_initialization_units():
    """Test NBB T unit at instantiation"""
    bb_with_units = core.NormalizedBlackBody(1200 * u.K)
    assert bb_with_units.T.unit is u.K

    bb_with_no_units = core.NormalizedBlackBody(1200)
    assert bb_with_no_units.T.unit is None


def test_NormalizedBlackBody_evaluation_units():
    """Test NBB T and freq units at evaluation"""
    bb_T_with_units = core.NormalizedBlackBody(1200 * u.K)
    freq_with_units = np.arange(1, 10) * u.Hz
    result_with_units = bb_T_with_units(freq_with_units)
    # only Quantity if T is Quantity
    assert isinstance(result_with_units, u.Quantity)
    assert result_with_units.unit.is_unity()

    bb_T_with_no_units = core.NormalizedBlackBody(1200)
    freq_with_no_units = np.arange(1, 10)
    result_with_no_units = bb_T_with_no_units(freq_with_no_units)
    # only Quantity if T is Quantity
    assert not isinstance(result_with_no_units, u.Quantity)
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
    fitted_model, fit_info = core.normalized_blackbody_fitter(
        freq, normalized_flux, T0=900
    )

    np.testing.assert_almost_equal(
        fitted_model.T.value, T_kelvin, decimal=decimal_tolerance
    )


# =============================================================================
# NIRDUST RESULT
# =============================================================================


def test_NirdustResults_temperature():
    nr_inst = core.NirdustResults(
        20 * u.K,
        "Hyperion",
        2356.89,
        "redblackhole",
        freq_axis=None,
        flux_axis=None,
    )
    assert nr_inst.temperature == 20 * u.K


def test_NirdustResults_info():
    nr_inst = core.NirdustResults(
        20 * u.K,
        "Hyperion",
        2356.89,
        "redblackhole",
        freq_axis=None,
        flux_axis=None,
    )
    assert nr_inst.info == "Hyperion"


def test_NirdustResults_uncertainty():
    nr_inst = core.NirdustResults(
        20 * u.K,
        "Hyperion",
        2356.89,
        "redblackhole",
        freq_axis=None,
        flux_axis=None,
    )
    assert nr_inst.uncertainty == 2356.89


def test_NirdustResults_fitted_blackbody():
    nr_inst = core.NirdustResults(
        20 * u.K,
        "Hyperion",
        2356.89,
        "redblackhole",
        freq_axis=None,
        flux_axis=None,
    )
    assert nr_inst.fitted_blackbody == "redblackhole"


def test_NirdustResults_freq_axis(NGC4945_continuum):
    axis = NGC4945_continuum.spectral_axis.to(u.Hz, equivalencies=u.spectral())
    nr_inst = core.NirdustResults(
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
    nr_inst = core.NirdustResults(
        20 * u.K,
        "Hyperion",
        2356.89,
        "redblackhole",
        freq_axis=None,
        flux_axis=fluxx,
    )
    assert len(nr_inst.flux_axis) == len(fluxx)


# =============================================================================
# FIT BLACK BODY
# =============================================================================


def test_fit_blackbody(NGC4945_continuum_rest_frame):
    real_spectrum = NGC4945_continuum_rest_frame
    freq_axis = real_spectrum.spectral_axis.to(
        u.Hz, equivalencies=u.spectral()
    )
    sinthetic_model = BlackBody(1000 * u.K)
    sinthetic_flux = sinthetic_model(freq_axis)

    dispersion = 3.51714285129581
    first_wave = 18940.578099674

    spectrum_length = len(real_spectrum.flux)
    spectral_axis = (
        first_wave + dispersion * np.arange(0, spectrum_length)
    ) * u.AA

    snth_blackbody = core.NirdustSpectrum(
        flux=sinthetic_flux,
        spectral_axis=spectral_axis,
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


# PLOT


@check_figures_equal()
def test_nplot(fig_test, fig_ref, NGC4945_continuum):

    spectrum = NGC4945_continuum.cut_edges(19500, 22900).normalize()

    freq_axis = spectrum.frequency_axis
    flux = spectrum.flux

    stella = core.NormalizedBlackBody(1100)
    instanstella = stella(freq_axis.value)

    fit_results = core.NirdustResults(
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


@check_figures_equal()
def test_nplot_default_axis(fig_test, fig_ref, NGC4945_continuum):
    spectrum = NGC4945_continuum.cut_edges(19500, 22900).normalize()

    freq_axis = spectrum.frequency_axis
    flux = spectrum.flux

    stella = core.NormalizedBlackBody(1100)
    instanstella = stella(freq_axis.value)

    fit_results = core.NirdustResults(
        1100, "Claire Dunphy", 71, stella, freq_axis, flux
    )

    ax_test = fig_test.subplots()
    with patch("matplotlib.pyplot.gca", return_value=ax_test):
        fit_results.nplot()

    ax_ref = fig_ref.subplots()

    ax_ref.plot(freq_axis, flux, color="firebrick", label="continuum")
    ax_ref.plot(freq_axis, instanstella, color="navy", label="model")
    ax_ref.set_xlabel("Frequency [Hz]")
    ax_ref.set_ylabel("Normalized Energy [arbitrary units]")
    ax_ref.legend()


# =============================================================================
# LINE SPECTRUM
# =============================================================================


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

    snth_line_spectrum = core.NirdustSpectrum(
        flux=y_tot,
        spectral_axis=sp_axis,
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

    positions = core.line_spectrum(
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

    snth_line_spectrum = core.NirdustSpectrum(
        flux=y_tot,
        spectral_axis=sp_axis,
        z=0,
    )

    positions = core.line_spectrum(
        snth_line_spectrum, 23000, 24000, 5, window=80
    )[1]

    assert len(positions[0]) == 2


def test_spectral_dispersion(NGC4945_continuum_rest_frame):

    sp = NGC4945_continuum_rest_frame

    dispersion = sp.spectral_dispersion.value
    expected = sp.metadata.CD1_1

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


# ===============================================================================
# MATCH SPECTRAL AXES
# =============================================================================


def test_spectrum_resampling_downscale():

    rng = np.random.default_rng(75)

    g1 = models.Gaussian1D(0.6, 21200, 10)
    g2 = models.Gaussian1D(-0.3, 22000, 15)

    axis = np.arange(1500, 2500, 1) * u.Angstrom

    y = g1(axis.value) + g2(axis.value) + rng.normal(0.0, 0.03, axis.shape)
    y_tot = (y + 0.0001 * axis.value + 1000) * u.adu

    low_disp_sp = core.NirdustSpectrum(
        flux=y_tot,
        spectral_axis=axis,
        z=0,
    )

    # same as axis but half the points, hence twice the dispersion
    new_axis = np.arange(1500, 2500, 2) * u.Angstrom
    new_flux = np.ones(len(new_axis)) * u.adu

    high_disp_sp = core.NirdustSpectrum(
        flux=new_flux,
        spectral_axis=new_axis,
        z=0,
    )

    # check without cleaning nan values
    f_sp, s_sp = core.match_spectral_axes(
        low_disp_sp, high_disp_sp, scaling="downscale", clean=False
    )

    assert len(f_sp.flux) == len(s_sp.flux)
    assert len(s_sp.flux) == 500

    # check cleaning nan values.
    # we know only 1 nan occurs for these spectrums
    f_sp, s_sp = core.match_spectral_axes(
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

    low_disp_sp = core.NirdustSpectrum(
        flux=y_tot,
        spectral_axis=axis,
        z=0,
    )

    # same as axis but half the points, hence twice the dispersion
    new_axis = np.arange(1500, 2500, 2) * u.Angstrom
    new_flux = np.ones(len(new_axis)) * u.adu

    high_disp_sp = core.NirdustSpectrum(
        flux=new_flux,
        spectral_axis=new_axis,
        z=0,
    )

    # check without cleaning nan values
    f_sp, s_sp = core.match_spectral_axes(
        low_disp_sp, high_disp_sp, scaling="upscale", clean=False
    )

    assert len(f_sp.flux) == len(s_sp.flux)
    assert len(s_sp.flux) == 1000

    # check cleaning nan values.
    # we know only 1 nan occurs for these spectrums
    f_sp, s_sp = core.match_spectral_axes(
        low_disp_sp, high_disp_sp, scaling="upscale", clean=True
    )

    assert len(f_sp.flux) == len(s_sp.flux)
    assert len(s_sp.flux) == 999


def test_spectrum_resampling_invalid_scaling():
    with pytest.raises(ValueError):
        core.match_spectral_axes(
            None,
            None,
            scaling="equal",
        )


@pytest.mark.parametrize("true_temp", [500.0, 1000.0, 5000.0])
@pytest.mark.parametrize("scaling", ["downscale", "upscale"])
def test_fit_blackbody_with_resampling(
    NGC4945_continuum_rest_frame,
    NGC3998_sp_lower_resolution,
    true_temp,
    scaling,
):
    real_spectrum = NGC4945_continuum_rest_frame
    freq_axis = real_spectrum.frequency_axis
    sinthetic_model = BlackBody(true_temp * u.K)
    sinthetic_flux = sinthetic_model(freq_axis)

    # the core.NirdustSpectrum object is instantiated
    snth_blackbody = core.NirdustSpectrum(
        flux=sinthetic_flux,
        spectral_axis=real_spectrum.spectral_axis,
        z=0,
    )

    # resampling
    f_sp, s_sp = core.match_spectral_axes(
        snth_blackbody, NGC3998_sp_lower_resolution, scaling=scaling
    )
    snth_bb_temp = (
        f_sp.normalize()
        .convert_to_frequency()
        .fit_blackbody(2000.0)
        .temperature
    )
    np.testing.assert_almost_equal(snth_bb_temp.value, true_temp, decimal=1)


@pytest.mark.parametrize("true_temp", [500.0, 1000.0, 5000.0])
@pytest.mark.parametrize("scaling", ["downscale", "upscale"])
def test_fit_blackbody_with_resampling_in_inverse_order(
    NGC4945_continuum_rest_frame,
    NGC3998_sp_lower_resolution,
    true_temp,
    scaling,
):
    real_spectrum = NGC4945_continuum_rest_frame
    freq_axis = real_spectrum.frequency_axis
    sinthetic_model = BlackBody(true_temp * u.K)
    sinthetic_flux = sinthetic_model(freq_axis)

    # the core.NirdustSpectrum object is instantiated
    snth_blackbody = core.NirdustSpectrum(
        flux=sinthetic_flux,
        spectral_axis=real_spectrum.spectral_axis,
        z=0,
    )

    # resampling but inverting the input order than prevoius test
    f_sp, s_sp = core.match_spectral_axes(
        NGC3998_sp_lower_resolution, snth_blackbody, scaling=scaling
    )
    snth_bb_temp = (
        s_sp.normalize()
        .convert_to_frequency()
        .fit_blackbody(2000.0)
        .temperature
    )
    np.testing.assert_almost_equal(snth_bb_temp.value, true_temp, decimal=1)


def test_match_spectral_axes_first_if(NGC4945_continuum_rest_frame):
    # tests the case where the first spectrum is the largest (no resampling)

    first_sp = NGC4945_continuum_rest_frame
    second_sp = NGC4945_continuum_rest_frame.cut_edges(22000, 23000)

    new_first_sp, new_second_sp = core.match_spectral_axes(first_sp, second_sp)

    assert new_first_sp.spectral_length == new_second_sp.spectral_length


def test_match_spectral_axes_second_if(NGC4945_continuum_rest_frame):
    # tests the case where the second spectrum is the largest (no resampling)

    first_sp = NGC4945_continuum_rest_frame.cut_edges(22000, 23000)
    second_sp = NGC4945_continuum_rest_frame

    new_first_sp, new_second_sp = core.match_spectral_axes(first_sp, second_sp)

    assert new_first_sp.spectral_length == new_second_sp.spectral_length