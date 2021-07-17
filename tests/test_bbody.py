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

from nirdust import bbody, core

import numpy as np

import pytest


# =============================================================================
# TESTS
# =============================================================================


def test_NormalizedBlackBody_normalization(NGC4945_continuum):
    """Test NBB normalization by the mean"""
    n_black = bbody.NormalizedBlackBody(1200 * u.K)
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
    bb = bbody.NormalizedBlackBody(1200 * u.K)
    assert bb.T.value == bb.temperature.value
    assert bb.T.unit == bb.temperature.unit


def test_NormalizedBlackBody_initialization_units():
    """Test NBB T unit at instantiation"""
    bb_with_units = bbody.NormalizedBlackBody(1200 * u.K)
    assert bb_with_units.T.unit is u.K

    bb_with_no_units = bbody.NormalizedBlackBody(1200)
    assert bb_with_no_units.T.unit is None


def test_NormalizedBlackBody_evaluation_units():
    """Test NBB T and freq units at evaluation"""
    bb_T_with_units = bbody.NormalizedBlackBody(1200 * u.K)
    freq_with_units = np.arange(1, 10) * u.Hz
    result_with_units = bb_T_with_units(freq_with_units)
    # only Quantity if T is Quantity
    assert isinstance(result_with_units, u.Quantity)
    assert result_with_units.unit.is_unity()

    bb_T_with_no_units = bbody.NormalizedBlackBody(1200)
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
    fitted_model, fit_info = bbody.normalized_blackbody_fitter(
        freq, normalized_flux, T0=900
    )

    np.testing.assert_almost_equal(
        fitted_model.T.value, T_kelvin, decimal=decimal_tolerance
    )


# =============================================================================
# BLACKBODY RESULT
# =============================================================================


def test_NirdustResults_temperature():
    nr_inst = bbody.NirdustResults(
        20 * u.K,
        "Hyperion",
        2356.89,
        "redblackhole",
        freq_axis=None,
        flux_axis=None,
    )
    assert nr_inst.temperature == 20 * u.K


def test_NirdustResults_info():
    nr_inst = bbody.NirdustResults(
        20 * u.K,
        "Hyperion",
        2356.89,
        "redblackhole",
        freq_axis=None,
        flux_axis=None,
    )
    assert nr_inst.info == "Hyperion"


def test_NirdustResults_uncertainty():
    nr_inst = bbody.NirdustResults(
        20 * u.K,
        "Hyperion",
        2356.89,
        "redblackhole",
        freq_axis=None,
        flux_axis=None,
    )
    assert nr_inst.uncertainty == 2356.89


def test_NirdustResults_fitted_blackbody():
    nr_inst = bbody.NirdustResults(
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
    nr_inst = bbody.NirdustResults(
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
    nr_inst = bbody.NirdustResults(
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

    snth_bb_temp = bbody.fit_blackbody(
        snth_blackbody.normalize().convert_to_frequency(), 1200
    ).temperature
    np.testing.assert_almost_equal(snth_bb_temp.value, 1000, decimal=7)

    # test also if fit_backbody can recieve T with units
    snth_bb_temp = bbody.fit_blackbody(
        snth_blackbody.normalize().convert_to_frequency(), 100 * u.K
    ).temperature

    np.testing.assert_almost_equal(snth_bb_temp.value, 1000, decimal=7)


# =============================================================================
# RESULT PLOTS
# =============================================================================


@check_figures_equal()
def test_nplot(fig_test, fig_ref, NGC4945_continuum):

    spectrum = NGC4945_continuum.cut_edges(19500, 22900).normalize()

    freq_axis = spectrum.frequency_axis
    flux = spectrum.flux

    stella = bbody.NormalizedBlackBody(1100)
    instanstella = stella(freq_axis.value)

    fit_results = bbody.NirdustResults(
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

    stella = bbody.NormalizedBlackBody(1100)
    instanstella = stella(freq_axis.value)

    fit_results = bbody.NirdustResults(
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
