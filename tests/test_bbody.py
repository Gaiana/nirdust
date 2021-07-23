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

from nirdust import bbody, core, preprocessing

import numpy as np

import pytest


# =============================================================================
# NIRDUST FITTER CLASS
# =============================================================================


def test_fitter_snth_data(NGC4945_continuum):

    spectrum = NGC4945_continuum.cut_edges(19500, 22900)

    # BlackBody model
    true_T = 1233 * u.K
    true_scale = 23.0
    model = models.BlackBody(true_T, scale=true_scale)
    bb = model(spectrum.frequency_axis).value

    # Linear model
    def tp_line(x, x1, x2, y1, y2):
        return (y2 - y1) / (x2 - x1) * (x - x1) + y1

    wave = spectrum.spectral_axis.value
    delta_bb = bb[-1] - bb[0]
    y1_line, y2_line = bb[0] + 2 / 3 * delta_bb, bb[0] + 1 / 3 * delta_bb
    line = tp_line(wave, wave[0], wave[-1], y1_line, y2_line)

    # Total model
    flux = line * u.adu + bb * u.adu

    spectrumT = core.NirdustSpectrum(
        flux=flux, spectral_axis=spectrum.spectral_axis
    )
    externalT = core.NirdustSpectrum(
        flux=33 * line * u.adu, spectral_axis=spectrum.spectral_axis
    )

    fitter = bbody.NirdustFitter(spectrumT, externalT)
    fitter.fit()

    expected_temp = fitter.result(400).temperature.mean
    expected_scale = fitter.result(400).scale.mean

    np.testing.assert_almost_equal(expected_temp.value, 1233.0, decimal=10)
    np.testing.assert_almost_equal(expected_scale, 23.0, decimal=10)
    assert expected_temp.unit == true_T.unit


def test_fit_error(NGC4945_continuum):

    spectrum = NGC4945_continuum.cut_edges(19500, 22900)

    fitter = bbody.NirdustFitter(spectrum, spectrum)

    fitter.fit(steps=10)

    with pytest.raises(RuntimeError):
        fitter.fit(steps=10)


def test_fit_error_2(NGC4945_continuum):

    spectrum = NGC4945_continuum.cut_edges(19500, 22900)

    fitter = bbody.NirdustFitter(spectrum, spectrum)

    initial_st = [1000, 10, 9]

    with pytest.raises(ValueError):
        fitter.fit(initial_state=initial_st)


# =============================================================================
# BLACKBODY RESULT
# =============================================================================


def test_NirdustResults_temperature():
    nr_inst = bbody.NirdustResults(
        20 * u.K, 25, fitted_blackbody=None, dust=None
    )
    assert nr_inst.temperature == 20 * u.K


def test_NirdustResults_info():
    nr_inst = bbody.NirdustResults(
        20 * u.K, 25, fitted_blackbody=None, dust=None
    )
    assert nr_inst.scale == 25


def test_NirdustResults_uncertainty():
    bb = BlackBody(1000 * u.K)

    nr_inst = bbody.NirdustResults(
        20 * u.K, 25, fitted_blackbody=bb, dust=None
    )
    assert nr_inst.fitted_blackbody == bb


def test_NirdustResults_flux_axis(NGC4945_continuum):
    spectrum = NGC4945_continuum
    nr_inst = bbody.NirdustResults(
        20 * u.K, 25, fitted_blackbody=None, dust=spectrum
    )
    assert spectrum == nr_inst.dust


# =============================================================================
# FIT BLACK BODY
# =============================================================================


@pytest.mark.xfail
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
    f_sp, s_sp = preprocessing.match_spectral_axes(
        snth_blackbody, NGC3998_sp_lower_resolution, scaling=scaling
    )
    snth_bb_temp = bbody.fit_blackbody(
        f_sp.normalize().convert_to_frequency(), 2000.0
    ).temperature

    np.testing.assert_almost_equal(snth_bb_temp.value, true_temp, decimal=1)


@pytest.mark.xfail
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
    _, s_sp = preprocessing.match_spectral_axes(
        NGC3998_sp_lower_resolution, snth_blackbody, scaling=scaling
    )
    snth_bb_temp = bbody.fit_blackbody(
        s_sp.normalize().convert_to_frequency(), 2000.0
    ).temperature

    np.testing.assert_almost_equal(snth_bb_temp.value, true_temp, decimal=1)


# =============================================================================
# RESULT PLOTS
# =============================================================================


@check_figures_equal()
def test_nplot(fig_test, fig_ref, NGC4945_continuum):

    spectrum = NGC4945_continuum.cut_edges(19500, 22900).normalize()

    sp_axis = spectrum.spectral_axis
    flux = spectrum.flux

    stella = BlackBody(1100 * u.K)
    instanstella = stella(sp_axis)

    fit_results = bbody.NirdustResults(
        1100, 25, fitted_blackbody=stella, dust=spectrum
    )

    ax_test = fig_test.subplots()
    fit_results.nplot(ax=ax_test)

    ax_ref = fig_ref.subplots()

    ax_ref.plot(sp_axis, flux, color="firebrick", label="continuum")
    ax_ref.plot(sp_axis, instanstella, color="navy", label="model")
    ax_ref.set_xlabel("Angstrom [A]")
    ax_ref.set_ylabel("Normalized Energy [arbitrary units]")
    ax_ref.legend()


@check_figures_equal()
def test_nplot_default_axis(fig_test, fig_ref, NGC4945_continuum):
    spectrum = NGC4945_continuum.cut_edges(19500, 22900).normalize()

    sp_axis = spectrum.spectral_axis
    flux = spectrum.flux

    stella = BlackBody(1100 * u.K)
    instanstella = stella(sp_axis)

    fit_results = bbody.NirdustResults(
        1100, 25, fitted_blackbody=stella, dust=spectrum
    )

    ax_test = fig_test.subplots()
    with patch("matplotlib.pyplot.gca", return_value=ax_test):
        fit_results.nplot()

    ax_ref = fig_ref.subplots()

    ax_ref.plot(sp_axis, flux, color="firebrick", label="continuum")
    ax_ref.plot(sp_axis, instanstella, color="navy", label="model")
    ax_ref.set_xlabel("Angstrom [A]")
    ax_ref.set_ylabel("Normalized Energy [arbitrary units]")
    ax_ref.legend()
