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
    fitter.fit(steps=500)

    expected_temp = fitter.result(400).temperature.mean
    expected_scale = fitter.result(400).scale.mean

    np.testing.assert_almost_equal(expected_temp.value, 1233.0, decimal=6)
    np.testing.assert_almost_equal(expected_scale, 23.0, decimal=6)
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


@check_figures_equal()
def test_plot(fig_test, fig_ref, NGC4945_continuum):

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
    fitter.fit(steps=20)

    # test figure is generated
    ax_test = fig_test.subplots(2, 1, sharex=True)
    fitter.plot(ax=ax_test)

    # ref figure is constructed

    ax_ref = fig_ref.subplots(2, 1, sharex=True)

    ax_t, ax_log = ax_ref
    fig = ax_t.get_figure()
    fig.subplots_adjust(hspace=0)

    chain = fitter.chain(discard=0)
    arr_t = chain[:, :, 0]
    mean_t = arr_t.mean(axis=1)

    arr_log = chain[:, :, 1]
    mean_log = arr_log.mean(axis=1)

    # plot
    ax_t.set_title(
        f"Sampled parameters\n Steps={fitter.steps_} - Discarded={0}"
    )

    ax_t.plot(arr_t, alpha=0.5)
    ax_t.plot(mean_t, color="k", label="Mean")
    ax_t.set_ylabel("T")

    ax_log.plot(arr_log, alpha=0.5)
    ax_log.plot(mean_log, color="k", label="Mean")
    ax_log.set_ylabel("log(scale)")
    ax_log.set_xlabel("Steps")
    ax_log.legend()


@check_figures_equal()
def test_plot_non_axis(fig_test, fig_ref, NGC4945_continuum):

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
    fitter.fit(steps=20)

    # test figure is generated
    ax_test = fig_test.subplots(2, 1, sharex=True)
    with patch("matplotlib.pyplot.subplots", return_value=(fig_test, ax_test)):
        fitter.plot()

    # ref figure is constructed

    ax_ref = fig_ref.subplots(2, 1, sharex=True)

    ax_t, ax_log = ax_ref
    fig = ax_t.get_figure()
    fig.subplots_adjust(hspace=0)

    chain = fitter.chain(discard=0)
    arr_t = chain[:, :, 0]
    mean_t = arr_t.mean(axis=1)

    arr_log = chain[:, :, 1]
    mean_log = arr_log.mean(axis=1)

    # plot
    ax_t.set_title(
        f"Sampled parameters\n Steps={fitter.steps_} - Discarded={0}"
    )

    ax_t.plot(arr_t, alpha=0.5)
    ax_t.plot(mean_t, color="k", label="Mean")
    ax_t.set_ylabel("T")

    ax_log.plot(arr_log, alpha=0.5)
    ax_log.plot(mean_log, color="k", label="Mean")
    ax_log.set_ylabel("log(scale)")
    ax_log.set_xlabel("Steps")
    ax_log.legend()


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
