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
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import BlackBody

import emcee

from matplotlib.testing.decorators import check_figures_equal

from nirdust import bbody, core

import numpy as np

from scipy.optimize import OptimizeResult

import pytest


# =============================================================================
# TARGET MODEL FUNCTIONS
# =============================================================================


@pytest.mark.parametrize("spectral_unit", [u.AA, u.Hz])
def test_target_model(spectral_unit):

    n = 100
    spectral_axis = np.linspace(20000, 25000, n) * u.AA
    external_flux = np.full(n, 10)
    T = 1000
    alpha = 5
    beta = 7
    gamma = -1

    blackbody = BlackBody(u.Quantity(T, u.K))
    bb_flux = blackbody(
        spectral_axis.to(spectral_unit, equivalencies=u.spectral())
    ).value

    expected = alpha * external_flux + 10 ** beta * bb_flux + 10 ** gamma

    external_spectrum = core.NirdustSpectrum(spectral_axis, external_flux)
    result = bbody.target_model(external_spectrum, T, alpha, beta, gamma)

    np.testing.assert_almost_equal(expected, result, decimal=5)


# =============================================================================
# PROBABILITY FUNCTIONS
# =============================================================================


def test_negative_gaussian_log_likelihood(
    synth_total_noised, synth_external_noised, true_params
):

    ordered_params = (
        true_params["T"],
        true_params["alpha"],
        true_params["beta"],
        true_params["gamma"],
    )

    # evaluate same parameters
    ngll = bbody.negative_gaussian_log_likelihood(
        ordered_params, synth_total_noised, synth_external_noised
    )
    assert np.ndim(ngll) == 0
    assert np.isfinite(ngll)
    assert ngll < 0

    ngll_higher_params = bbody.negative_gaussian_log_likelihood(
        (1100 * u.K, 3.1, 8.1, -4.1), synth_total_noised, synth_external_noised
    )
    ngll_lower_params = bbody.negative_gaussian_log_likelihood(
        (900 * u.K, 2.9, 7.9, -3.9), synth_total_noised, synth_external_noised
    )
    assert ngll < ngll_higher_params
    assert ngll < ngll_lower_params


@pytest.mark.xfail
@pytest.mark.parametrize("t", [-1, 1, 2999, 3001])
@pytest.mark.parametrize("a", [-1, 1])
@pytest.mark.parametrize("b", [-1, 1])
def test_log_likelihood_prior(t, a, b):

    # no seed, because the test is independent of gamma
    gamma = np.random.random()
    theta = (t, a, b, gamma)

    Tok = 0 < t < 3000
    alphaok = a > 0
    betaok = b > 0

    if Tok and alphaok and betaok:
        expected = 0.0
    else:
        expected = -np.inf

    llp = bbody.log_likelihood_prior(theta)
    assert llp == expected


@pytest.mark.xfail
def test_log_probability():

    spectral_axis = 1 * u.AA
    flux = 10.0
    xternal = 8.5
    T = 1000
    alpha = 5
    beta = 5
    gamma = 10
    ordered_params = (T, alpha, beta, gamma)
    noise = 1.0

    gll = bbody.gaussian_log_likelihood(
        ordered_params, spectral_axis, flux, xternal, noise
    )

    llp = bbody.log_likelihood_prior(ordered_params)

    lp = bbody.log_probability(
        ordered_params, spectral_axis, flux, xternal, noise
    )

    assert lp == llp + gll


# =============================================================================
# RESULT CLASES
# =============================================================================


def test_NirdustParameter_init():

    name = "Spock"
    val = 120 * u.K
    error = (10, 11) * u.K

    param = bbody.NirdustParameter(name, val, error)
    assert param.name == name
    assert param.value == val
    assert np.all(param.uncertainty == error)


def test_NirdustParameter_invalid_init():

    name = 42
    val = 120 * u.K
    error = (10, 11) * u.K

    # name must be string
    with pytest.raises(TypeError):
        bbody.NirdustParameter(name, val, error)


def test_NirdustResults_parameters(NGC4945_continuum):
    nr_inst = bbody.NirdustResults(
        bbody.NirdustParameter("AAA", 11, (5, 6)),
        bbody.NirdustParameter("BBB", 22, (5, 6)),
        bbody.NirdustParameter("CCC", 33, (5, 6)),
        bbody.NirdustParameter("DDD", 44, (5, 6)),
        fitted_blackbody=BlackBody(0.0 * u.K),
        target_spectrum=NGC4945_continuum,
        external_spectrum=NGC4945_continuum,
        minimizer_results=OptimizeResult({}),
    )
    assert nr_inst.temperature.value == 11
    assert nr_inst.alpha.value == 22
    assert nr_inst.beta.value == 33
    assert isinstance(nr_inst.gamma, bbody.NirdustParameter)
    assert isinstance(nr_inst.fitted_blackbody, BlackBody)
    assert isinstance(nr_inst.target_spectrum, core.NirdustSpectrum)
    assert isinstance(nr_inst.external_spectrum, core.NirdustSpectrum)
    assert isinstance(nr_inst.minimizer_results, OptimizeResult)


def test_NirdustResults_invalid_parameters():
    with pytest.raises(TypeError):
        bbody.NirdustResults(
            0,
            0,
            0,
            0,
            fitted_blackbody=None,
            target_spectrum=None,
            external_spectrum=None,
            minimizer_results=None,
        )


# =============================================================================
# RESULT PLOTS
# =============================================================================


@check_figures_equal()
def test_plot_results(
    fig_test, fig_ref, true_params, synth_total_noised, synth_external_noised
):

    fit_results = bbody.NirdustResults(
        bbody.NirdustParameter("Temperature", true_params["T"], (0, 0)),
        bbody.NirdustParameter("Alpha", true_params["alpha"], (0, 0)),
        bbody.NirdustParameter("Beta", true_params["beta"], (0, 0)),
        bbody.NirdustParameter("Gamma", true_params["gamma"], (0, 0)),
        fitted_blackbody=BlackBody(true_params["T"]),
        target_spectrum=synth_total_noised,
        external_spectrum=synth_external_noised,
        minimizer_results=OptimizeResult({}),
    )

    # Nirdust plot
    ax_test = fig_test.subplots()
    fit_results.plot(ax=ax_test)

    # Expected plot
    ax_ref = fig_ref.subplots()

    prediction = bbody.target_model(
        synth_external_noised,
        true_params["T"],
        true_params["alpha"],
        true_params["beta"],
        true_params["gamma"],
    )
    ax_ref.plot(
        synth_total_noised.spectral_axis.value,
        synth_total_noised.flux.value,
        label="target",
        color="firebrick",
    )
    ax_ref.plot(
        synth_total_noised.spectral_axis.value,
        prediction,
        label="prediction",
        color="Navy",
    )
    ax_ref.set_xlabel("Angstrom [A]")
    ax_ref.set_ylabel("Intensity [arbitrary units]")
    ax_ref.legend()


@check_figures_equal()
def test_plot_results_default_axis(
    fig_test, fig_ref, true_params, synth_total_noised, synth_external_noised
):

    fit_results = bbody.NirdustResults(
        bbody.NirdustParameter("Temperature", true_params["T"], (0, 0)),
        bbody.NirdustParameter("Alpha", true_params["alpha"], (0, 0)),
        bbody.NirdustParameter("Beta", true_params["beta"], (0, 0)),
        bbody.NirdustParameter("Gamma", true_params["gamma"], (0, 0)),
        fitted_blackbody=BlackBody(true_params["T"]),
        target_spectrum=synth_total_noised,
        external_spectrum=synth_external_noised,
        minimizer_results=OptimizeResult({}),
    )

    ax_test = fig_test.subplots()
    with patch("matplotlib.pyplot.gca", return_value=ax_test):
        fit_results.plot()

    # Expected plot
    ax_ref = fig_ref.subplots()

    prediction = bbody.target_model(
        synth_external_noised,
        true_params["T"],
        true_params["alpha"],
        true_params["beta"],
        true_params["gamma"],
    )
    ax_ref.plot(
        synth_total_noised.spectral_axis,
        synth_total_noised.flux,
        label="target",
        color="firebrick",
    )
    ax_ref.plot(
        synth_total_noised.spectral_axis,
        prediction,
        label="prediction",
        color="Navy",
    )
    ax_ref.set_xlabel("Angstrom [A]")
    ax_ref.set_ylabel("Intensity [arbitrary units]")
    ax_ref.legend()


# =============================================================================
# FITTER CLASES
# =============================================================================


class Test_AstropyNirdustFitter:
    @pytest.fixture
    def params(self, synth_total_noised, synth_external_noised):
        # BaseFitter params
        base = {
            "target_spectrum": synth_total_noised,
            "external_spectrum": synth_external_noised,
            "extra_conf": {"maxiter": 10},
        }
        # AstropyNirdustFitter params
        apy = {
            "calc_uncertainties": True,
        }
        return base, apy

    @pytest.fixture
    def fitter(self, params):
        base_params, apy_params = params
        return bbody.AstropyNirdustFitter(**base_params, **apy_params)

    @pytest.fixture
    def fitter_fit(self, fitter):
        return fitter.fit(initial_state=(1000.0, 1.0, 1e9, 1.0))

    def test_direct_init(self, params):
        base_params, apy_params = params
        fitter = bbody.AstropyNirdustFitter(**base_params, **apy_params)

        assert isinstance(fitter, bbody.AstropyNirdustFitter)
        assert isinstance(fitter.target_spectrum, core.NirdustSpectrum)
        assert isinstance(fitter.external_spectrum, core.NirdustSpectrum)
        assert fitter.extra_conf == base_params["extra_conf"]
        assert fitter.calc_uncertainties == apy_params["calc_uncertainties"]
        assert isinstance(fitter.fitter_, LevMarLSQFitter)

    def test_total_noise_(self, params):
        base_params, apy_params = params
        fitter = bbody.AstropyNirdustFitter(**base_params, **apy_params)

        noise_tar = base_params["target_spectrum"].noise
        noise_ext = base_params["external_spectrum"].noise

        expected = np.sqrt(noise_ext ** 2 + noise_tar ** 2)
        result = fitter.total_noise_

        np.testing.assert_almost_equal(result, expected, decimal=14)

    def test_isfitted_(self, fitter):

        assert not fitter.isfitted_
        fitter.fit(initial_state=(1000.0, 1.0, 1e9, 1.0))
        assert fitter.isfitted_

    def test_fit_bad_initial_state(self, fitter):

        with pytest.raises(ValueError):
            fitter.fit(initial_state=(1000.0, 1.0, 1.0))

    def test_fit_already_fitted(self, fitter):

        fitter.fit(initial_state=(1000.0, 1.0, 1e9, 1.0))
        with pytest.raises(RuntimeError):
            fitter.fit(initial_state=(1000.0, 1.0, 1e9, 1.0))

    def test_best_parameters(self, fitter_fit):
        temp, alpha, beta, gamma = fitter_fit.best_parameters()

        assert isinstance(temp, bbody.NirdustParameter)
        assert isinstance(alpha, bbody.NirdustParameter)
        assert isinstance(beta, bbody.NirdustParameter)
        assert isinstance(gamma, bbody.NirdustParameter)
        assert temp.name == "Temperature"
        assert alpha.name == "Alpha"
        assert beta.name == "Beta"
        assert gamma.name == "Gamma"
        assert isinstance(temp.value, u.Quantity)
        assert temp.value.unit == u.K
        for param in [temp, alpha, beta, gamma]:
            assert param.uncertainty is None or len(param.uncertainty) == 2

    def test_result(self, fitter_fit):
        result = fitter_fit.result()

        assert isinstance(result, bbody.NirdustResults)
        assert isinstance(result.temperature, bbody.NirdustParameter)
        assert isinstance(result.alpha, bbody.NirdustParameter)
        assert isinstance(result.beta, bbody.NirdustParameter)
        assert isinstance(result.gamma, bbody.NirdustParameter)
        assert isinstance(result.fitted_blackbody, BlackBody)
        assert isinstance(result.target_spectrum, core.NirdustSpectrum)
        assert isinstance(result.external_spectrum, core.NirdustSpectrum)
