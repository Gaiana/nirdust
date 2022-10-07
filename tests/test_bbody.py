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
from astropy.modeling.models import BlackBody

from matplotlib.testing.decorators import check_figures_equal

from nirdust import bbody, core

import numpy as np

import pytest

from scipy.optimize import OptimizeResult


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

    expected = alpha * external_flux + 10**beta * bb_flux + 10**gamma

    external_spectrum = core.NirdustSpectrum(spectral_axis, external_flux)
    result = bbody.target_model(external_spectrum, T, alpha, beta, gamma)

    np.testing.assert_almost_equal(expected, result, decimal=5)


def test_target_model_from_fixture(synth_total, synth_external, true_params):
    T, alpha, beta, gamma = (
        true_params["T"],
        true_params["alpha"],
        true_params["beta"],
        true_params["gamma"],
    )

    prediction = bbody.target_model(synth_external, T, alpha, beta, gamma)
    diff = prediction - synth_total.flux.value
    np.testing.assert_allclose(diff, 0.0, atol=1e-14)


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


@check_figures_equal(extensions=("png", "pdf"))
@pytest.mark.parametrize("show_components", [True, False])
def test_plot_results_show_components(
    fig_test,
    fig_ref,
    true_params,
    synth_total_noised,
    synth_external_noised,
    show_components,
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
    axes_test = fig_test.subplots(2, 1)
    fit_results.plot(axes=axes_test, show_components=show_components)

    # Expected plot
    axes_ref = fig_ref.subplots(2, 1)

    prediction = bbody.target_model(
        synth_external_noised,
        true_params["T"],
        true_params["alpha"],
        true_params["beta"],
        true_params["gamma"],
    )
    wave_axis = fit_results.target_spectrum.spectral_axis.value

    ax, axr = axes_ref

    # Target
    data_kws = {}
    data_kws.setdefault("color", "firebrick")
    ax.plot(
        wave_axis,
        fit_results.target_spectrum.flux.value,
        label="target",
        **data_kws,
    )

    # Prediction
    model_kws = {}
    model_kws.setdefault("color", "Navy")
    ax.plot(
        wave_axis,
        prediction,
        label="prediction",
        **model_kws,
    )

    # Show components
    if show_components:
        alpha_term = (
            fit_results.alpha.value * fit_results.external_spectrum.flux.value
        )
        beta_term = (
            10**fit_results.beta.value
        ) * fit_results.fitted_blackbody(
            fit_results.target_spectrum.spectral_axis
        ).value
        gamma_term = (10**fit_results.gamma.value) * np.ones_like(wave_axis)

        ax.plot(
            wave_axis,
            alpha_term,
            label=r"$\alpha$-term",
            linestyle="--",
            color="sandybrown",
        )
        ax.plot(
            wave_axis,
            beta_term,
            label=r"$\beta$-term",
            linestyle="--",
            color="darkorchid",
        )
        ax.plot(
            wave_axis,
            gamma_term,
            label=r"$\gamma$-term",
            linestyle="--",
            color="darkgreen",
        )

    # Residuals
    residuals = (
        fit_results.target_spectrum.flux.value - prediction
    ) / fit_results.target_spectrum.noise
    axr.plot(
        wave_axis,
        residuals,
        linestyle="solid",
        color="gray",
    )

    # Ticks and Labels
    ax.tick_params(axis="y", labelsize=12)
    axr.tick_params(axis="both", labelsize=12)

    axr.set_xlabel(r"Wavelength [$\AA$]", fontsize=12)
    axr.set_ylabel("Residual", fontsize=12)
    ax.set_ylabel("Intensity [arbitrary units]", fontsize=12)
    ax.legend(fontsize=12)


@check_figures_equal(extensions=("png", "pdf"))
@pytest.mark.parametrize("show_components", [True, False])
def test_plot_results_default_axis_show_components(
    fig_test,
    fig_ref,
    true_params,
    synth_total_noised,
    synth_external_noised,
    show_components,
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
    gkw = {"height_ratios": [4, 1], "hspace": 0}
    fig_test.update({"size_inches": (8, 6)})
    fig_test.tight_layout()
    with patch("matplotlib.pyplot.figure", return_value=fig_test):
        fit_results.plot(show_components=show_components)

    # Expected plot
    gkw = {"height_ratios": [4, 1], "hspace": 0}
    fig_ref.update({"size_inches": (8, 6)})
    fig_ref.tight_layout()
    axes_ref = fig_ref.subplots(2, 1, sharex=True, gridspec_kw=gkw)

    ax, axr = axes_ref

    prediction = bbody.target_model(
        synth_external_noised,
        true_params["T"],
        true_params["alpha"],
        true_params["beta"],
        true_params["gamma"],
    )
    wave_axis = fit_results.target_spectrum.spectral_axis.value

    ax, axr = axes_ref

    # Target
    data_kws = {}
    data_kws.setdefault("color", "firebrick")
    ax.plot(
        wave_axis,
        fit_results.target_spectrum.flux.value,
        label="target",
        **data_kws,
    )

    # Prediction
    model_kws = {}
    model_kws.setdefault("color", "Navy")
    ax.plot(
        wave_axis,
        prediction,
        label="prediction",
        **model_kws,
    )

    # Show components
    if show_components:
        alpha_term = (
            fit_results.alpha.value * fit_results.external_spectrum.flux.value
        )
        beta_term = (
            10**fit_results.beta.value
        ) * fit_results.fitted_blackbody(
            fit_results.target_spectrum.spectral_axis
        ).value
        gamma_term = (10**fit_results.gamma.value) * np.ones_like(wave_axis)

        ax.plot(
            wave_axis,
            alpha_term,
            label=r"$\alpha$-term",
            linestyle="--",
            color="sandybrown",
        )
        ax.plot(
            wave_axis,
            beta_term,
            label=r"$\beta$-term",
            linestyle="--",
            color="darkorchid",
        )
        ax.plot(
            wave_axis,
            gamma_term,
            label=r"$\gamma$-term",
            linestyle="--",
            color="darkgreen",
        )

    # Residuals
    residuals = (
        fit_results.target_spectrum.flux.value - prediction
    ) / fit_results.target_spectrum.noise
    axr.plot(
        wave_axis,
        residuals,
        linestyle="solid",
        color="gray",
    )

    # Ticks and Labels
    ax.tick_params(axis="y", labelsize=12)
    axr.tick_params(axis="both", labelsize=12)

    axr.set_xlabel(r"Wavelength [$\AA$]", fontsize=12)
    axr.set_ylabel("Residual", fontsize=12)
    ax.set_ylabel("Intensity [arbitrary units]", fontsize=12)
    ax.legend(fontsize=12)


# =============================================================================
# CONSTRAINT FUNCTIONS
# =============================================================================


def test_alpha_vs_beta(synth_total, synth_external, true_params):
    T, alpha, beta, gamma = (
        true_params["T"],
        true_params["alpha"],
        true_params["beta"],
        true_params["gamma"],
    )
    theta = T, alpha, beta, gamma

    alpha_term = np.mean(alpha * synth_external.flux.value)
    beta_term = np.mean(synth_total.flux.value - alpha_term - 10**gamma)

    expected = alpha_term - beta_term
    result = bbody.alpha_vs_beta(theta, synth_total, synth_external)

    assert np.ndim(result) == 0
    assert np.isfinite(result)
    np.testing.assert_almost_equal(expected, result, 1e-14)


@pytest.mark.parametrize("gamma", [-1, 10])
def test_make_gamma_vs_target_flux_invalid_fraction(gamma):
    with pytest.raises(ValueError):
        bbody.make_gamma_vs_target_flux(gamma)


@pytest.mark.parametrize("gamma", [0, 0.5, 1])
def test_make_gamma_vs_target_flux_callable(gamma):
    foo = bbody.make_gamma_vs_target_flux(gamma)
    assert callable(foo)


def test_make_gamma_vs_target_flux(synth_total, synth_external, true_params):
    T, alpha, beta, gamma = (
        true_params["T"],
        true_params["alpha"],
        true_params["beta"],
        true_params["gamma"],
    )
    theta = T, alpha, beta, gamma
    gamma_vs_target_flux = bbody.make_gamma_vs_target_flux(gamma_fraction=0.05)
    result = gamma_vs_target_flux(theta, synth_total, synth_external)
    expected = 0.14237615

    assert np.ndim(result) == 0
    assert np.isfinite(result)
    np.testing.assert_almost_equal(result, expected, 1e-6)


def test_make_constraints(synth_total, synth_external):
    args = (synth_total, synth_external)
    gf = 0.05
    result = bbody.make_constraints(args, gf)

    assert len(result) == 2
    assert type(result[0]) is dict
    assert type(result[1]) is dict
    for d in result:
        assert d["type"] == "ineq"
        assert d["args"] == args
        assert callable(d["fun"])


@pytest.mark.parametrize("options", [None, {"maxiter": 10}])
def test_minimizer_kwargs(synth_total, synth_external, options):
    args = (synth_total, synth_external)
    bounds = ((0.0, 2000.0), (0, 20), (6, 10), (-10, 0))
    constraints = bbody.make_constraints(args, 0.05)

    if options is None:
        opt = {"maxiter": 1000}
    else:
        opt = options

    result = bbody.make_minimizer_kwargs(args, bounds, constraints, options)
    expected = {
        "method": "SLSQP",
        "args": args,
        "bounds": bounds,
        "constraints": constraints,
        "options": opt,
        "jac": "3-point",
    }
    assert result == expected


# =============================================================================
# FITTER CLASES
# =============================================================================


class Test_BasinhoppingFitter:
    @pytest.fixture
    def params(self, synth_total, synth_external):
        # BasinhoppingFitter params
        kwargs = {
            "target_spectrum": synth_total,
            "external_spectrum": synth_external,
            "basinhopping_kwargs": {
                "niter": 2,  # Only 2 for fast testing
                "T": 100,
                "stepsize": 1,
                "seed": 42,
            },
        }
        return kwargs

    @pytest.fixture
    def fitter(self, params):
        return bbody.BasinhoppingFitter(**params)

    @pytest.fixture
    def minimizer_kwargs(self, synth_total, synth_external):
        # todo
        bounds = ((0.0, 2000.0), (0, 20), (6, 10), (-10, 0))
        args = (synth_total, synth_external)
        constraints = bbody.make_constraints(args, 0.05)
        return bbody.make_minimizer_kwargs(args, bounds, constraints)

    def test_direct_init_types(self, params):
        fitter = bbody.BasinhoppingFitter(**params)

        assert isinstance(fitter, bbody.BasinhoppingFitter)
        assert isinstance(fitter.target_spectrum, core.NirdustSpectrum)
        assert isinstance(fitter.external_spectrum, core.NirdustSpectrum)
        assert isinstance(fitter.basinhopping_kwargs, dict)

    def test_direct_init_values(self, params):
        fitter = bbody.BasinhoppingFitter(**params)

        assert fitter.target_spectrum is params["target_spectrum"]
        assert fitter.external_spectrum is params["external_spectrum"]
        assert fitter.basinhopping_kwargs is params["basinhopping_kwargs"]

    def test_total_noise_(self, params):
        fitter = bbody.BasinhoppingFitter(**params)

        noise_tar = params["target_spectrum"].noise
        noise_ext = params["external_spectrum"].noise

        expected = np.sqrt(noise_ext**2 + noise_tar**2)
        result = fitter.total_noise_

        np.testing.assert_almost_equal(result, expected, decimal=14)

    def test_ndim_property(self, fitter):
        assert fitter.ndim_ == 4

    def test_run_model(self, fitter, minimizer_kwargs):
        x0 = (1000.0, 8.0, 9.0, -5.0)
        result = fitter.run_model(x0, minimizer_kwargs)
        assert isinstance(result, OptimizeResult)

    def test_ConvergenceWarning(self, fitter, minimizer_kwargs):
        # Really bad initial guess, shouldn't converge
        x0 = (1000.0, 0.0, 0.0, 0.0)
        with pytest.warns(bbody.ConvergenceWarning):
            fitter.run_model(x0, minimizer_kwargs)

    def test_fit(self, fitter, minimizer_kwargs):
        x0 = (1000.0, 8.0, 9.0, -5.0)
        result = fitter.fit(x0, minimizer_kwargs)
        assert isinstance(result, bbody.NirdustResults)

    def test_fit_invalid_x0(self, fitter, minimizer_kwargs):
        x0 = (1000.0, 8.0, 9.0)
        with pytest.raises(ValueError):
            fitter.fit(x0, minimizer_kwargs)


@pytest.mark.slow
class Test_NirdustSanity:
    @pytest.mark.parametrize("snr", [200, 500, 1000])
    def test_results_within_bounds(
        self, synth_total, synth_external, with_noise, snr
    ):

        total_noised = with_noise(synth_total, snr, seed=0)
        external_noised = with_noise(synth_external, snr, seed=123)

        result = bbody.fit_blackbody(total_noised, external_noised, seed=42)
        bounds = bbody.BOUNDS

        T = result.temperature.value.value
        alpha = result.alpha.value
        beta = result.beta.value
        gamma = result.gamma.value

        assert bounds[0][0] <= T <= bounds[0][1]
        assert bounds[1][0] <= alpha <= bounds[1][1]
        assert bounds[2][0] <= beta <= bounds[2][1]
        assert bounds[3][0] <= gamma <= bounds[3][1]

    @pytest.mark.parametrize("snr", [100, 300, 500])
    def test_aprox_fitted_values_high_snr(
        self, synth_total, synth_external, true_params, with_noise, snr
    ):

        total_noised = with_noise(synth_total, snr, seed=42)
        external_noised = with_noise(synth_external, snr, seed=50)

        result = bbody.fit_blackbody(
            total_noised,
            external_noised,
            niter=100,
            seed=0,
            stepsize=1,
        )

        success = result.minimizer_results.success
        T = result.temperature.value.value
        alpha = result.alpha.value
        beta = result.beta.value
        gamma = result.gamma.value

        assert success
        assert np.abs(T - true_params["T"].value) < 100.0
        assert np.abs(alpha - true_params["alpha"]) < 3.0
        assert np.abs(beta - true_params["beta"]) < 2.0
        # gamma is noisy and cant be fitted very precisely
        assert np.abs(gamma - true_params["gamma"]) < 10.0
