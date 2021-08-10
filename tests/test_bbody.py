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
    beta = 1e5
    gamma = 10

    blackbody = BlackBody(u.Quantity(T, u.K))
    bb_flux = blackbody(
        spectral_axis.to(spectral_unit, equivalencies=u.spectral())
    ).value

    expected = alpha * external_flux + beta * bb_flux + gamma
    result = bbody.target_model(
        spectral_axis, external_flux, T, alpha, beta, gamma
    )

    np.testing.assert_almost_equal(expected, result, decimal=5)


def test_TargetModel_init():

    external_flux = np.ones(99)
    T = 1000
    alpha = 5
    beta = 1e5
    gamma = 10

    model = bbody.TargetModel(external_flux, T, alpha, beta, gamma)
    assert np.all(model.external_flux == external_flux)
    assert model.T == T
    assert model.alpha == alpha
    assert model.beta == beta
    assert model.gamma == gamma


def test_TargetModel_evaluate():

    spectral_axis = 1 * u.AA
    external_flux = 10
    T = 1000
    alpha = 5
    beta = 1e5
    gamma = 10
    ordered_params = (T, alpha, beta, gamma)

    model = bbody.TargetModel(external_flux, *ordered_params)

    # assert target_model is called
    with patch("nirdust.bbody.target_model") as tm:
        # this calls .evaluate()
        model(spectral_axis)
        tm.assert_called_once_with(
            spectral_axis,
            external_flux,
            *ordered_params,
        )

    # assert that evaluate returns the output same as target_model
    tm = model(spectral_axis)
    expected = bbody.target_model(
        spectral_axis, external_flux, *ordered_params
    )
    assert np.all(tm == expected)


# =============================================================================
# PROBABILITY FUNCTIONS
# =============================================================================


def test_gaussian_log_likelihood():

    spectral_axis = np.linspace(20100.0, 23000, 1000) * u.AA
    T = 1000 * u.K
    alpha = 5
    beta = 1e5
    gamma = 10
    ordered_params = (T, alpha, beta, gamma)

    # BlackBody model
    model = BlackBody(T)
    bb = model(spectral_axis).value

    # Linear model
    def tp_line(x, x1, x2, y1, y2):
        return (y2 - y1) / (x2 - x1) * (x - x1) + y1

    wave = spectral_axis.value
    delta_bb = bb[-1] - bb[0]
    y1_line, y2_line = bb[0] + 2 / 3 * delta_bb, bb[0] + 1 / 3 * delta_bb
    nuclear = tp_line(wave, wave[0], wave[-1], y1_line, y2_line) + 0.2

    # Total model
    xternal = nuclear / alpha
    flux = nuclear + bb + gamma

    noise = 1.0

    # evaluate same parameters
    gll = bbody.gaussian_log_likelihood(
        ordered_params, spectral_axis, flux, xternal, noise
    )
    assert np.ndim(gll) == 0
    assert np.isfinite(gll)
    assert gll < 0

    gll_higher_params = bbody.gaussian_log_likelihood(
        (2000 * u.K, 100, 1e6, 100), spectral_axis, flux, xternal, noise
    )
    gll_lower_params = bbody.gaussian_log_likelihood(
        (0 * u.K, 0, 0, 0), spectral_axis, flux, xternal, noise
    )
    assert gll > gll_higher_params
    assert gll > gll_lower_params


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
    )
    assert nr_inst.temperature.value == 11
    assert nr_inst.alpha.value == 22
    assert nr_inst.beta.value == 33
    assert isinstance(nr_inst.gamma, bbody.NirdustParameter)
    assert isinstance(nr_inst.fitted_blackbody, BlackBody)
    assert isinstance(nr_inst.target_spectrum, core.NirdustSpectrum)
    assert isinstance(nr_inst.external_spectrum, core.NirdustSpectrum)


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
    )

    # Nirdust plot
    ax_test = fig_test.subplots()
    fit_results.plot(ax=ax_test)

    # Expected plot
    ax_ref = fig_ref.subplots()

    prediction = bbody.target_model(
        synth_total_noised.spectral_axis,
        synth_external_noised.flux.value,
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
    )

    ax_test = fig_test.subplots()
    with patch("matplotlib.pyplot.gca", return_value=ax_test):
        fit_results.plot()

    # Expected plot
    ax_ref = fig_ref.subplots()

    prediction = bbody.target_model(
        synth_total_noised.spectral_axis,
        synth_external_noised.flux.value,
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


def test_BaseFitter_new_class(synth_total_noised):
    class NewFitter(bbody.BaseFitter):
        def run_model(self):
            pass

        def best_parameters(self):
            pass

    fitter = NewFitter(
        target_spectrum=synth_total_noised,
        external_spectrum=synth_total_noised,
        extra_conf={},
    )
    assert hasattr(fitter, "fit")


def test_BaseFitter_abstract_best_parameters(synth_total_noised):
    class NewFitter(bbody.BaseFitter):
        # No best_parameters
        def run_model(self):
            pass

    with pytest.raises(TypeError):
        NewFitter(
            target_spectrum=synth_total_noised,
            external_spectrum=synth_total_noised,
            extra_conf={},
        )


def test_BaseFitter_abstract_run_model(synth_total_noised):
    class NewFitter(bbody.BaseFitter):
        # No run_model
        def best_parameters(self):
            pass

    with pytest.raises(TypeError):
        NewFitter(
            target_spectrum=synth_total_noised,
            external_spectrum=synth_total_noised,
            extra_conf={},
        )


class Test_EMCEENirdustFitter:
    @pytest.fixture
    def params(self, synth_total_noised, synth_external_noised):
        # BaseFitter params
        base = {
            "target_spectrum": synth_total_noised,
            "external_spectrum": synth_external_noised,
            "extra_conf": {},
        }
        # EMCEENirdustFitter params
        # short steps for a fast run, convergence is not important here
        emcee = {
            "nwalkers": 9,
            "seed": 0,
            "steps": 20,
        }
        return base, emcee

    @pytest.fixture
    def fitter(self, params):
        base_params, emcee_params = params
        return bbody.EMCEENirdustFitter(**base_params, **emcee_params)

    @pytest.fixture
    def fitter_fit(self, fitter):
        return fitter.fit(initial_state=(1000.0, 1.0, 1e9, 1.0))

    def test_direct_init(self, params):
        base_params, emcee_params = params
        fitter = bbody.EMCEENirdustFitter(**base_params, **emcee_params)

        assert isinstance(fitter, bbody.EMCEENirdustFitter)
        assert isinstance(fitter.target_spectrum, core.NirdustSpectrum)
        assert isinstance(fitter.external_spectrum, core.NirdustSpectrum)
        assert fitter.extra_conf == base_params["extra_conf"]
        assert fitter.nwalkers == emcee_params["nwalkers"]
        assert fitter.seed == emcee_params["seed"]
        assert fitter.steps == emcee_params["steps"]
        assert isinstance(fitter.sampler_, emcee.EnsembleSampler)

    def test_total_noise_(self, params):
        base_params, emcee_params = params
        fitter = bbody.EMCEENirdustFitter(**base_params, **emcee_params)

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

    def test_chain(self, fitter_fit, params):
        _, emcee_params = params
        nwalkers = emcee_params["nwalkers"]
        steps = emcee_params["steps"]

        c = fitter_fit.chain()
        assert c.shape == (steps, nwalkers, 4)
        c = fitter_fit.chain(discard=10)
        assert c.shape == (steps - 10, nwalkers, 4)

    def test_best_parameters(self, fitter_fit):
        temp, alpha, beta, gamma = fitter_fit.best_parameters(discard=0)

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
        assert len(temp.uncertainty) == 2
        assert len(alpha.uncertainty) == 2
        assert len(beta.uncertainty) == 2
        assert len(gamma.uncertainty) == 2

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

    def test_fit_plot_unfitted(self, fitter):

        with pytest.raises(RuntimeError):
            fitter.plot()

    @check_figures_equal()
    def test_fit_plot(self, fig_test, fig_ref, fitter_fit):

        chain = fitter_fit.chain(discard=0)

        # test figure is generated
        ax_test = fig_test.subplots(4, 1, sharex=True)
        fitter_fit.plot(ax=ax_test)

        # ref figure is constructed

        ax_ref = fig_ref.subplots(4, 1, sharex=True)
        ax_t, ax_a, ax_b, ax_g = ax_ref

        fig = ax_t.get_figure()
        fig.subplots_adjust(hspace=0)

        ax_t.set_title(
            f"Sampled parameters\n Steps={fitter_fit.steps} - Discarded={0}"
        )
        for idx, ax in enumerate(ax_ref):
            arr = chain[:, :, idx]
            mean = arr.mean(axis=1)
            ax.plot(arr, alpha=0.5)
            ax.plot(mean, color="k", label="Mean")
            ax.legend()

        ax_t.set_ylabel("T")
        ax_a.set_ylabel("alpha")
        ax_b.set_ylabel("beta")
        ax_g.set_ylabel("gamma")
        ax_g.set_xlabel("Steps")

    @check_figures_equal()
    def test_fit_plot_non_axis(self, fig_test, fig_ref, fitter_fit):

        chain = fitter_fit.chain(discard=0)

        # test figure is generated
        ax_test = fig_test.subplots(4, 1, sharex=True)
        with patch(
            "matplotlib.pyplot.subplots", return_value=(fig_test, ax_test)
        ):
            fitter_fit.plot()

        # ref figure is constructed

        ax_ref = fig_ref.subplots(4, 1, sharex=True)
        ax_t, ax_a, ax_b, ax_g = ax_ref

        fig = ax_t.get_figure()
        fig.subplots_adjust(hspace=0)

        ax_t.set_title(
            f"Sampled parameters\n Steps={fitter_fit.steps} - Discarded={0}"
        )
        for idx, ax in enumerate(ax_ref):
            arr = chain[:, :, idx]
            mean = arr.mean(axis=1)
            ax.plot(arr, alpha=0.5)
            ax.plot(mean, color="k", label="Mean")
            ax.legend()

        ax_t.set_ylabel("T")
        ax_a.set_ylabel("alpha")
        ax_b.set_ylabel("beta")
        ax_g.set_ylabel("gamma")
        ax_g.set_xlabel("Steps")


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


class Test_Backend:
    @pytest.fixture
    def apy_params(self, synth_total_noised, synth_external_noised):
        # BaseFitter and AstropyNirdustFitter params
        apy = {
            "backend": "astropy",
            "target_spectrum": synth_total_noised,
            "external_spectrum": synth_external_noised,
            "maxiter": 10,
            "calc_uncertainties": True,
        }
        return apy

    @pytest.fixture
    def emcee_params(self, synth_total_noised, synth_external_noised):
        # BaseFitter and EMCEENirdustFitter params
        emcee = {
            "backend": "emcee",
            "target_spectrum": synth_total_noised,
            "external_spectrum": synth_external_noised,
            "nwalkers": 9,
            "seed": 0,
            "steps": 50,
        }
        return emcee

    def test_backend_emcee(self, emcee_params):
        fitter = bbody.fit_blackbody(**emcee_params)
        assert isinstance(fitter, bbody.EMCEENirdustFitter)
        assert fitter.isfitted_

    def test_backend_astropy(self, apy_params):
        fitter = bbody.fit_blackbody(**apy_params)
        assert isinstance(fitter, bbody.AstropyNirdustFitter)
        assert fitter.isfitted_

    def test_backend_invalid(self, apy_params):
        apy_params["backend"] = "nirdust"

        with pytest.raises(bbody.InvalidBackendError):
            bbody.fit_blackbody(**apy_params)
