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
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import BlackBody

from matplotlib.testing.decorators import check_figures_equal

from nirdust import bbody, core

import numpy as np

import pytest

import emcee


# =============================================================================
# TARGET MODEL FUNCTIONS
# =============================================================================


@pytest.mark.parametrize('spectral_unit', [u.AA, u.Hz])
def test_target_model(spectral_unit):

    n = 100
    spectral_axis = np.linspace(20000, 25000, n) * u.AA
    external_flux = np.full(n, 10) 
    T = 1000
    alpha = 5
    beta = 1e5
    gamma = 10

    blackbody = BlackBody(u.Quantity(T, u.K))
    bb_flux = blackbody(spectral_axis.to(spectral_unit, equivalencies=u.spectral())).value

    expected = alpha * external_flux + beta * bb_flux + gamma
    result = bbody.target_model(spectral_axis, external_flux, T, alpha, beta, gamma)

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
    expected = bbody.target_model(spectral_axis, external_flux, *ordered_params)
    assert np.all(tm == expected)


# =============================================================================
# PROBABILITY FUNCTIONS
# =============================================================================

def test_gaussian_log_likelihood():

    spectral_axis = np.linspace(20100., 23000, 1000) * u.AA
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

    noise = 1.

    #evaluate same parameters
    gll = bbody.gaussian_log_likelihood(
        ordered_params, spectral_axis, flux, xternal, noise)
    assert np.ndim(gll) == 0
    assert np.isfinite(gll)
    assert gll < 0

    gll_higher_params = bbody.gaussian_log_likelihood(
        (2000 * u.K, 100, 1e6, 100), spectral_axis, flux, xternal, noise)
    gll_lower_params = bbody.gaussian_log_likelihood(
        (0 * u.K, 0, 0, 0), spectral_axis, flux, xternal, noise)
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
    flux = 10.
    xternal = 8.5
    T = 1000
    alpha = 5
    beta = 5
    gamma = 10     
    ordered_params = (T, alpha, beta, gamma)
    noise = 1.

    gll = bbody.gaussian_log_likelihood(
        ordered_params, spectral_axis, flux, xternal, noise)

    llp = bbody.log_likelihood_prior(ordered_params)

    lp = bbody.log_probability(
        ordered_params, spectral_axis, flux, xternal, noise)

    assert lp == llp + gll



def test_NirdustParameter_init():

    name = 'Spock'
    val = 120 * u.K
    error = (10, 11) * u.K

    param = bbody.NirdustParameter(name, val, error)
    assert param.name == name
    assert param.value == val
    assert np.all(param.uncertainty == error)

    # name must be string
    with pytest.raises(TypeError):
        bbody.NirdustParameter(42, val, error)


# =============================================================================
# BLACKBODY RESULT
# =============================================================================

def test_NirdustResults_parameters():
    nr_inst = bbody.NirdustResults(
        11, 22, 33, bbody.NirdustParameter("gamma", 44, (5, 6)), 
        fitted_blackbody=None, 
        target_spectrum=None, 
        external_spectrum=None,
    )
    assert nr_inst.temperature == 11
    assert nr_inst.alpha == 22
    assert nr_inst.beta == 33
    assert isinstance(nr_inst.gamma, bbody.NirdustParameter)
    assert nr_inst.gamma.name == "gamma"

    assert nr_inst.fitted_blackbody is None
    assert nr_inst.target_spectrum is None
    assert nr_inst.external_spectrum is None


# =============================================================================
# RESULT PLOTS
# =============================================================================

@pytest.mark.xfail
@check_figures_equal()
def test_plot_results(fig_test, fig_ref, NGC4945_continuum):

    spectrum = NGC4945_continuum.cut_edges(19500, 22900).normalize()

    sp_axis = spectrum.spectral_axis
    flux = spectrum.flux

    stella = BlackBody(1100 * u.K)
    instanstella = stella(sp_axis)

    fit_results = bbody.NirdustResults(
        1100, 25, fitted_blackbody=stella, dust=spectrum
    )

    ax_test = fig_test.subplots()
    fit_results.plot(ax=ax_test)

    ax_ref = fig_ref.subplots()

    ax_ref.plot(sp_axis, flux, color="firebrick", label="Dust emission")
    ax_ref.plot(sp_axis, instanstella, color="navy", label="Black body")
    ax_ref.set_xlabel("Angstrom [A]")
    ax_ref.set_ylabel("Intensity [arbitrary units]")
    ax_ref.legend()

@pytest.mark.xfail
@check_figures_equal()
def test_plot_results_default_axis(fig_test, fig_ref, NGC4945_continuum):
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
        fit_results.plot()

    ax_ref = fig_ref.subplots()

    ax_ref.plot(sp_axis, flux, color="firebrick", label="Dust emission")
    ax_ref.plot(sp_axis, instanstella, color="navy", label="Black body")
    ax_ref.set_xlabel("Angstrom [A]")
    ax_ref.set_ylabel("Intensity [arbitrary units]")
    ax_ref.legend()




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
            "steps": 50,
        }
        return base, emcee

    @pytest.fixture
    def fitter(self, params):
        base_params, emcee_params = params
        return bbody.EMCEENirdustFitter(**base_params, **emcee_params)        

    @pytest.fixture
    def fitter_fit(self, fitter):
        return fitter.fit(initial_state=(1000., 1., 1e9, 1.))

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

    def test_isfitted_(self, fitter):
        
        assert not fitter.isfitted_
        fitter.fit(initial_state=(1000., 1., 1e9, 1.))
        assert fitter.isfitted_

    def test_fit_bad_initial_state(self, fitter):

        with pytest.raises(ValueError):
            fitter.fit(initial_state=(1000., 1., 1.))

    def test_fit_already_fitted(self, fitter):

        fitter.fit(initial_state=(1000., 1., 1e9, 1.))
        with pytest.raises(RuntimeError):
            fitter.fit(initial_state=(1000., 1., 1e9, 1.))


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
        return fitter.fit(initial_state=(1000., 1., 1e9, 1.))

    def test_direct_init(self, params):
        base_params, apy_params = params
        fitter = bbody.AstropyNirdustFitter(**base_params, **apy_params)

        assert isinstance(fitter, bbody.AstropyNirdustFitter)
        assert isinstance(fitter.target_spectrum, core.NirdustSpectrum)
        assert isinstance(fitter.external_spectrum, core.NirdustSpectrum)
        assert fitter.extra_conf == base_params["extra_conf"]
        assert fitter.calc_uncertainties == apy_params["calc_uncertainties"]
        assert isinstance(fitter.fitter_, LevMarLSQFitter)

    def test_isfitted_(self, fitter):
        
        assert not fitter.isfitted_
        fitter.fit(initial_state=(1000., 1., 1e9, 1.))
        assert fitter.isfitted_

    def test_fit_bad_initial_state(self, fitter):

        with pytest.raises(ValueError):
            fitter.fit(initial_state=(1000., 1., 1.))

    def test_fit_already_fitted(self, fitter):

        fitter.fit(initial_state=(1000., 1., 1e9, 1.))
        with pytest.raises(RuntimeError):
            fitter.fit(initial_state=(1000., 1., 1e9, 1.))

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

# =============================================================================
#
#    ____  _      _____    _______ ______  _____ _______ _____ 
#   / __ \| |    |  __ \  |__   __|  ____|/ ____|__   __/ ____|
#  | |  | | |    | |  | |    | |  | |__  | (___    | | | (___  
#  | |  | | |    | |  | |    | |  |  __|  \___ \   | |  \___ \ 
#  | |__| | |____| |__| |    | |  | |____ ____) |  | |  ____) |
#   \____/|______|_____/     |_|  |______|_____/   |_| |_____/ 
#                                                             
#
# =============================================================================


@pytest.mark.xfail
@check_figures_equal()
def test_fit_plot(fig_test, fig_ref, NGC4945_continuum):

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

    fitter = bbody.fit_blackbody(spectrumT, externalT, steps=20, seed=42)

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

@pytest.mark.xfail
@check_figures_equal()
def test_fit_plot_non_axis(fig_test, fig_ref, NGC4945_continuum):

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

    fitter = bbody.fit_blackbody(spectrumT, externalT, steps=20, seed=42)

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

@pytest.mark.xfail
def test_fit_plot_unfitted(NGC4945_continuum):

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

    fitter = bbody.NirdustFitter.from_params(
        target_spectrum=spectrumT,
        external_spectrum=externalT,
        seed=42,
    )

    with pytest.raises(RuntimeError):
        fitter.plot()


# =============================================================================
# RESULT PLOTS
# =============================================================================

@pytest.mark.xfail
@check_figures_equal()
def test_plot_results(fig_test, fig_ref, NGC4945_continuum):

    spectrum = NGC4945_continuum.cut_edges(19500, 22900).normalize()

    sp_axis = spectrum.spectral_axis
    flux = spectrum.flux

    stella = BlackBody(1100 * u.K)
    instanstella = stella(sp_axis)

    fit_results = bbody.NirdustResults(
        1100, 25, fitted_blackbody=stella, dust=spectrum
    )

    ax_test = fig_test.subplots()
    fit_results.plot(ax=ax_test)

    ax_ref = fig_ref.subplots()

    ax_ref.plot(sp_axis, flux, color="firebrick", label="Dust emission")
    ax_ref.plot(sp_axis, instanstella, color="navy", label="Black body")
    ax_ref.set_xlabel("Angstrom [A]")
    ax_ref.set_ylabel("Intensity [arbitrary units]")
    ax_ref.legend()

@pytest.mark.xfail
@check_figures_equal()
def test_plot_results_default_axis(fig_test, fig_ref, NGC4945_continuum):
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
        fit_results.plot()

    ax_ref = fig_ref.subplots()

    ax_ref.plot(sp_axis, flux, color="firebrick", label="Dust emission")
    ax_ref.plot(sp_axis, instanstella, color="navy", label="Black body")
    ax_ref.set_xlabel("Angstrom [A]")
    ax_ref.set_ylabel("Intensity [arbitrary units]")
    ax_ref.legend()
