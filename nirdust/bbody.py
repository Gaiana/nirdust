#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   NirDust Project (https://github.com/Gaiana/nirdust)
# Copyright (c) 2020, 2021 Gaia Gaspar, Jose Alacoria
# License: MIT
#   Full Text: https://github.com/Gaiana/nirdust/LICENSE

# ==============================================================================
# DOCS
# ==============================================================================

"""Blackbody/temperature utilities."""


# ==============================================================================
# IMPORTS
# ==============================================================================

import abc

from astropy import units as u
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.models import BlackBody
from astropy.modeling.fitting import LevMarLSQFitter

import attr
from attr import validators

import emcee

import matplotlib.pyplot as plt

import numpy as np

from .core import NirdustSpectrum


# ==============================================================================
# TARGET SPECTRUM MODEL
# ==============================================================================


def reduced_target_model(spectral_axis, external_flux, T, alpha, beta, gamma):
    """Compute the expected dust spectrum given a blackbody prediction.

    Parameters
    ----------
    blackbody_flux: `~numpy.ndarray`
        Blackbody spectrum intensity.

    target_flux: `~numpy.ndarray`
        Total spectrum intensity.

    external_flux: `~numpy.ndarray`
        External spectrum intensity.

    Return
    ------
    dust_flux: `~numpy.ndarray`
        Remaining dust spectrum intensity.
    """
    # calculate the model
    blackbody = BlackBody(u.Quantity(T, u.K))
    bb_flux = blackbody(spectral_axis).value

    # bb_reduced = bb_flux - bb_flux.mean()
    # external_reduced = external_flux - external_flux.mean()

    # prediction = alpha * external_reduced + beta * bb_reduced
    prediction = alpha * external_flux + beta * bb_flux + gamma
    return prediction


class ReducedTargetModel(Fittable1DModel):

    external_flux = Parameter(fixed=True)
    T = Parameter(min=0., max=3000.)
    alpha = Parameter(min=0.)
    beta = Parameter(min=0.)
    gamma = Parameter()

    @staticmethod
    def evaluate(spectral_axis, external_flux, T, alpha, beta, gamma):
        # calculate the model
        return reduced_target_model(
            spectral_axis, external_flux, T, alpha, beta, gamma,
            )

# ==============================================================================
# EMCEE FUNCTIONS
# ==============================================================================

def gaussian_log_likelihood(theta, spectral_axis, target_flux, external_flux):
    """Gaussian logarithmic likelihood.

    Compute the likelihood of the model represented by the parameter theta
    given the data.

    Parameters
    ----------
    theta: `~numpy.ndarray`
        Parameter vector: (temperature, logscale).
    spectral_axis: `~astropy.units.Quantity`
        Wavelength axis. Should be the same for target_flux and external_flux.
    target_flux: `~numpy.ndarray`
        Total spectrum intensity.
    external_flux: `~numpy.ndarray`
        External spectrum intensity.

    Return
    ------
    loglike: scalar
        Logarithmic likelihood for parameter theta.
    """
    T, alpha, beta, gamma = theta

    prediction = reduced_target_model(
        spectral_axis, external_flux, T, alpha, beta, gamma,
        )

    #target_reduced = target_flux - target_flux.mean()
    diff = target_flux - prediction

    # assume constant noise for every point
    stddev = 50. # diff.std()

    loglike = np.sum(
        -0.5 * np.log(2.0 * np.pi)
        - np.log(stddev)
        - diff ** 2 / (2.0 * stddev ** 2)
    )
    return loglike


def log_likelihood_prior(theta):
    """Prior logarithmic likelihood.

    A priori likelihood for parameter theta. This is used to constrain
    the parameter space, for example: 0 < Temperature < 3000.

    Parameters
    ----------
    theta: `~numpy.ndarray`
        Parameter vector: (temperature, logscale).

    Return
    ------
    loglike: scalar
        A priori logarithmic likelihood for parameter theta.
    """
    T, alpha, beta, gamma = theta

    # Maximum temperature for dust should be lower than 3000 K
    Tok = 0 < T < 3000
    alphaok = alpha > 0
    betaok = beta > 0

    if Tok and alphaok and betaok:
        return 0.0
    else:
        return -np.inf


def log_probability(theta, spectral_axis, target_flux, external_flux):
    """Posterior logarithmic likelihood.

    Compute the likelihood of the model represented by the parameter theta
    given the data and assuming a priori information (priors).

    Parameters
    ----------
    theta: `~numpy.ndarray`
        Parameter vector: (temperature, logscale).
    spectral_axis: `~astropy.units.Quantity`
        Wavelength axis. Should be the same for target_flux and external_flux.
    target_flux: `~numpy.ndarray`
        Total spectrum intensity.
    external_flux: `~numpy.ndarray`
        External spectrum intensity.

    Return
    ------
    loglike: scalar
        Posterior logarithmic likelihood for parameter theta.
    """
    lp = log_likelihood_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + gaussian_log_likelihood(
            theta, spectral_axis, target_flux, external_flux
        )


# ==============================================================================
# RESULT CLASSES
# ==============================================================================


@attr.s(frozen=True)
class NirdustParameter:
    """Doc.

    Attributes
    ----------
    name: str
        Parameter name.
    value: scalar, `~astropy.units.Quantity`
        Expected value for parameter after fitting procedure.
    uncertainty: tuple, `~astropy.units.Quantity`
        Assimetric uncertainties: (lower_uncertainty, higher_uncertainty)
    """

    name = attr.ib()
    value = attr.ib()
    uncertainty = attr.ib()


@attr.s(frozen=True)
class NirdustResults:
    """Create the class NirdustResults.

    Storages the results obtained with NirdustFitter plus the dust spectrum.
    The method nplot() can be called to plot the spectrum and the blackbody
    model obtained in the fitting.

    Attributes
    ----------
    temperature: Parameter
        Parameter object with the expected blackbody temperature and
        its uncertainty.

    scale: Parameter
        Parameter object with the expected blackbody scale and
        its uncertainty. Note: in the fitting procedure the log10(scale) is
        sampled to achieve better convergence. However, here we provide the
        linear value of scale as expected by the astropy BlackBody model. No
        unit is provided as the intensity is in arbitrary units.

    fitted_blackbody: `~astropy.modeling.models.BlackBody`
        BlackBody instance with the best fit values of temperature and scale.

    dust: NirdustSpectrum
        Reconstructed dust emission.
    """

    temperature = attr.ib()
    alpha = attr.ib()
    beta = attr.ib()
    gamma = attr.ib()

    fitted_blackbody = attr.ib()
    target_spectrum = attr.ib()
    external_spectrum = attr.ib()

    def plot(self, ax=None, data_kws=None, model_kws=None):
        """Build a plot of the fitted spectrum and the fitted model.

        Parameters
        ----------
        ax: ``matplotlib.pyplot.Axis`` object
            Object of type Axes containing complete information of the
            properties to generate the image, by default it is None.

        data_kws: ``dict``
            Dictionaries of keyword arguments. Passed to the data plotting
            function.

        model_kws: ``dict``
            Dictionaries of keyword arguments. Passed to the model plotting
            function.

        Return
        ------
        out: ``matplotlib.pyplot.Axis`` :
            The axis where the method draws.
        """
        # bb_fit = self.fitted_blackbody(self.target_spectrum.spectral_axis)

        prediction = reduced_target_model(
            self.target_spectrum.spectral_axis,
            self.external_spectrum.flux.value,
            self.temperature.value,
            self.alpha.value,
            self.beta.value,
            self.gamma.value,
        )
        target_reduced = (
            self.target_spectrum.flux - self.target_spectrum.flux.mean()
        )

        if ax is None:
            ax = plt.gca()

        data_kws = {} if data_kws is None else data_kws
        data_kws.setdefault("color", "firebrick")
        ax.plot(
            self.target_spectrum.spectral_axis,
            target_reduced,
            label="target reduced",
            **data_kws,
        )

        model_kws = {} if model_kws is None else model_kws
        model_kws.setdefault("color", "Navy")
        ax.plot(
            self.target_spectrum.spectral_axis,
            prediction,
            label="prediction",
            **model_kws,
        )
        ax.set_xlabel("Angstrom [A]")
        ax.set_ylabel("Intensity [arbitrary units]")
        ax.legend()

        return ax


# ==============================================================================
# FITTER CLASSES
# ==============================================================================

@attr.s
class BaseFitter(metaclass=abc.ABCMeta):

    target_spectrum = attr.ib(
        validator=validators.instance_of(NirdustSpectrum)
    )
    external_spectrum = attr.ib(
        validator=validators.instance_of(NirdustSpectrum)
    )
    extra_conf = attr.ib(converter=dict)

    _fitted = attr.ib(init=False, default=False)

    # ABSTRACT
    @abc.abstractmethod
    def best_parameters(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def run_model(self, initial_state):
        raise NotImplementedError()

    # DEFINED
    @property
    def ndim_(self):
        return 4

    @property
    def isfitted_(self):
        return self._fitted

    def fit(self, initial_state):
        if self.isfitted_:
            raise RuntimeError("Model already fitted.")

        if initial_state is None:
            initial_state = (1000.0, 1.0, 1.0, 1.0)
        elif len(initial_state) != self.ndim_:
            raise ValueError("Invalid initial state.")

        self.run_model(initial_state)
        self._fitted = True
        return self

    def result(self, **kwargs):
        """Get the chain array.
        Parameters
        ----------
        kwargs:
            Parameters for ``best_parameters()``.
        Return
        ------
        result: NirdustResult
            Results of the fitting procedure.
        """
        temp, alpha, beta, gamma = self.best_parameters(**kwargs)

        result = NirdustResults(
            temperature=temp,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            fitted_blackbody=BlackBody(temp.value),
            target_spectrum=self.target_spectrum,
            external_spectrum=self.external_spectrum,
        )
        return result


@attr.s
class EMCEENirdustFitter(BaseFitter):
    """Emcee fitter class.

    Fit a BlackBody model to the data using Markov Chain Monte Carlo (MCMC)
    sampling of the parameter space using the emcee implementation.

    Attributes
    ----------
    target_spectrum: NirdustSpectrum object
        Instance of NirdustSpectrum containing the nuclear spectrum.

    external_spectrum: NirdustSpectrum object
        Instance of NirdustSpectrum containing the external spectrum.

    seed: int or None
        Seed for random number generation.

    sampler: ``emcee.EnsembleSampler``
        Sampler instance to run the MCMC model.

    """
    nwalkers = attr.ib(default=11, validator=validators.instance_of(int))
    seed = attr.ib(default=None, validator=validators.optional(validators.instance_of(int)))
    steps = attr.ib(default=1000, validator=validators.instance_of(int))

    sampler_ = attr.ib(init=False)

    @sampler_.default
    def _sampler__default(self):
        model_args = (
            self.target_spectrum.spectral_axis,
            self.target_spectrum.flux.value,
            self.external_spectrum.flux.value,
        )
        return emcee.EnsembleSampler(
            nwalkers=self.nwalkers,
            ndim=self.ndim_,
            log_prob_fn=log_probability,
            args=model_args,
            **self.extra_conf
        )

    def chain(self, discard=0):
        """Get the chain array.

        Parameters
        ----------
        discard: int
            Number of steps to discard from the chain, counting from the
            begining.

        Return
        ------
        chain: `~numpy.ndarray`
            Array with sampled parameters.
        """
        return self.sampler_.get_chain(discard=discard).copy()

    # redefinitions

    def run_model(self, initial_state):
        # acomodate initial values
        rng = np.random.default_rng(seed=self.seed)
        p0 = rng.random((self.nwalkers, self.ndim_))
        p0 += np.asarray(initial_state)

        self.sampler_.run_mcmc(p0, self.steps)

    def best_parameters(self, discard=0):
        """Marginalize parameter distributions.

        Parameters
        ----------
        discard: int
            Number of chain steps to discard before marginalizing.

        Return
        ------
        temperature: Parameter
            Parameter object with expected temperature and its uncertainty.
        scale: Parameter
            Parameter object with expected scale and its uncertainty.
            Note: in the fitting procedure the log10(scale) is sampled to
            achieve better convergence. However, here we provide the linear
            value of scale as expected by the astropy BlackBody model. No unit
            is provided as the intensity is in arbitrary units.
        """
        chain = self.chain(discard=discard).reshape((-1, self.ndim_))
        # chain[:, 1] = 10 ** chain[:, 1]

        # median, lower_error, upper_error
        values = [50, 16, 84]
        t, t_low, t_up = np.percentile(chain[:, 0], values) * u.K
        a, a_low, a_up = np.percentile(chain[:, 1], values)  # u arbitrary
        b, b_low, b_up = np.percentile(chain[:, 2], values)
        g, g_low, g_up = np.percentile(chain[:, 3], values)

        temp = NirdustParameter("Temperature", t, (t - t_low, t_up - t))
        alpha = NirdustParameter("Alpha", a, (a - a_low, a_up - a))
        beta = NirdustParameter("Beta", b, (b - b_low, b_up - b))
        gamma = NirdustParameter("Gamma", g, (g - g_low, g_up - g))
        return temp, alpha, beta, gamma

    def plot(
        self,
        discard=0,
        ax=None,
        temp_kws=None,
        temp_mean_kws=None,
        log_kws=None,
        log_mean_kws=None,
    ):
        """Get the chain array.

        Parameters
        ----------
        discard: int
            Number of steps to discard from the chain, counting from the
            begining.

        ax: ``matplotlib.pyplot.Axis`` object
            Object of type Axes containing complete information of the
            properties to generate the image, by default it is None.

        temp_kws: ``dict``
            Dictionaries of keyword arguments. Passed to the temperature
            plotting function.
        temp_mean_kws: ``dict``
            Dictionaries of keyword arguments. Passed to the temperature mean
            plotting function.
        log_kws: ``dict``
            Dictionaries of keyword arguments. Passed to the logarithmic
            plotting function.
        log_mean_kws: ``dict``
            Dictionaries of keyword arguments. Passed to the logarithmic mean
            plotting function.

        Return
        ------
        out: ``matplotlib.pyplot.Axis`` :
            The axis where the method draws.
        """
        if not self.isfitted_:
            raise RuntimeError("The model is not fitted.")

        # axis orchestration
        if ax is None:
            _, ax = plt.subplots(4, 1, sharex=True, figsize=(8, 6))

        ax_t, ax_alpha, ax_beta, ax_gamma = ax
        fig = ax_t.get_figure()
        fig.subplots_adjust(hspace=0)

        chain = self.chain(discard=discard)

        arr_t = chain[:, :, 0]
        mean_t = arr_t.mean(axis=1)

        arr_alpha = chain[:, :, 1]
        mean_alpha = arr_alpha.mean(axis=1)

        arr_beta = chain[:, :, 2]
        mean_beta = arr_beta.mean(axis=1)

        arr_gamma = chain[:, :, 3]
        mean_gamma = arr_gamma.mean(axis=1)

        # title
        ax_t.set_title(
            f"Sampled parameters\n Steps={self.steps} - Discarded={discard}"
        )

        temp_kws = {} if temp_kws is None else temp_kws
        temp_kws.setdefault("alpha", 0.5)
        ax_t.plot(arr_t, **temp_kws)

        # temp mean
        temp_mean_kws = {} if temp_mean_kws is None else temp_mean_kws
        temp_mean_kws.setdefault("color", "k")
        ax_t.plot(mean_t, label="Mean", **temp_mean_kws)

        # temp labels
        ax_t.set_ylabel("T")

        log_kws = {} if log_kws is None else log_kws
        log_kws.setdefault("alpha", 0.5)
        ax_alpha.plot(arr_alpha, **log_kws)
        ax_beta.plot(arr_beta, **log_kws)
        ax_gamma.plot(arr_gamma, **log_kws)

        # alpha,beta mean
        log_mean_kws = {} if log_mean_kws is None else log_mean_kws
        log_mean_kws.setdefault("color", "k")
        ax_alpha.plot(mean_alpha, label="Mean", **log_mean_kws)
        ax_beta.plot(mean_beta, label="Mean", **log_mean_kws)
        ax_gamma.plot(mean_gamma, label="Mean", **log_mean_kws)

        # alpha,beta labels
        ax_alpha.set_ylabel("alpha")
        ax_alpha.set_xlabel("Steps")
        ax_alpha.legend()

        ax_beta.set_ylabel("beta")
        ax_beta.set_xlabel("Steps")
        ax_beta.legend()

        ax_gamma.set_ylabel("gamma")
        ax_gamma.set_xlabel("Steps")
        ax_gamma.legend()
        return ax




@attr.s
class AstropyNirdustFitter(BaseFitter):
    """Astropy fitter class.

    Fit a BlackBody model to the data using Astropy modeling methods.

    Attributes
    ----------
    target_spectrum: NirdustSpectrum object
        Instance of NirdustSpectrum containing the nuclear spectrum.

    external_spectrum: NirdustSpectrum object
        Instance of NirdustSpectrum containing the external spectrum.

    seed: int or None
        Seed for random number generation.

    """

    calc_uncertainties = attr.ib(default=True, converter=bool)

    fitter_ = attr.ib(init=False)

    _fitted_model = attr.ib(default=None, init=False)

    @fitter_.default
    def _fitter__default(self):
        return LevMarLSQFitter(self.calc_uncertainties)

    @property
    def fitted_model(self):
        return self._fitted_model
    
    @property
    def isfitted_(self):
        return self.fitted_model is not None

    def best_parameters(self):
        """doc"""
        t = self.fitted_model.T
        a = self.fitted_model.alpha
        b = self.fitted_model.beta
        g = self.fitted_model.gamma

        temp = NirdustParameter("Temperature", t.value * u.K, t.std)
        alpha = NirdustParameter("Alpha", a.value, a.std)
        beta = NirdustParameter("Beta", b.value, b.std)
        gamma = NirdustParameter("Gamma", g.value, g.std)
        return temp, alpha, beta, gamma

    def run_model(self, initial_state):

        # Compute model
        target_flux = self.target_spectrum.flux
        target_reduced = target_flux - target_flux.mean()

        rtm = ReducedTargetModel(
            self.external_spectrum.flux.value, *initial_state
            )
        fmodel = self.fitter_(
            rtm,
            self.target_spectrum.frequency_axis.value,
            target_reduced.value,
            weights=None,   # noise goes here: weights=1/sigma
            **self.extra_conf,
            )

        self._fitted_model = fmodel



# ==============================================================================
# FITTER FUNCTION WRAPPER
# ==============================================================================

FITTER_BACKENDS = {
    "astropy": AstropyNirdustFitter,
    "emcee": EMCEENirdustFitter,
}

def fit_blackbody(
    target_spectrum,
    external_spectrum,
    initial_state=None,
    backend="emcee",
    **kwargs,
):
    """Fitter function.

    Fit a BlackBody model to the data using Markov Chain Monte Carlo (MCMC)
    sampling of the parameter space using the emcee implementation.
    This function serves as a wrapper around the NirdustFitter class.

    Parameters
    ----------
    target_spectrum: NirdustSpectrum object
        Instance of NirdustSpectrum containing the nuclear spectrum.

    external_spectrum: NirdustSpectrum object
        Instance of NirdustSpectrum containing the external spectrum.

    initial_state: tuple, optional
        Vector indicating the initial guess values of temperature and
        log10(scale). Default: (1000.0 K, 8.0)

    steps: int, optional
        Number of times the parameter space is be sampled. Default: 1000.

    kwargs: dict
        Parameters to be passed to the emcee.EnsembleSampler class.

    Return
    ------
    fitter: NirdustFitter object
        Instance of NirdustFitter after the fitting procedure.
    """
    fcls = FITTER_BACKENDS[backend.lower()]

    # divide kwargs in two parts
    instance_kws = {}  # attributes of the fitter class
    extra_conf = {}  # extra configuration

    attrs_fields = set(attr.fields_dict(fcls).keys())
    for k, v in kwargs.items():
        if k in attrs_fields:
            instance_kws[k] = v
        else:
            extra_conf[k] = v

    fitter = fcls(
        target_spectrum = target_spectrum,
        external_spectrum = external_spectrum,
        extra_conf = extra_conf,
        **instance_kws)

    fitter.fit(initial_state=initial_state)
    return fitter
