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


from astropy import units as u
from astropy.modeling.models import BlackBody

import attr
from attr import validators

import emcee

import matplotlib.pyplot as plt

import numpy as np

from .core import NirdustSpectrum


# ==============================================================================
# EMCEE FUNCTIONS
# ==============================================================================


def target_model(spectral_axis, external_flux, T, alpha, beta):
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

    bb_reduced = bb_flux - bb_flux.mean()
    external_reduced = external_flux - external_flux.mean()
    
    prediction = alpha * external_reduced + beta * bb_reduced
    return prediction


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
    T, alpha, beta = theta

    prediction = target_model(spectral_axis, external_flux, T, alpha, beta)
    
    target_reduced = target_flux - target_flux.mean()
    diff = target_reduced - prediction

    # assume constant noise for every point
    stddev = np.full_like(diff, diff.std(ddof=1))

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
    T, alpha, beta = theta

    # Maximum temperature for dust should be lower than 3000 K
    Tok = 0 < T < 3000
    alphaok = alpha > 0
    betaok = beta > 0

    if Tok:# and alphaok and betaok:
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
class Parameter:
    """Doc.

    Attributes
    ----------
    name: str
        Parameter name.
    mean: scalar, `~astropy.units.Quantity`
        Expected value for parameter after fitting procedure.
    uncertainty: tuple, `~astropy.units.Quantity`
        Assimetric uncertainties: (lower_uncertainty, higher_uncertainty)
    """

    name = attr.ib()
    mean = attr.ib()
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
        bb_fit = self.fitted_blackbody(self.target_spectrum.spectral_axis)

        prediction = target_model(
            self.target_spectrum.spectral_axis,
            self.external_spectrum.flux.value,
            self.temperature.mean,
            self.alpha.mean,
            self.beta.mean,
        )
        target_reduced = self.target_spectrum.flux - self.target_spectrum.flux.mean()

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
            self.target_spectrum.spectral_axis, prediction,
            label="prediction",
            **model_kws
        )
        ax.set_xlabel("Angstrom [A]")
        ax.set_ylabel("Intensity [arbitrary units]")
        ax.legend()

        return ax


# ==============================================================================
# FITTER CLASS
# ==============================================================================


@attr.s
class NirdustFitter:
    """Fitter class.

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

    target_spectrum = attr.ib(
        validator=validators.instance_of(NirdustSpectrum)
    )
    external_spectrum = attr.ib(
        validator=validators.instance_of(NirdustSpectrum)
    )
    seed = attr.ib(
        validator=validators.optional(attr.validators.instance_of(int))
    )
    sampler = attr.ib(validator=validators.instance_of(emcee.EnsembleSampler))
    steps_ = attr.ib(init=False, default=None)

    @classmethod
    def from_params(
        cls,
        *,
        target_spectrum,
        external_spectrum,
        nwalkers=11,
        seed=None,
        **kwargs,
    ):
        """Create NirdustFitter object from keyword parameters.

        Parameters
        ----------
        target_spectrum: NirdustSpectrum object
            Instance of NirdustSpectrum containing the nuclear spectrum.

        external_spectrum: NirdustSpectrum object
            Instance of NirdustSpectrum containing the external spectrum.

        nwalkers: int
            Number of walkers. Must be higher than 4 (twice the number of free
            parameters). Default: 11.

        seed: int
            Seed for random number generation. Defaul: None

        kwargs:
            Parameters to be passed to the emcee.EnsembleSampler class.
        """
        sampler_args = (
            target_spectrum.spectral_axis,
            target_spectrum.flux.value,
            external_spectrum.flux.value,
        )
        sampler = emcee.EnsembleSampler(
            nwalkers,
            3,
            log_probability,
            args=sampler_args,
            **kwargs,
        )
        return cls(
            target_spectrum=target_spectrum,
            external_spectrum=external_spectrum,
            seed=seed,
            sampler=sampler,
        )

    @property
    def nwalkers(self):
        """Get number of walkers."""
        return self.sampler.nwalkers

    @property
    def isfitted_(self):
        """Check if model has already been fitted."""
        return self.steps_ is not None

    @property
    def ndim_(self):
        """Get number of dimensions to sample. Always 2."""
        return 3

    def marginalize_parameters(self, discard=0):
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
        #chain[:, 1] = 10 ** chain[:, 1]

        # median, lower_error, upper_error
        values = [50, 16, 84]
        t_mean, t_low, t_up = np.percentile(chain[:, 0], values) * u.K
        a_mean, a_low, a_up = np.percentile(chain[:, 1], values)  # u arbitrary
        b_mean, b_low, b_up = np.percentile(chain[:, 2], values)

        temp = Parameter(
            "Temperature", t_mean, (t_mean - t_low, t_up - t_mean)
        )
        alpha = Parameter("Alpha", a_mean, (a_mean - a_low, a_up - a_mean))
        beta = Parameter("Beta", b_mean, (b_mean - b_low, b_up - b_mean))
        return temp, alpha, beta

    def fit(self, initial_state=None, steps=1000):
        """Run MCMC sampler.

        Parameters
        ----------
        initial_state: tuple, optional
            Vector indicating the initial guess values of temperature and
            log10(scale). Default: (1000.0 K, 8.0)
        steps: int, optional
            Number of times the parameter space is be sampled. Default: 1000.

        Return
        ------
        self: NirdustFitter
            New instance of the fitter.
        """
        if self.isfitted_:
            raise RuntimeError("Model already fitted.")

        if initial_state is None:
            initial_state = (1000.0, 1.0, 1.0)

        elif len(initial_state) != 2:
            raise ValueError("Invalid initial state.")

        # acomodate initial values
        rng = np.random.default_rng(seed=self.seed)
        p0 = rng.random((self.nwalkers, self.ndim_))
        p0 += np.asarray(initial_state)

        self.steps_ = steps
        self.sampler.run_mcmc(p0, steps)
        return self

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
        return self.sampler.get_chain(discard=discard).copy()

    def result(self, discard=0):
        """Get the chain array.

        Parameters
        ----------
        discard: int
            Number of steps to discard from the chain, counting from the
            begining.

        Return
        ------
        result: NirdustResult
            Results of the fitting procedure.
        """
        temp, alpha, beta = self.marginalize_parameters(discard=discard)

        result = NirdustResults(
            temperature=temp,
            alpha=alpha,
            beta=beta,
            fitted_blackbody=BlackBody(temp.mean),
            target_spectrum=self.target_spectrum,
            external_spectrum=self.external_spectrum,
        )
        return result

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
            _, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 6))

        ax_t, ax_alpha, ax_beta = ax
        fig = ax_t.get_figure()
        fig.subplots_adjust(hspace=0)

        # data ==========================================================
        chain = self.chain(discard=discard)

        arr_t = chain[:, :, 0]
        mean_t = arr_t.mean(axis=1)

        arr_alpha = chain[:, :, 1]
        mean_alpha = arr_alpha.mean(axis=1)

        arr_beta = chain[:, :, 2]
        mean_beta = arr_beta.mean(axis=1)

        # title
        ax_t.set_title(
            f"Sampled parameters\n Steps={self.steps_} - Discarded={discard}"
        )

        # temp ==========================================================
        temp_kws = {} if temp_kws is None else temp_kws
        temp_kws.setdefault("alpha", 0.5)
        ax_t.plot(arr_t, **temp_kws)

        # temp mean
        temp_mean_kws = {} if temp_mean_kws is None else temp_mean_kws
        temp_mean_kws.setdefault("color", "k")
        ax_t.plot(mean_t, label="Mean", **temp_mean_kws)

        # temp labels
        ax_t.set_ylabel("T")

        # log ==========================================================
        log_kws = {} if log_kws is None else log_kws
        log_kws.setdefault("alpha", 0.5)
        ax_alpha.plot(arr_alpha, **log_kws)
        ax_beta.plot(arr_beta, **log_kws)

        # log mean
        log_mean_kws = {} if log_mean_kws is None else log_mean_kws
        log_mean_kws.setdefault("color", "k")
        ax_alpha.plot(mean_alpha, label="Mean", **log_mean_kws)
        ax_beta.plot(mean_beta, label="Mean", **log_mean_kws)

        # log labels
        ax_alpha.set_ylabel("alpha")
        ax_alpha.set_xlabel("Steps")
        ax_alpha.legend()
        ax_beta.set_ylabel("beta")
        ax_beta.set_xlabel("Steps")
        ax_beta.legend()

        return ax


# ==============================================================================
# FITTER FUNCTION WRAPPER
# ==============================================================================


def fit_blackbody(
    target_spectrum,
    external_spectrum,
    initial_state=None,
    steps=1000,
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
    fitter = NirdustFitter.from_params(
        target_spectrum=target_spectrum,
        external_spectrum=external_spectrum,
        **kwargs,
    )
    fitter.fit(initial_state=initial_state, steps=steps)
    return fitter
