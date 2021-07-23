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


def dust_component(blackbody_flux, target_flux, external_flux):
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
    fB = blackbody_flux.mean()
    fO = target_flux.mean()
    fX = external_flux.mean()
    dust_flux = target_flux - external_flux * (fO - fB) / fX
    return dust_flux


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
    T, logscale = theta
    scale = 10 ** logscale

    # calculate the model
    blackbody = BlackBody(u.Quantity(T, u.K), scale)
    bb_flux = blackbody(spectral_axis).value
    dust = dust_component(bb_flux, target_flux, external_flux)
    diff = dust - bb_flux

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
    T, logscale = theta

    # Maximum temperature for dust should be lower than 3000 K
    Tok = 0 < T < 3000
    logscaleok = -2 <= logscale < 12

    if Tok and logscaleok:
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

    Atributes
    ---------
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
    scale = attr.ib()
    fitted_blackbody = attr.ib()
    dust = attr.ib(repr=False)

    def nplot(self, ax=None, data_color="firebrick", model_color="navy"):
        """Build a plot of the fitted spectrum and the fitted model.

        Parameters
        ----------
        ax: ``matplotlib.pyplot.Axis`` object
            Object of type Axes containing complete information of the
            properties to generate the image, by default it is None.

        data_color: str
            The color in wich the spectrum must be plotted, default is
            "firebrick".

        model_color: str
            The color in wich the fitted black body must be plotted, default
            if "navy".

        Return
        ------
        out: ``matplotlib.pyplot.Axis`` :
            The axis where the method draws.
        """
        bb_fit = self.fitted_blackbody(self.dust.spectral_axis)
        if ax is None:
            ax = plt.gca()

        ax.plot(
            self.dust.spectral_axis,
            self.dust.flux,
            color=data_color,
            label="continuum",
        )
        ax.plot(
            self.dust.spectral_axis, bb_fit, color=model_color, label="model"
        )
        ax.set_xlabel("Angstrom [A]")
        ax.set_ylabel("Normalized Energy [arbitrary units]")
        ax.legend()

        return ax


# ==============================================================================
# FITTER CLASS
# ==============================================================================


@attr.s(auto_detect=True)
class NirdustFitter:
    """Fitter class.

    Fit a BlackBody model to the data using Markov Chain Monte Carlo (MCMC)
    sampling of the parameter space using the emcee implementation.

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

    target_spectrum = attr.ib(
        validator=validators.instance_of(NirdustSpectrum)
    )
    external_spectrum = attr.ib(
        validator=validators.instance_of(NirdustSpectrum)
    )
    nwalkers = attr.ib()
    seed = attr.ib()
    kwargs = attr.ib(factory=dict)

    ndim_ = attr.ib(init=False, default=2)
    steps_ = attr.ib(init=False, default=None)

    def __init__(
        self,
        target_spectrum,
        external_spectrum,
        nwalkers=11,
        seed=None,
        **kwargs,
    ):
        """Redefine attrs init to accept kwargs for emcee config."""
        self.__attrs_init__(
            target_spectrum, external_spectrum, nwalkers, seed, kwargs
        )

    def __attrs_post_init__(self):
        """Instantiate the sampler."""
        fargs = (
            self.target_spectrum.spectral_axis,
            self.target_spectrum.flux.value,
            self.external_spectrum.flux.value,
        )
        self.sampler = emcee.EnsembleSampler(
            self.nwalkers,
            self.ndim_,
            log_probability,
            args=fargs,
            **self.kwargs,
        )

    @property
    def isfitted(self):
        """Check if model has already been fitted."""
        return self.steps_ is not None

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
        chain[:, 1] = 10 ** chain[:, 1]

        # median, lower_error, upper_error
        values = [50, 16, 84]
        t_mean, t_low, t_up = np.percentile(chain[:, 0], values) * u.K
        s_mean, s_low, s_up = np.percentile(chain[:, 1], values)  # u arbitrary

        temp = Parameter(
            "Temperature", t_mean, (t_mean - t_low, t_up - t_mean)
        )
        scale = Parameter("Scale", s_mean, (s_mean - s_low, s_up - s_mean))
        return temp, scale

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
        if self.isfitted:
            raise RuntimeError("Model already fitted.")

        if initial_state is None:
            initial_state = (1000.0, 8.0)

        elif len(initial_state) != 2:
            raise ValueError("Invalid initial state.")

        rng = np.random.default_rng(seed=self.seed)
        p0 = rng.random((self.nwalkers, self.ndim_))
        p0[:, 0] += initial_state[0]
        p0[:, 1] += initial_state[1]

        self.steps_ = steps
        self.sampler.run_mcmc(p0, steps)
        return self

    def chain(self, discard=0):
        """Get the chain array.

        Parameter
        ---------
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

        Parameter
        ---------
        discard: int
            Number of steps to discard from the chain, counting from the
            begining.

        Return
        ------
        result: NirdustResult
            Results of the fitting procedure.
        """
        temp, scale = self.marginalize_parameters(discard=discard)

        bb_model = BlackBody(temp.mean, scale.mean)
        dust = dust_component(
            bb_model(self.target_spectrum.spectral_axis).value,
            self.target_spectrum.flux.value,
            self.external_spectrum.flux.value,
        )
        result = NirdustResults(
            temperature=temp,
            scale=scale,
            fitted_blackbody=bb_model,
            dust=NirdustSpectrum(self.target_spectrum.spectral_axis, dust),
        )
        return result

    def plot(self, discard=0, ax=None):
        """Get the chain array.

        Parameter
        ---------
        discard: int
            Number of steps to discard from the chain, counting from the
            begining.

        ax: ``matplotlib.pyplot.Axis`` object
            Object of type Axes containing complete information of the
            properties to generate the image, by default it is None.

        Return
        ------
        out: ``matplotlib.pyplot.Axis`` :
            The axis where the method draws.
        """
        # axis orchestration
        if ax is None:
            _, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        ax_t, ax_log = ax
        fig = ax_t.get_figure()
        fig.subplots_adjust(hspace=0)

        # data
        chain = self.chain(discard=discard)

        arr_t = chain[:, :, 0]
        mean_t = arr_t.mean(axis=1)

        arr_log = chain[:, :, 1]
        mean_log = arr_log.mean(axis=1)

        # plot
        ax_t.set_title(
            f"Sampled parameters\n Steps={self.steps_} - Discarded={discard}"
        )

        ax_t.plot(arr_t, alpha=0.5)
        ax_t.plot(mean_t, color="k", label="Mean")
        ax_t.set_ylabel("T")

        ax_log.plot(arr_log, alpha=0.5)
        ax_log.plot(mean_log, color="k", label="Mean")
        ax_log.set_ylabel("log(scale)")
        ax_log.set_xlabel("Steps")
        ax_log.legend()

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
    fitter = NirdustFitter(
        target_spectrum=target_spectrum,
        extenal_spectrum=external_spectrum,
        **kwargs,
    )
    fitter.fit(initial_state=initial_state, steps=steps)
    return fitter
