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

import emcee

import matplotlib.pyplot as plt

import numpy as np

from .core import NirdustSpectrum


# ==============================================================================
# EMCEE FUNCTIONS
# ==============================================================================


def dust_component(bb_flux, total_flux, external_flux):
    """Documentar."""
    fB = bb_flux.mean()
    fO = total_flux.mean()
    fX = external_flux.mean()
    data = total_flux - external_flux * (fO - fB) / fX
    return data


# probability of the data given the model
def gaussian_log_likelihood(theta, spectral_axis, total_flux, external_flux):
    """Documentar."""
    T, logscale = theta
    scale = 10 ** logscale

    # calculate the model
    blackbody = BlackBody(u.Quantity(T, u.K), scale)
    bb_flux = blackbody(spectral_axis).value
    dust = dust_component(bb_flux, total_flux, external_flux)
    diff = dust - bb_flux

    # assume constant noise for every point
    stddev = np.full_like(diff, diff.std(ddof=1))

    loglike = np.sum(
        -0.5 * np.log(2.0 * np.pi)
        - np.log(stddev)
        - diff ** 2 / (2.0 * stddev ** 2)
    )
    return loglike


# uninformative prior
def log_likelihood_prior(theta):
    """Documentar."""
    T, logscale = theta

    # Maximum temperature for dust should be lower than 3000 K
    # poner citas a esos numeros
    Tok = 0 < T < 3000
    logscaleok = -2 <= logscale < 12

    if Tok and logscaleok:
        return 0.0
    else:
        return -np.inf


# posterior probability
def log_probability(theta, spectral_axis, total_flux, external_flux):
    """Documentar."""
    lp = log_likelihood_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + gaussian_log_likelihood(
            theta, spectral_axis, total_flux, external_flux
        )


# ==============================================================================
# RESULT CLASSES
# ==============================================================================


@attr.s(frozen=True)
class Parameter:
    name = attr.ib()
    mean = attr.ib()
    uncertainty = attr.ib()


@attr.s(frozen=True)
class NirdustResults:
    """Create the class NirdustResults.

    Storages the results obtained with fit_blackbody plus the spectral and flux
    axis of the fitted spectrum. The method nplot() can be called to plot the
    spectrum and the blackbody model obtained in the fitting.

    Attributes
    ----------
    temperature: Quantity
        The temperature obtainted in the best black body fit in Kelvin.

    info: dict
        The fit_info dictionary contains the values returned by
        scipy.optimize.leastsq for the most recent fit, including the values
        from the infodict dictionary it returns. See the scipy.optimize.leastsq
        documentation for details on the meaning of these values.

    uncertainty: scalar
        The uncertainty in the temparture fit as calculed by LevMarLSQFitter.

    fitted_blackbody: model
        The normalized_blackbody model for the best fit.

    freq_axis: SpectralAxis object
        The axis containing the spectral information of the spectrum in
        units of Hz.

    flux_axis: Quantity
        The flux of the spectrum in arbitrary units.
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

        Returns
        -------
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


@attr.s
class NirdustFitter:

    total = attr.ib()
    external = attr.ib()
    nwalkers = attr.ib(default=11)
    nthreads = attr.ib(default=1)
    seed = attr.ib(default=None)
    ndim_ = attr.ib(init=False, default=2)
    steps_ = attr.ib(init=False, default=None)

    def __attrs_post_init__(self):
        fargs = (
            self.total.spectral_axis,
            self.total.flux.value,
            self.external.flux.value,
        )
        self.sampler = emcee.EnsembleSampler(
            self.nwalkers,
            self.ndim_,
            log_probability,
            args=fargs,
            threads=self.nthreads,
        )

    def marginalize_parameters(self, discard=0):
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

        if self.steps_ is not None:
            raise RuntimeError("Model already fitted.")

        if initial_state is None:
            initial_state = [1000.0, 8.0]
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
        return self.sampler.get_chain(discard=discard).copy()

    def result(self, discard=0):
        temp, scale = self.marginalize_parameters(discard=discard)

        bb_model = BlackBody(temp.mean, scale.mean)
        dust = dust_component(
            bb_model(self.total.spectral_axis).value,
            self.total.flux.value,
            self.external.flux.value,
        )
        result = NirdustResults(
            temperature=temp,
            scale=scale,
            fitted_blackbody=bb_model,
            dust=NirdustSpectrum(self.total.spectral_axis, dust),
        )
        return result

    def plot(self, discard=0, ax=None):

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
