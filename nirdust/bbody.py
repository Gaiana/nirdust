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


from astropy import constants as const
from astropy import units as u
from astropy.modeling.models import BlackBody

import attr

import matplotlib.pyplot as plt

import numpy as np

import emcee


def blackbody(nu, temperature, scale):
    """Normalized blackbody model.
    This class is similar to the BlackBody model provided by Astropy except
    that the 'scale' parameter is eliminated, i.e. is allways equal to 1.
    The normalization is performed by dividing the blackbody flux by its
    numerical mean. The result is a dimensionless Quantity.
    Parameters
    ----------
    temperature: float, `~numpy.ndarray`, or `~astropy.units.Quantity`
        Temperature of the blackbody.
    Notes
    -----
    The Astropy BlackBody model can not be used directly to obtain a
    normalized black body since the 'scale' parameter is not known a priori.
    The scaling value (the mean in this case) directly depends on the
    black body value.
    """

    # Convert to units for calculations, also force double precision
    with u.add_enabled_equivalencies(u.spectral() + u.temperature()):
        freq = u.Quantity(nu, u.Hz, dtype=np.float64)
        temp = u.Quantity(temperature, u.K)

    # check the units of scale and setup the output units
    bb_unit = u.erg / (u.cm ** 2 * u.s * u.Hz * u.sr)  # default unit
    # use the scale that was used at initialization for determining
    # the units to return to support returning the right units when
    # fitting where units are stripped
    if hasattr(scale, "unit") and scale.unit is not None:
        # check that the units on scale are covertable to surface
        # brightness units
        if not scale.unit.is_equivalent(bb_unit, u.spectral_density(nu)):
            raise ValueError(
                f"scale units not surface brightness: {scale.unit}"
            )
        # use the scale passed to get the value for scaling
        if hasattr(scale, "unit"):
            mult_scale = scale.value
        else:
            mult_scale = scale
        bb_unit = scale.unit
    else:
        mult_scale = scale

    # Check if input values are physically possible
    if np.any(temp < 0):
        raise ValueError(f"Temperature should be positive: {temp}")

    log_boltz = const.h * freq / (const.k_B * temp)
    boltzm1 = np.expm1(log_boltz)

    # Calculate blackbody flux
    bb_nu = 2.0 * const.h * freq ** 3 / (const.c ** 2 * boltzm1) / u.sr
    y = mult_scale * bb_nu.to(bb_unit, u.spectral_density(freq))

    # If the temperature parameter has no unit, we should return a unitless
    # value. This occurs for instance during fitting, since we drop the
    # units temporarily.
    if hasattr(temperature, "unit"):
        return y
    return y.value


# ==============================================================================
# CLASSES
# ==============================================================================


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

    temperature = attr.ib(converter=lambda t: u.Quantity(t, u.K))
    info = attr.ib()
    uncertainty = attr.ib()
    fitted_blackbody = attr.ib()
    freq_axis = attr.ib(repr=False)
    flux_axis = attr.ib(repr=False)

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
        instance = self.fitted_blackbody(self.freq_axis.value)
        if ax is None:
            ax = plt.gca()

        ax.plot(
            self.freq_axis, self.flux_axis, color=data_color, label="continuum"
        )
        ax.plot(self.freq_axis, instance, color=model_color, label="model")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Normalized Energy [arbitrary units]")
        ax.legend()

        return ax


# ==============================================================================
# EMCEE FUNCTIONS
# ==============================================================================

# probability of the data given the model
def gaussian_log_likelihood(theta, nu, total_flux, external_flux, stddev):
    T, logscale = theta
    scale = 10 ** logscale

    # calculate the model
    B = blackbody(nu, T, scale)
    fB = B.mean()
    fO = total_flux.mean()
    fX = external_flux.mean()
    data = total_flux - external_flux * (fO - fB) / fX

    diff = data - B
    if stddev is None:
        stddev = np.full_like(nu, diff.std(ddof=1))

    loglike = np.sum(
        -0.5 * np.log(2.0 * np.pi)
        - np.log(stddev)
        - diff ** 2 / (2.0 * stddev ** 2)
    )
    return loglike


# uninformative prior
def log_likelihood_prior(theta):
    T, logscale = theta

    Tok = 1 < T < 2500
    logscaleok = -2 <= logscale < 12

    if Tok and logscaleok:
        return 0.0
    else:
        return -np.inf


# posterior probability
def log_probability(theta, nu, total_flux, external_flux, stddev):
    lp = log_likelihood_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + gaussian_log_likelihood(
            theta, nu, total_flux, external_flux, stddev
        )


# ==============================================================================
# FITTER CLASS
# ==============================================================================


def _percentiles(chain, per):
    values = map(
        lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
        zip(*np.percentile(chain, per, axis=0)),
    )
    return list(values)


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
            self.total.frequency_axis.value,
            self.total.flux.value,
            self.external.flux.value,
            None,
        )
        self.sampler = emcee.EnsembleSampler(
            self.nwalkers,
            self.ndim_,
            log_probability,
            args=fargs,
            threads=self.nthreads,
        )

    def run(self, initial_state=None, steps=1000):

        if initial_state is None:
            initial_state = [1000.0, 8.0]
        elif isinstance(initial_state, emcee.State):
            return self.sampler.run_mcmc(initial_state, steps)
        elif len(initial_state) != 2:
            raise ValueError("Invalid initial state.")

        rng = np.random.default_rng(seed=self.seed)
        p0 = rng.random((self.nwalkers, self.ndim_))
        p0[:, 0] += initial_state[0]
        p0[:, 1] += initial_state[1]

        self.steps_ = steps
        return self.sampler.run_mcmc(p0, steps)

    def chain(self, discard=0):
        return self.sampler.get_chain(discard=discard)

    def best_fit(self, discard=0):
        chain = self.chain(discard=discard)
        fit = map(
            lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
            zip(
                *np.percentile(
                    chain.reshape((-1, self.ndim_)), [16, 50, 84], axis=0
                )
            ),
        )
        return list(fit)

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
