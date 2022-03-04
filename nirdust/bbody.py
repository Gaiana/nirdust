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
from astropy.modeling.models import BlackBody

import attr
from attr import validators

import matplotlib.pyplot as plt

import numpy as np

from scipy.optimize import basinhopping

from .core import NirdustSpectrum

# ==============================================================================
# EXCEPTIONS
# ==============================================================================


class InvalidBackendError(KeyError):
    """Raised when an invalid backend is requested."""

    pass


# ==============================================================================
# TARGET SPECTRUM MODEL
# ==============================================================================


def target_model(spectral_axis, external_flux, T, alpha, beta, gamma):
    """Compute the expected spectrum given a blackbody prediction.

    Parameters
    ----------
    spectral_axis: `~astropy.units.Quantity`
        Wavelength axis for BlackBody evaluation. Should be the same for
        external_flux.
    external_flux: `~numpy.ndarray`
        External spectrum intensity.
    T: float
        BlackBody temperature in Kelvin.
    alpha: float
        Multiplicative coefficient for external_flux.
    beta: float
        Multiplicative coefficient for blackbody.
    gamma: float
        Additive coefficient.

    Return
    ------
    prediction: `~numpy.ndarray`
        Expected flux given the input parameters.
    """
    # calculate the model
    blackbody = BlackBody(u.Quantity(T, u.K))
    bb_flux = blackbody(spectral_axis).value

    prediction = alpha * external_flux + beta * bb_flux + gamma
    return prediction


# ==============================================================================
# LIKELIHOOD FUNCTIONS
# ==============================================================================


def gaussian_log_likelihood(
    theta, spectral_axis, target_flux, external_flux, noise
):
    """Gaussian logarithmic likelihood.

    Compute the likelihood of the model represented by the parameter theta
    given the data.

    Parameters
    ----------
    theta: `~numpy.ndarray`
        Parameter vector: (temperature, alpha, beta, gamma).
    spectral_axis: `~astropy.units.Quantity`
        Wavelength axis. Should be the same for target_flux and external_flux.
    target_flux: `~numpy.ndarray`
        Total spectrum intensity.
    external_flux: `~numpy.ndarray`
        External spectrum intensity.
    noise: `~numpy.ndarray`
        Combined noise of target_flux and external_flux.

    Return
    ------
    loglike: scalar
        Logarithmic likelihood for parameter theta.
    """
    T, alpha, log_beta, log_gamma = theta
    beta = 10 ** log_beta
    gamma = 10 ** log_gamma

    prediction = target_model(
        spectral_axis,
        external_flux,
        T,
        alpha,
        beta,
        gamma,
    )

    diff = target_flux - prediction

    loglike = np.sum(
        -0.5 * np.log(2.0 * np.pi)
        - np.log(noise)
        - diff ** 2 / (2.0 * noise ** 2)
    )
    return loglike


def negative_gaussian_log_likelihood(
    theta, spectral_axis, target_flux, external_flux, noise
):
    """Negative Gaussian logarithmic likelihood.

    Compute the negative likelihood of the model represented by the parameter
    theta given the data. The negative sign is added for minimization
    purposes, i.e. finding the maximum likelihood parameters is the same
    as minimizing the negative likelihood.

    Parameters
    ----------
    theta: `~numpy.ndarray`
        Parameter vector: (temperature, alpha, beta, gamma).
    spectral_axis: `~astropy.units.Quantity`
        Wavelength axis. Should be the same for target_flux and external_flux.
    target_flux: `~numpy.ndarray`
        Total spectrum intensity.
    external_flux: `~numpy.ndarray`
        External spectrum intensity.
    noise: `~numpy.ndarray`
        Combined noise of target_flux and external_flux.

    Return
    ------
    neg_loglike: scalar
        Negative logarithmic likelihood for parameter theta.
    """
    loglike = gaussian_log_likelihood(
        theta, spectral_axis, target_flux, external_flux, noise
    )
    return -loglike


# ==============================================================================
# PHYSICAL CONSTRAINTS FUNCTIONS
# ==============================================================================


def alpha_vs_beta(theta, spectral_axis, target_flux, external_flux, noise):
    # we assume that alpha*ExternalSpectrum > beta*BlackBody, in mean values
    T, alpha, log_beta, log_gamma = theta
    beta = 10 ** log_beta
    gamma = 10 ** log_gamma

    prediction = target_model(
        spectral_axis,
        external_flux,
        T,
        alpha,
        beta,
        gamma,
    )

    alpha_term = np.mean(alpha * external_flux)
    beta_term = np.mean(prediction - alpha_term - gamma)

    alpha_positivity = alpha_term - beta_term

    # Positive output is True
    return alpha_positivity

def make_gamma_vs_target_flux(gamma_fraction):
    def gamma_vs_target_flux(
        theta, spectral_axis, target_flux, external_flux, noise
    ):
        # we assume that gamma can account for 5 percent or less of target flux
        T, alpha, log_beta, log_gamma = theta
        gamma = 10 ** log_gamma

        gamma_positivity = gamma_fraction * target_flux.min() - gamma

        # Positive output is True
        return gamma_positivity
    return gamma_vs_target_flux

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

    name = attr.ib(validator=validators.instance_of(str))
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
    temperature: NirdustParameter
        Parameter object with the expected blackbody temperature and
        its uncertainty.

    alpha: NirdustParameter
        Parameter object with the expected alpha value and
        its uncertainty. Note: No unit is provided as the intensity is in
        arbitrary units.

    beta: NirdustParameter
        Parameter object with the expected beta value and
        its uncertainty. Note: No unit is provided as the intensity is in
        arbitrary units.

    gamma: NirdustParameter
        Parameter object with the expected gamma value and
        its uncertainty. Note: No unit is provided as the intensity is in
        arbitrary units.

    fitted_blackbody: `~astropy.modeling.models.BlackBody`
        BlackBody instance with the best fit value of temperature.

    target_spectrum: NirdustSpectrum object
        Instance of NirdustSpectrum containing the central spectrum.

    external_spectrum: NirdustSpectrum object
        Instance of NirdustSpectrum containing the external spectrum.
    """

    temperature = attr.ib(validator=validators.instance_of(NirdustParameter))
    alpha = attr.ib(validator=validators.instance_of(NirdustParameter))
    beta = attr.ib(validator=validators.instance_of(NirdustParameter))
    gamma = attr.ib(validator=validators.instance_of(NirdustParameter))

    fitted_blackbody = attr.ib(validator=validators.instance_of(BlackBody))
    target_spectrum = attr.ib(
        validator=validators.instance_of(NirdustSpectrum)
    )
    external_spectrum = attr.ib(
        validator=validators.instance_of(NirdustSpectrum)
    )

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
        prediction = target_model(
            self.target_spectrum.spectral_axis,
            self.external_spectrum.flux.value,
            self.temperature.value,
            self.alpha.value,
            self.beta.value,
            self.gamma.value,
        )

        if ax is None:
            ax = plt.gca()

        # Target
        data_kws = {} if data_kws is None else data_kws
        data_kws.setdefault("color", "firebrick")
        ax.plot(
            self.target_spectrum.spectral_axis,
            self.target_spectrum.flux,
            label="target",
            **data_kws,
        )

        # Prediction
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
class BasinhoppingFitter:
    """Scipy Basinhopping fitter class.

    Fit a BlackBody model to the data using scipy modeling methods.

    Attributes
    ----------
    target_spectrum: NirdustSpectrum object
        Instance of NirdustSpectrum containing the central spectrum.

    external_spectrum: NirdustSpectrum object
        Instance of NirdustSpectrum containing the external spectrum.


    """
    target_spectrum = attr.ib(
        validator=validators.instance_of(NirdustSpectrum)
    )
    external_spectrum = attr.ib(
        validator=validators.instance_of(NirdustSpectrum)
    )
    basinhopping_kwargs = attr.ib()

    total_noise_ = attr.ib(init=False)

    _fitted_model = attr.ib(default=None, init=False)

    # DEFINED
    @total_noise_.default
    def _total_noise__default(self):
        """Propagated noise."""
        return np.sqrt(
            self.target_spectrum.noise ** 2 + self.external_spectrum.noise ** 2
        )

    @property
    def ndim_(self):
        """Return number of fittable parameters."""
        return 4

    @property
    def fitted_model(self):
        """Target model with best values."""
        return self._fitted_model

    @property
    def isfitted_(self):
        """Indicate if fit() was excecuted."""
        return self.fitted_model is not None

    def best_parameters(self):
        """Return best fitting values for the model.

        Return
        ------
        temperature: NirdustParameter
            Parameter object with the expected blackbody temperature and
            its uncertainty.

        alpha: NirdustParameter
            Parameter object with the expected alpha value and
            its uncertainty. Note: No unit is provided as the intensity is in
            arbitrary units.

        beta: NirdustParameter
            Parameter object with the expected beta value and
            its uncertainty. Note: No unit is provided as the intensity is in
            arbitrary units.

        gamma: NirdustParameter
            Parameter object with the expected gamma value and
            its uncertainty. Note: No unit is provided as the intensity is in
            arbitrary units.
        """
        t = self.fitted_model.T
        a = self.fitted_model.alpha
        b = self.fitted_model.beta
        g = self.fitted_model.gamma

        temp = NirdustParameter("Temperature", t.value * u.K, t.std)
        alpha = NirdustParameter("Alpha", a.value, a.std)
        beta = NirdustParameter("Beta", b.value, b.std)
        gamma = NirdustParameter("Gamma", g.value, g.std)
        return temp, alpha, beta, gamma

    def fit(self, x0, minimizer_kwargs):
        """Start fitting computation.

        Parameters
        ----------
        x0: tuple
            Vector indicating the initial guess values in order, i.e:
            (T, alpha, beta, gamma). Default: (1000.0, 1.0, 1.0, 1.0)
        """
        if self.isfitted_:
            raise RuntimeError("Model already fitted.")

        if x0 is None:
            x0 = (1000.0, 8.0, 9.0, -5.0)
        elif len(x0) != self.ndim_:
            raise ValueError("Invalid initial guess.")

        self.run_model(x0, minimizer_kwargs)
        self._fitted = True
        return self

    def run_model(self, x0, minimizer_kwargs):
        """Run fitter given an initial guess.

        Parameters
        ----------
        x0: tuple
            Vector indicating the initial guess values in order, i.e:
            (T, alpha, beta, gamma). Default: (1000.0, 1.0, 1.0, 1.0)
        """
        # Compute model
        frequency_axis = self.target_spectrum.frequency_axis.value
        target_flux = self.target_spectrum.flux.value

        res = basinhopping(
            negative_gaussian_log_likelihood,
            x0,
            minimizer_kwargs=minimizer_kwargs,
            **self.basinhopping_kwargs,
        )

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


# ==============================================================================
# FITTER FUNCTION WRAPPER
# ==============================================================================



def make_constraints(args, gamma_fraction):
    gamma_vs_target_flux = make_gamma_vs_target_flux(gamma_fraction)
    constraints = (
        {"type": "ineq", "fun": alpha_vs_beta, "args": args},
        {"type": "ineq", "fun": gamma_vs_target_flux, "args": args},
    )
    return constraints


def make_minimizer_kwargs(args, bounds, constraints):
    minimizer_kwargs = {
        "method": "SLSQP",
        "args": args,
        "bounds": bounds,
        "constraints": constraints,
        "options": {"maxiter": 1000},
    }
    return minimizer_kwargs


def fit_blackbody(
    target_spectrum,
    external_spectrum,
    x0=None,
    bounds=None,
    gamma_target_fraction=0.05,
    seed=None,
    niter=100,
    # basinhopping_kwargs=None,
    # minimizer_kwargs=None,
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

    basinhopping_kwargs = {
        "niter":niter,
        "T":100,
        "stepsize":1,
        "seed":seed,
    }
    fitter = BasinhoppingFitter(
        target_spectrum=target_spectrum,
        external_spectrum=external_spectrum,
        basinhopping_kwargs=basinhopping_kwargs,
    )
    args = (
        target_spectrum.spectral_axis,
        target_spectrum.flux.value,
        external_spectrum.flux.value,
        target_spectrum.noise,
    )
    constraints = make_constraints(args, gamma_target_fraction)
    minimizer_kwargs = make_minimizer_kwargs(args, bounds, constraints)

    fitter.fit(x0, minimizer_kwargs=minimizer_kwargs)
    return fitter
