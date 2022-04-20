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

import matplotlib.pyplot as plt

import numpy as np

from scipy.optimize import OptimizeResult, basinhopping

from .core import NirdustSpectrum

# ==============================================================================
# EXCEPTIONS
# ==============================================================================


# ==============================================================================
# TARGET SPECTRUM MODEL
# ==============================================================================


def target_model(external_spectrum, T, alpha, beta, gamma):
    """Compute the expected spectrum given a blackbody prediction.

    Parameters
    ----------
    external_spectrum: NirdustSpectrum object
        Instance of NirdustSpectrum containing the external spectrum.
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
    spectral_axis = external_spectrum.spectral_axis
    external_flux = external_spectrum.flux.value

    # calculate the model
    blackbody = BlackBody(u.Quantity(T, u.K))
    bb_flux = blackbody(spectral_axis).value

    prediction = alpha * external_flux + 10 ** beta * bb_flux + 10 ** gamma
    return prediction


# ==============================================================================
# LIKELIHOOD FUNCTIONS
# ==============================================================================


def negative_gaussian_log_likelihood(
    theta, target_spectrum, external_spectrum
):
    """Negative Gaussian logarithmic likelihood.

    Compute the negative likelihood of the model represented by the parameter
    theta given the data. The negative sign is added for minimization
    purposes, i.e. finding the maximum likelihood parameters is the same
    as minimizing the negative likelihood.

    Parameters
    ----------
    theta: array-like
        Parameter vector: (temperature, alpha, beta, gamma).
    target_spectrum: NirdustSpectrum object
        Instance of NirdustSpectrum containing the total central spectrum.
    external_spectrum: NirdustSpectrum object
        Instance of NirdustSpectrum containing the external spectrum.

    Return
    ------
    loglike: scalar
        Negative logarithmic likelihood for parameter theta.
    """
    prediction = target_model(external_spectrum, *theta)

    noise = target_spectrum.noise
    diff = target_spectrum.flux.value - prediction

    loglike = np.sum(
        -0.5 * np.log(2.0 * np.pi)
        - np.log(noise)
        - diff ** 2 / (2.0 * noise ** 2)
    )
    return -loglike


# ==============================================================================
# PHYSICAL CONSTRAINTS FUNCTIONS
# ==============================================================================


def alpha_vs_beta(theta, target_spectrum, external_spectrum):
    """Alpha term positivity relative to beta term.

    Here we assume that: alpha * ExternalSpectrum > 10**beta * BlackBody
    in mean values.

    Parameters
    ----------
    theta: array-like
        Parameter vector: (temperature, alpha, beta, gamma).
    target_spectrum: NirdustSpectrum object
        Instance of NirdustSpectrum containing the total central spectrum.
    external_spectrum: NirdustSpectrum object
        Instance of NirdustSpectrum containing the external spectrum.

    Return
    ------
    alpha_positivity: scalar
        The difference between alpha term and beta term mean values, given the
        data; i.e.: mean(alpha * ExternalSpectrum) - (10**beta * BlackBody)
    """
    # we assume that alpha*ExternalSpectrum > beta*BlackBody, in mean values
    T, alpha, beta, gamma = theta

    prediction = target_model(external_spectrum, T, alpha, beta, gamma)

    alpha_term = np.mean(alpha * external_spectrum.flux.value)
    beta_term = np.mean(prediction - alpha_term - 10 ** gamma)

    alpha_positivity = alpha_term - beta_term

    # Positive output is True
    return alpha_positivity


def make_gamma_vs_target_flux(gamma_fraction):
    """Encapsulate gamma_fraction for gamma constraint function.

    Parameters
    ----------
    gamma_fraction: scalar
        Value between [0, 1] representing the maximum fraction of Target
        flux allowed for the gamma term.

    Return
    ------
    gamma_vs_target_flux: function
        Function that computes the gamma term positivity relative to the
        total target flux. Call signature:
        gamma_vs_target_flux(theta, target_spectrum, external_spectrum)
    """
    if not (0 <= gamma_fraction <= 1):
        raise ValueError("Gamma fraction must be in the (0, 1) range.")

    def gamma_vs_target_flux(theta, target_spectrum, external_spectrum):
        # we assume that gamma can account for 5 percent or less of target flux
        T, alpha, beta, gamma = theta

        min_flux = target_spectrum.flux.value.min()
        gamma_positivity = gamma_fraction * min_flux - 10 ** gamma

        # Positive output is True
        return gamma_positivity

    return gamma_vs_target_flux


# ==============================================================================
# RESULT CLASSES
# ==============================================================================


@attr.s(frozen=True)
class NirdustParameter:
    """Parameter representation.

    Attributes
    ----------
    name: str
        Parameter name.
    value: scalar, `~astropy.units.Quantity`
        Expected value for parameter after fitting procedure.
    uncertainty: scalar, `~astropy.units.Quantity`
        Uncertainties associated to the fitted value.
    """

    name = attr.ib(validator=validators.instance_of(str))
    value = attr.ib()
    uncertainty = attr.ib(default=None)


@attr.s(frozen=True)
class NirdustResults:
    """Create the class NirdustResults.

    Storages the results obtained with BasinhoppingFitter plus the dust
    spectrum. The method plot() can be called to plot the spectrum and
    the blackbody model obtained in the fitting.

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

    minimizer_results: OptimizeResult object
        Instance of OptimizeResult that generates in the fitting procedure.
    """

    temperature = attr.ib(validator=validators.instance_of(NirdustParameter))
    alpha = attr.ib(validator=validators.instance_of(NirdustParameter))
    beta = attr.ib(validator=validators.instance_of(NirdustParameter))
    gamma = attr.ib(validator=validators.instance_of(NirdustParameter))

    fitted_blackbody = attr.ib(validator=validators.instance_of(BlackBody))
    target_spectrum = attr.ib(
        validator=validators.instance_of(NirdustSpectrum),
    )
    external_spectrum = attr.ib(
        validator=validators.instance_of(NirdustSpectrum),
    )
    minimizer_results = attr.ib(
        validator=validators.instance_of(OptimizeResult)
    )

    def plot(
        self, axes=None, data_kws=None, model_kws=None, show_components=False
    ):
        """Build a plot of the fitted spectrum and the fitted model.

        Parameters
        ----------
        axes: tuple of ``matplotlib.pyplot.Axis`` objects
            Tuple with two objects of type Axes containing complete
            information of the properties to generate the image, by default
            it is None.

        data_kws: ``dict``
            Dictionaries of keyword arguments. Passed to the data plotting
            function.

        model_kws: ``dict``
            Dictionaries of keyword arguments. Passed to the model plotting
            function.

        show_components: ``bool``
            Flag to indicate if the three components of the model should be
            plotted.

        Return
        ------
        out: ``matplotlib.pyplot.Axis`` :
            The axis where the method draws.
        """
        prediction = target_model(
            self.external_spectrum,
            self.temperature.value,
            self.alpha.value,
            self.beta.value,
            self.gamma.value,
        )
        wave_axis = self.target_spectrum.spectral_axis.value

        if axes is None:
            gkw = {"height_ratios": [4, 1], "hspace": 0}
            fig = plt.figure(figsize=(8, 6), tight_layout=True)
            axes = fig.subplots(2, 1, sharex=True, gridspec_kw=gkw)

        ax, axr = axes

        # Target
        data_kws = {} if data_kws is None else data_kws
        data_kws.setdefault("color", "firebrick")
        ax.plot(
            wave_axis,
            self.target_spectrum.flux.value,
            label="target",
            **data_kws,
        )

        # Prediction
        model_kws = {} if model_kws is None else model_kws
        model_kws.setdefault("color", "Navy")
        ax.plot(
            wave_axis,
            prediction,
            label="prediction",
            **model_kws,
        )

        if show_components:
            alpha_term = self.alpha.value * self.external_spectrum.flux.value
            beta_term = (10 ** self.beta.value) * self.fitted_blackbody(
                self.target_spectrum.spectral_axis
            ).value
            gamma_term = (10 ** self.gamma.value) * np.ones_like(wave_axis)

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
        residuals = (
            self.target_spectrum.flux.value - prediction
        ) / self.target_spectrum.noise
        axr.plot(
            wave_axis,
            residuals,
            linestyle="solid",
            color="gray",
        )

        axr.set_xlabel("Angstroms [A]")
        ax.set_ylabel("Intensity [arbitrary units]")
        ax.legend()

        return ax, axr


# ==============================================================================
# FITTER CLASSES
# ==============================================================================


@attr.s
class BasinhoppingFitter:
    """Basinhopping fitter class.

    Fit a BlackBody model to the data using scipy modeling methods.

    Attributes
    ----------
    target_spectrum: NirdustSpectrum object
        Instance of NirdustSpectrum containing the central spectrum.

    external_spectrum: NirdustSpectrum object
        Instance of NirdustSpectrum containing the external spectrum.

    basinhopping_kwargs: dict
        Dictionary of keyword arguments to be passed to the scipy basinhopping
        routine. Read the documentation for a detailed description:
        docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html
    """

    target_spectrum = attr.ib(
        validator=validators.instance_of(NirdustSpectrum)
    )
    external_spectrum = attr.ib(
        validator=validators.instance_of(NirdustSpectrum)
    )
    basinhopping_kwargs = attr.ib(validator=validators.instance_of(dict))

    total_noise_ = attr.ib(init=False)

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

    def fit(self, x0, minimizer_kwargs):
        """Start fitting computation.

        Parameters
        ----------
        x0: tuple
            Vector indicating the initial guess values in order, i.e:
            (T, alpha, beta, gamma). Default: (1000.0, 8.0, 9.0, -5.0)
        minimizer_kwargs: dict
            Extra keyword arguments to be passed to the local minimizer.

        Return
        ------
        results: NirdustResult object
            Results of the fitting procedure.
        """
        if x0 is None:
            x0 = (1000.0, 8.0, 9.0, -5.0)
            # x0 = make_initial_guess()
        elif len(x0) != self.ndim_:
            raise ValueError("Invalid initial parameters.")

        res = self.run_model(x0, minimizer_kwargs)

        # Now estimate parameters uncertainties
        # res_with_errors = self.estimate_uncertainties(res)

        # return result
        temp, alpha, beta, gamma = res.x
        temp = NirdustParameter("Temperature", temp * u.K, 0)
        alpha = NirdustParameter("Alpha", alpha, 0)
        beta = NirdustParameter("Beta", beta, 0)
        gamma = NirdustParameter("Gamma", gamma, 0)
        return NirdustResults(
            temperature=temp,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            fitted_blackbody=BlackBody(temp.value),
            target_spectrum=self.target_spectrum,
            external_spectrum=self.external_spectrum,
            minimizer_results=res,
        )

    def run_model(self, x0, minimizer_kwargs):
        """Run fitter given an initial guess.

        Parameters
        ----------
        x0: tuple
            Vector indicating the initial guess values in order, i.e:
            (T, alpha, beta, gamma). Default: (1000.0, 1.0, 1.0, 1.0)
        minimizer_kwargs: dict
            Extra keyword arguments to be passed to the local minimizer.

        Return
        ------
        results: OptimizeResult object
            Results of the local minimizer.
        """
        res = basinhopping(
            negative_gaussian_log_likelihood,
            x0,
            minimizer_kwargs=minimizer_kwargs,
            **self.basinhopping_kwargs,
        )
        # should rise warning if the fit failed ?
        return res


# ==============================================================================
# FITTER FUNCTION WRAPPER
# ==============================================================================


def make_constraints(args, gamma_fraction):
    """Make scipy minimizer constraints.

    Parameters
    ----------
    args: tuple
        Extra arguments to be passed to likelihood and model functions.
        args = (target_spectrum, external_spectrum)
    gamma_fraction: float
        Maximum fraction allowed to constraint the gamma value in the fitting
        procedure.

    Return
    ------
    contraints: tuple
        Constraints as required by the scipy SLSQP minimizer.

    """
    gamma_vs_target_flux = make_gamma_vs_target_flux(gamma_fraction)
    constraints = (
        {"type": "ineq", "fun": alpha_vs_beta, "args": args},
        {"type": "ineq", "fun": gamma_vs_target_flux, "args": args},
    )
    return constraints


def make_minimizer_kwargs(args, bounds, constraints, options=None):
    """Make scipy minimizer keyword arguments.

    Parameters
    ----------
    args: tuple
        Extra arguments to be passed to likelihood and model functions.
        args = (target_spectrum, external_spectrum)
    bounds: tuple
        Tuple of 4 pairs of values indicating the minimum and maximum allowed
        values of the fitted parameters. The order is: T, alpha, beta, gamma.
        Example: bounds = ((0, 2000), (0, 20), (6, 10), (-10, 0))
    contraints: tuple
        Constraints as required by the scipy SLSQP minimizer.
    options: dict
        Extra options to be passed to the local minimizer through the
        `options` keyword. Default: {"maxiter": 1000}
    """
    if options is None:
        options = {"maxiter": 1000}

    minimizer_kwargs = {
        "method": "SLSQP",
        "args": args,
        "bounds": bounds,
        "constraints": constraints,
        "options": options,
        "jac": "3-point",
    }
    return minimizer_kwargs


# bounds
BOUNDS = ((0.0, 2000.0), (0, 20), (6, 10), (-10, 0))


def fit_blackbody(
    target_spectrum,
    external_spectrum,
    x0=None,
    bounds=None,
    gamma_target_fraction=0.05,
    seed=None,
    niter=200,
    stepsize=1,
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

    x0: tuple, optional
        Vector indicating the initial guess values of temperature, alpha, beta
        and gamma.

    bounds: tuple
        Tuple of 4 pairs of values indicating the minimum and maximum allowed
        values of the fitted parameters. The order is: T, alpha, beta, gamma.
        Example: bounds = ((0, 2000), (0, 20), (6, 10), (-10, 0))

    gamma_target_fraction: float
        Maximum fraction of gamma vs target flux allowed to constraint the
        gamma value in the fitting procedure. Default: 0.05.

    seed: int
        Random number generation seed for the basinhopping algorithm.

    niter: int
        Number of basinhopping iterations. This numbers represents how many
        times the local minimizer will be excecuted. Default: 200.

    stepsize: float
        Maximum step size for use in the random displacement of x0 for each
        basinhopping iteration. Default: 1.0.

    Return
    ------
    result: NirdustResults object
        Instance of NirdustResults after the fitting procedure.
    """
    # Check defaults
    if bounds is None:
        bounds = BOUNDS

    basinhopping_kwargs = {
        "niter": niter,
        "T": 100,
        "stepsize": stepsize,
        "seed": seed,
        "niter_success": None,
        "interval": 50,
    }
    fitter = BasinhoppingFitter(
        target_spectrum=target_spectrum,
        external_spectrum=external_spectrum,
        basinhopping_kwargs=basinhopping_kwargs,
    )

    args = (target_spectrum, external_spectrum)
    constraints = make_constraints(args, gamma_target_fraction)
    minimizer_kwargs = make_minimizer_kwargs(args, bounds, constraints)

    result = fitter.fit(x0, minimizer_kwargs=minimizer_kwargs)
    return result
