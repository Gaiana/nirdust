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
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling import fitting

import attr

import matplotlib.pyplot as plt

import numpy as np


# ==============================================================================
# CLASSES
# ==============================================================================


class NormalizedBlackBody(Fittable1DModel):
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

    # We parametrize this model with a temperature.
    temperature = Parameter(default=None, min=0)

    @property
    def T(self):
        """Proxy for temperature."""
        return self.temperature

    def evaluate(self, nu, temperature):
        """Evaluate the model.

        Parameters
        ----------
        nu : float, `~numpy.ndarray`, or `~astropy.units.Quantity`
            Frequency at which to compute the blackbody. If no units are given,
            this defaults to Hz.

        temperature : float, `~numpy.ndarray`, or `~astropy.units.Quantity`
            Temperature of the blackbody. If no units are given, this defaults
            to Kelvin.

        Returns
        -------
        intensity : number or ndarray
            Blackbody spectrum.

        """
        # Convert to units for calculations, also force double precision
        # This is just in case the input units differ from K or Hz
        with u.add_enabled_equivalencies(u.spectral() + u.temperature()):
            freq = u.Quantity(nu, u.Hz, dtype=np.float64)
            temp = u.Quantity(temperature, u.K)

        log_boltz = const.h * freq / (const.k_B * temp)
        boltzm1 = np.expm1(log_boltz)

        # Calculate blackbody flux and normalize with the mean
        bb = 2.0 * const.h * freq ** 3 / (const.c ** 2 * boltzm1) / u.sr
        intensity = bb / np.mean(bb)

        # If the temperature parameter has no unit, we should return a unitless
        # value. This occurs for instance during fitting, since we drop the
        # units temporarily.
        if hasattr(temperature, "unit"):
            return intensity
        return intensity.value


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


# =============================================================================
# FUNCTIONS
# =============================================================================


def normalized_blackbody_fitter(frequency, flux, T0):
    """Fits a Normalized Blackbody model to spectrum.

    The fitting is performed by using the LevMarLSQFitter class from Astropy.

    Parameters
    ----------
    frequency: `~numpy.ndarray`, or `~astropy.units.Quantity`
        Spectral axis in units of Hz.

    flux: `~numpy.ndarray`, or `~astropy.units.Quantity`
        Normalized intensity.

    T0: float
        Initial temperature for the fitting procedure.

    Return
    ------
    out: tuple
        Best fitted model and a dictionary containing the parameters of the
        best fitted model.

    """
    # LevMarLSQFitter does not support inputs with units
    freq_value = u.Quantity(frequency, u.Hz).value
    flux_value = u.Quantity(flux).value
    T0_value = u.Quantity(T0, u.K).value

    bb_model = NormalizedBlackBody(temperature=T0_value)
    fitter = fitting.LevMarLSQFitter(calc_uncertainties=True)
    fitted_model = fitter(bb_model, freq_value, flux_value)

    return fitted_model, fitter.fit_info


def fit_blackbody(spectra, T0):
    """Call blackbody_fitter and store results in a NirdustResults object.

    Parameters
    ----------
    spectra:

    T0: float or `~astropy.units.Quantity`
        Initial temperature for the fit in Kelvin.

    Returns
    -------
    out: NirdustResults object
        An instance of the NirdustResults class that holds the resuslts of
        the blackbody fitting.
    """
    model, fit_info = normalized_blackbody_fitter(
        spectra.spectral_axis, spectra.flux, T0
    )

    storage = NirdustResults(
        temperature=model.temperature,
        info=fit_info,
        uncertainty=model.temperature.std,
        fitted_blackbody=model,
        freq_axis=spectra.spectral_axis,
        flux_axis=spectra.flux,
    )
    return storage
