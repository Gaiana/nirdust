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

"""Nirdust specter temperature based on K-band (2.2 micrometers).

Nirdust is a python package that uses K-band (2.2 micrometers) spectra to
measure the temperature of the dust heated by an Active Galactic Nuclei (AGN)
accretion disk.

"""


# ==============================================================================
# IMPORTS
# ==============================================================================
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling import fitting

import attr

import matplotlib.pyplot as plt

import numpy as np

import specutils as su
import specutils.manipulation as sm


# ==============================================================================
# BLACK BODY METHODS
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
        print(temperature)
        if not isinstance(temperature, u.Quantity):
            in_temp = u.Quantity(temperature, u.K)
        else:
            in_temp = temperature

        if not isinstance(nu, u.Quantity):
            in_freq = u.Quantity(nu, u.Hz)
        else:
            in_freq = nu

        # Convert to units for calculations, also force double precision
        # This is just in case the input units differ from K or Hz
        with u.add_enabled_equivalencies(u.spectral() + u.temperature()):
            freq = u.Quantity(in_freq, u.Hz, dtype=np.float64)
            temp = u.Quantity(in_temp, u.K)

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


def blackbody_fitter(nirspec, T0):
    """Fits Blackbody model to spectrum.

    The fitting is performed by using the LevMarLSQFitter class from Astropy.

    Parameters
    ----------
    nirspec: NirdustSpectrum object
        A NirdustSpectrum instance containing a prepared spectrum for blackbody
        fitting. Note: the spectrum must be cuted, normalized and corrected for
        stellar population contribution.

    T0: float
        Initial temperature for the fitting procedure.

    Return
    ------
    out: tuple
        Best fitted model and a dictionary containing the parameters of the
        best fitted model.

    """
    # LevMarLSQFitter does not support inputs with units
    if isinstance(T0, u.Quantity):
        temp = T0.value
    else:
        temp = T0

    bb_model = NormalizedBlackBody(temp)
    fitter = fitting.LevMarLSQFitter()
    fitted_model = fitter(
        bb_model, nirspec.frequency_axis.value, nirspec.flux.value
    )

    return fitted_model, fitter.fit_info


# ==============================================================================
# CLASSES
# ==============================================================================


@attr.s(frozen=True)
class NirdustSpectrum:
    """
    Class containing a spectrum to operate with nirdust.

    Stores the spectrum in a Spectrum1D object and provides various methods
    for obtaining the dust component and perform blackbody fitting.

    Parameters
    ----------
    header: FITS header
        The header of the spectrum obtained from the fits file.

    z: float
        Redshift of the galaxy. Default is 0.

    spectrum_length: int
        The number of items in the spectrum axis as in len() method.

    dispersion_key: float
        Header keyword containing the dispersion in Å/pix.

    first_wavelength: float
        Header keyword containing the wavelength of first pixel.

    dispersion_type: str
        Header keyword containing the type of the dispersion function.

    spec1d: specutils.Spectrum1D object
        Containis the wavelength axis and the flux axis of the spectrum in
        unities of Å and ADU respectively.

    frequency_axis: SpectralAxis object
        Spectral axis in units of Hz


    """

    header = attr.ib(repr=False)
    z = attr.ib()
    spectrum_length = attr.ib()
    dispersion_key = attr.ib()
    first_wavelength = attr.ib()
    dispersion_type = attr.ib()
    spec1d = attr.ib(repr=False)
    frequency_axis = attr.ib(repr=False)

    def __getattr__(self, a):
        """Return an attribute from specutils.Spectrum1D class.

        Parameters
        ----------
        a: attribute from spectrum1D class.

        Returns
        -------
        out: a

        """
        return getattr(self.spec1d, a)

    def __getitem__(self, slice):
        """Define the method for getting a slice of a NirdustSpectrum object.

        Parameters
        ----------
        slice: pair of indexes given with the method [].

        Return
        ------
        out: NirsdustSpectrum object
            Return a new instance of the class NirdustSpectrum sliced by the
            given indexes.
        """
        spec1d = self.spec1d.__getitem__(slice)
        frequency_axis = spec1d.spectral_axis.to(u.Hz)
        kwargs = attr.asdict(self)
        kwargs.update(
            spec1d=spec1d,
            frequency_axis=frequency_axis,
        )
        return NirdustSpectrum(**kwargs)

    def cut_edges(self, mini, maxi):
        """Cut the spectrum in wavelength range.

        Parameters
        ----------
        mini: float
            Lower limit to cut the spectrum.

        maxi: float
            Upper limit to cut the spectrum.

        Returns
        -------
        out: NirsdustSpectrum object
            Return a new instance of class NirdustSpectrum cut in wavelength.
        """
        region = su.SpectralRegion(mini * u.AA, maxi * u.AA)
        cutted_spec1d = sm.extract_region(self.spec1d, region)
        cutted_freq_axis = cutted_spec1d.spectral_axis.to(u.Hz)
        new_len = len(cutted_spec1d.flux)
        kwargs = attr.asdict(self)
        kwargs.update(
            spec1d=cutted_spec1d,
            spectrum_length=new_len,
            frequency_axis=cutted_freq_axis,
        )

        return NirdustSpectrum(**kwargs)

    def convert_to_frequency(self):
        """Convert the spectral axis to frequency in units of Hz.

        Returns
        -------
        out: object NirsdustSpectrum
            New instance of the NirdustSpectrun class containing the spectrum
            with a frquency axis in units of Hz.
        """
        new_axis = self.spec1d.spectral_axis.to(u.Hz)
        kwargs = attr.asdict(self)
        kwargs.update(
            frequency_axis=new_axis,
        )
        return NirdustSpectrum(**kwargs)

    def normalize(self):
        """Normalize the spectrum to the unity using the mean value.

        Returns
        -------
        out: NirsdustSpectrum object
            New instance of the NirdustSpectrun class with the flux normalized
            to unity.
        """
        normalized_flux = self.spec1d.flux / np.mean(self.spec1d.flux)
        new_spec1d = su.Spectrum1D(normalized_flux, self.spec1d.spectral_axis)
        kwargs = attr.asdict(self)
        kwargs.update(spec1d=new_spec1d)
        return NirdustSpectrum(**kwargs)

    def fit_blackbody(self, T0):
        """Call blackbody_fitter and store results in a NirdustResults object.

        Parameters
        ----------
        T0: float
            Initial temperature for the fit.

        Returns
        -------
        out: NirdustResults object
            An instance of the NirdustResults class that holds the resuslts of
            the blackbody fitting.
        """
        inst = blackbody_fitter(self, T0)

        storage = NirdustResults(
            temperature=inst[0].T,
            info=inst[1],
            covariance=inst[1]["param_cov"],
            fitted_blackbody=inst[0],
            freq_axis=self.frequency_axis,
            flux_axis=self.flux,
        )
        return storage


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

    covariance:  scalar
        The covariance of the fit as calculed by LevMarLSQFitter.

    fitted_blackbody: model
        The normalized_blackbody model for the best fit.

    freq_axis: SpectralAxis object
        The axis containing the spectral information of the spectrum in
        units of Hz.

    flux_axis: Quantity
        The flux of the spectrum in arbitrary units.



    """

    def __init__(
        self,
        temperature,
        info,
        covariance,
        fitted_blackbody,
        freq_axis,
        flux_axis,
    ):

        self.temperature = temperature.value * u.K
        self.info = info
        self.covariance = covariance
        self.fitted_blackbody = fitted_blackbody
        self.freq_axis = freq_axis
        self.flux_axis = flux_axis

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
# LOAD SPECTRA
# ==============================================================================


def spectrum(
    flux,
    header,
    z=0,
    dispersion_key="CD1_1",
    first_wavelength="CRVAL1",
    dispersion_type="CTYPE1",
    **kwargs,
):
    """Instantiate a NirdustSpectrum object from FITS parameters.

    Parameters
    ----------
    flux: Quantity
        Intensity for each pixel in arbitrary units.

    header: FITS header
        Header of the spectrum.

    z: float
    Redshif of the galaxy.

    dispersion_key: str
        Header keyword that gives dispersion in Å/pix. Default is 'CD1_1'

    first_wavelength: str
        Header keyword that contains the wavelength of the first pixel. Default
        is ``CRVAL1``.

    dispersion_type: str
        Header keyword that contains the dispersion function type. Default is
        ``CTYPE1``.

    Return
    ------
    spectrum: ``NirsdustSpectrum``
        Return a instance of the class NirdustSpectrum with the entered
        parameters.
    """
    if header[dispersion_key] <= 0:
        raise ValueError("dispersion must be positive")

    spectrum_length = len(flux)
    spectral_axis = (
        (
            header[first_wavelength]
            + header[dispersion_key] * np.arange(0, spectrum_length)
        )
        / (1 + z)
        * u.AA
    )
    spec1d = su.Spectrum1D(
        flux=flux * u.adu, spectral_axis=spectral_axis, **kwargs
    )
    frequency_axis = spec1d.spectral_axis.to(u.Hz)

    return NirdustSpectrum(
        header=header,
        z=z,
        spectrum_length=spectrum_length,
        dispersion_key=dispersion_key,
        first_wavelength=first_wavelength,
        dispersion_type=dispersion_type,
        spec1d=spec1d,
        frequency_axis=frequency_axis,
    )


def read_spectrum(file_name, extension, z, **kwargs):
    """Read a spectrum in FITS format and store it in a NirdustSpectrum object.

    Parameters
    ----------
    file_name: str
        Path to where the fits file is stored.

    extension: int
        Extension of the FITS file where the spectrum is stored.

    z: float
        Redshift of the galaxy.

    Returns
    -------
    out: NirsdustSpectrum object
        Returns an instance of the class NirdustSpectrum.
    """
    with fits.open(file_name) as fits_spectrum:

        fluxx = fits_spectrum[extension].data
        header = fits.getheader(file_name)

    single_spectrum = spectrum(flux=fluxx, header=header, z=z, **kwargs)

    return single_spectrum


# ==============================================================================
# PREPARE SPECTRA FOR FITTING
# ==============================================================================


def sp_correction(nuclear_spectrum, external_spectrum):
    """Stellar Population substraction.

    The spectral continuum of Type 2 Seyfert galaxies in the K band
    (19.-2.5 $mu$m) is composed by the stellar population component and the
    hot dust component. The first one is the sum of the Planck functions of
    all the stars in the host galaxy and can be represented by a spectrum
    extracted at a prudential distance from the nucleus, where the emission
    is expected to be dominated by the stellar population. In sp_correction
    this is introduced in the parameter "external spectrum". The stellar
    population dominated spectrum must be substracted from the nuclear
    spectrum in order to obtain the hot dust component in the nuclear
    spectrum. The excess obtained from the substraction is expected to have
    blackbody-shape.

    The operations applied to prepare the nuclear spectrum for fitting are:

    1) normalization using the mean value of the flux for both spectra
    2) substraction of the external spectrum flux from the nuclear spectrum
       flux.

    Parameters
    ----------
    nuclear_spectrum: NirdustSpectrum object
        Instance of NirdusSpectrum containing the nuclear spectrum.

    external_spectrum: NirdustSpectrum object
        Instance of NirdusSpectrum containing the external spectrum.

    Returns
    -------
    out: NirsdustSpectrum object
        Returns a new instance of the class NirdustSpectrum containing the
        nuclear spectrum ready for blackbody fitting.
    """
    normalized_nuc = nuclear_spectrum.normalize()
    normalized_ext = external_spectrum.normalize()

    dif = len(normalized_nuc.spec1d.spectral_axis) - len(
        normalized_ext.spec1d.spectral_axis
    )

    if dif == 0:

        flux_resta = (
            normalized_nuc.spec1d.flux - normalized_ext.spec1d.flux
        ) + 1

        new_spectral_axis = nuclear_spectrum.spectral_axis

    elif dif < 0:

        new_ext = normalized_ext[-dif:]
        flux_resta = (normalized_nuc.spec1d.flux - new_ext.spec1d.flux) + 1

        new_spectral_axis = external_spectrum.spectral_axis[-dif:]

    elif dif > 0:

        new_nuc = normalized_nuc[dif:]
        flux_resta = (new_nuc.spec1d.flux - normalized_ext.spec1d.flux) + 1
        new_spectral_axis = nuclear_spectrum.spectral_axis[dif:]

    substracted_1d_spectrum = su.Spectrum1D(flux_resta, new_spectral_axis)

    new_freq_axis = substracted_1d_spectrum.spectral_axis.to(u.Hz)

    kwargs = attr.asdict(normalized_nuc)
    kwargs.update(spec1d=substracted_1d_spectrum, frequency_axis=new_freq_axis)

    return NirdustSpectrum(**kwargs)
