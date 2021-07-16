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

"""Core functionalities for nirdust."""


# ==============================================================================
# IMPORTS
# ==============================================================================

from collections.abc import Mapping

from astropy import constants as const
from astropy import units as u
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling import fitting, models

import attr

import matplotlib.pyplot as plt

import numpy as np

import specutils as su
import specutils.manipulation as sm
from specutils.fitting import find_lines_threshold
from specutils.fitting import fit_generic_continuum
from specutils.fitting import fit_lines
from specutils.manipulation import FluxConservingResampler
from specutils.manipulation import noise_region_uncertainty
from specutils.spectra import SpectralRegion, Spectrum1D

import uttr


# ==============================================================================
# PRIVATE FUNCTIONS
# ==============================================================================


def _filter_internals(attribute, value):
    """Filter internal attributes of a class."""
    return not (attribute.name.startswith("_") or attribute.name.endswith("_"))


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


# ==============================================================================
# NIRDUST_SPECTRUM CLASS
# ==============================================================================
@attr.s(frozen=True, slots=True, repr=False)
class _NDSpectrumMetadata(Mapping):
    """Convenience Wrapper around a mapping type."""

    _md = attr.ib(validator=attr.validators.instance_of(Mapping))

    def __getitem__(self, k):
        """x.__getitem__(y) <==> x[y]."""
        return self._md[k]

    def __getattr__(self, a):
        """x.__getattr__(y) <==> x.y."""
        try:
            return self[a]
        except KeyError:
            raise AttributeError(a)

    def __iter__(self):
        """x.__iter__() <==> iter(x)."""
        return iter(self._md)

    def __len__(self):
        """x.__len__() <==> len(x)."""
        return len(self._md)

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
        return f"metadata({repr(set(self._md))})"

    def __dir__(self):
        """x.__dir__() <==> dir(x)."""
        return super().__dir__() + list(self._md)


@attr.s(frozen=True, repr=False)
class NirdustSpectrum:
    """Class containing a spectrum to operate with nirdust.

    Stores the spectrum in a Spectrum1D object and provides various methods
    for obtaining the dust component and perform blackbody fitting.

    Parameters
    ----------
    flux: `~numpy.ndarray`, or `~astropy.units.Quantity`
        Spectral intensity.

    frequency_axis: `~numpy.ndarray`, or `~astropy.units.Quantity`
        Spectral axis in units of Hz.

    z: float, optional
        Redshift of the galaxy. Default is 0.

    metadata: mapping, optional
        Any dict like object. This is a good place to store the header
        of the fist file or any arbitrary mapping. Internally NirdustSpectrum
        wraps the object inside a convenient metadata object usefull to
        access the keys as attributes.

    Attributes
    ----------
    spec1d_: specutils.Spectrum1D object
        Contains the wavelength axis and the flux axis of the spectrum in
        unities of Å and ADU respectively.
    """

    spectral_axis = attr.ib(converter=u.Quantity)
    flux = attr.ib(converter=u.Quantity)

    z = attr.ib(default=0)
    metadata = attr.ib(factory=dict, converter=_NDSpectrumMetadata)

    spec1d_ = attr.ib(init=False)

    arr_ = uttr.array_accessor()

    @spec1d_.default
    def _spec1d_default(self):
        return su.Spectrum1D(
            flux=self.flux,
            spectral_axis=self.spectral_axis,  # redshift=self.z,
        )

    def __dir__(self):
        """List all the content of the NirdustSpectum and the internal \
        spec1d.

        dir(x) <==> x.__dir__()
        """
        return super().__dir__() + dir(self.spec1d_)

    def __repr__(self):
        """Representation of the NirdustSpectrum.

        repr(x) <==> x.__repr__()

        """
        sprange = self.spectral_range[0].value, self.spectral_range[1].value
        spunit = self.spectral_axis.unit

        return (
            f"NirdustSpectrum(z={self.z}, "
            f"spectral_length={len(self.flux)}, "
            f"spectral_range=[{sprange[0]:.2f}-{sprange[1]:.2f}] {spunit})"
        )

    def __getattr__(self, a):
        """Return an attribute from specutils.Spectrum1D class.

        Parameters
        ----------
        a: attribute from spectrum1D class.

        Returns
        -------
        out: a

        """
        return getattr(self.spec1d_, a)

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
        spec1d = self.spec1d_.__getitem__(slice)
        flux = spec1d.flux
        spectral_axis = spec1d.spectral_axis

        kwargs = attr.asdict(self, filter=_filter_internals)
        kwargs.update(
            flux=flux,
            spectral_axis=spectral_axis,
        )
        return NirdustSpectrum(**kwargs)

    def __len__(self):
        """x.__len__() <==> len(x)."""
        return len(self.flux)

    @property
    def frequency_axis(self):
        """Frequency axis access."""
        return self.spectral_axis.to(u.Hz, equivalencies=u.spectral())

    @property
    def spectral_range(self):
        """First and last values of spectral_axis."""
        return [
            np.min(self.spectral_axis),
            np.max(self.spectral_axis),
        ]

    # quitar esto ahora que esta el len()???
    @property
    def spectral_length(self):
        """Total number of spectral data points."""
        return len(self.flux)

    @property
    def spectral_dispersion(self):
        """Assume linearity to compute the dispersion."""
        a, b = self.spectral_range
        return (b - a) / (self.spectral_length - 1)

    def mask_spectrum(self, line_intervals=None, mask=None):
        """Mask spectrum to remove spectral lines.

        Recives either a boolean mask containing 'False' values in the line
        positions or a list with the line positions as given by the
        'line_spectrum' method of the NirdustSpectrum class. This method uses
        one of those imputs to remove points from the spectrum.

        Parameters
        ----------
        line_intervals: python iterable
            Any iterable object with pairs containing the beginning and end of
            the region were the spectral lines are. The second return of
            'line_spectrum()' is valid.

        mask: boolean array
            array with same length as the spectrum containing boolean values
            with False values in the indexes that should be masked.

        Return
        ------
        NirdustSpectrum object
            A new instance of NirdustSpectrum class with the especified
            intervals removed.
        """
        if all(v is None for v in (line_intervals, mask)):
            raise ValueError("Expected one parameter, recived none.")

        elif all(v is not None for v in (line_intervals, mask)):
            raise ValueError("Two mask parameters were given. Expected one.")

        elif line_intervals is not None:

            line_indexes = np.searchsorted(self.spectral_axis, line_intervals)
            auto_mask = np.ones(self.spectral_length, dtype=bool)

            for i, j in line_indexes:
                auto_mask[i : j + 1] = False  # noqa

            masked_spectrum = Spectrum1D(
                self.flux[auto_mask], self.spectral_axis[auto_mask]
            )

        elif mask is not None:

            if len(mask) != self.spectral_length:
                raise ValueError(
                    "Mask length must be equal to 'spectral_length'"
                )

            masked_spectrum = Spectrum1D(
                self.flux[mask], self.spectral_axis[mask]
            )

        kwargs = attr.asdict(self, filter=_filter_internals)
        kwargs.update(
            flux=masked_spectrum.flux,
            spectral_axis=masked_spectrum.spectral_axis,
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
        cutted_spec1d = sm.extract_region(self.spec1d_, region)

        kwargs = attr.asdict(self, filter=_filter_internals)
        kwargs.update(
            flux=cutted_spec1d.flux,
            spectral_axis=cutted_spec1d.spectral_axis,
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
        new_axis = self.spectral_axis.to(u.Hz, equivalencies=u.spectral())

        kwargs = attr.asdict(self, filter=_filter_internals)
        kwargs.update(spectral_axis=new_axis)
        return NirdustSpectrum(**kwargs)

    def normalize(self):
        """Normalize the spectrum to the unity using the mean value.

        Returns
        -------
        out: NirsdustSpectrum object
            New instance of the NirdustSpectrun class with the flux normalized
            to unity.
        """
        normalized_flux = self.flux / np.mean(self.flux)

        kwargs = attr.asdict(self, filter=_filter_internals)
        kwargs.update(flux=normalized_flux)
        return NirdustSpectrum(**kwargs)

    def fit_blackbody(self, T0):
        """Call blackbody_fitter and store results in a NirdustResults object.

        Parameters
        ----------
        T0: float or `~astropy.units.Quantity`
            Initial temperature for the fit in Kelvin.

        Returns
        -------
        out: NirdustResults object
            An instance of the NirdustResults class that holds the resuslts of
            the blackbody fitting.
        """
        model, fit_info = normalized_blackbody_fitter(
            self.spectral_axis, self.flux, T0
        )

        storage = NirdustResults(
            temperature=model.temperature,
            info=fit_info,
            uncertainty=model.temperature.std,
            fitted_blackbody=model,
            freq_axis=self.spectral_axis,
            flux_axis=self.flux,
        )
        return storage


# =============================================================================
# NIRDUST RESULT
# =============================================================================


def _temperature_to_kelvin(temperature):
    return u.Quantity(temperature, u.K)


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

    temperature = attr.ib(converter=_temperature_to_kelvin)
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
# RESAMPLE SPECTRA TO MATCH SPECTRAL RESOLUTIONS
# ==============================================================================


def _rescale(sp, reference_sp):
    """Resample a given spectrum to a reference spectrum.

    The first spectrum will be resampled to have the same spectral_axis as
    the reference spectrum. The resampling algorithm is the specutils method
    FluxConservingResampler.

    Notes
    -----
    nan values may occur at the edges where the resampler is forced
    to extrapolate.
    """
    input_sp1d = sp.spec1d_
    resample_axis = reference_sp.spectral_axis

    resampler = FluxConservingResampler(extrapolation_treatment="nan_fill")
    output_sp1d = resampler(input_sp1d, resample_axis)

    kwargs = attr.asdict(sp, filter=_filter_internals)
    kwargs.update(
        flux=output_sp1d.flux,
        spectral_axis=output_sp1d.spectral_axis,
    )
    return NirdustSpectrum(**kwargs)


def _clean_and_match(sp1, sp2):
    """Clean nan values and apply the same mask to both spectrums."""
    # nan values occur in the flux variable
    # check for invalid values in both spectrums
    mask = np.isfinite(sp1.flux) & np.isfinite(sp2.flux)

    sp_list = []
    for sp in [sp1, sp2]:
        kw = attr.asdict(sp, filter=_filter_internals)
        kw.update(flux=sp.flux[mask], spectral_axis=sp.spectral_axis[mask])
        sp_list.append(NirdustSpectrum(**kw))

    return sp_list


def match_spectral_axes(
    first_sp,
    second_sp,
    scaling="downscale",
    clean=True,
):
    """Resample the higher resolution spectrum.

    Spectrum_resampling uses the spectral_axis of the lower resolution
    spectrum to resample the higher resolution one. To do so this function
    uses the FluxConservingResampler() class of 'Specutils'. The order of the
    input spectra is arbitrary and the order in the output is the same as in
    the input. Only the higher resolution spectrum will be modified, the lower
    resolution spectrum will be unaltered. It is recommended to run
    spectrum_resampling after 'cut_edges'.

    Parameters
    ----------
    first_sp: NirdustSpectrum object

    second_sp: NirdustSpectrum object

    scaling: string
        If 'downscale' the higher resolution spectrum will be resampled to
        match the lower resolution spectrum. If 'upscale' the lower resolution
        spectrum will be resampled to match the higher resolution spectrum.

    clean: bool
        Flag to indicate if the spectrums have to be cleaned by nan values
        after the rescaling procedure. nan values occur at the edges of the
        resampled spectrum when it is forced to extrapolate beyond the
        spectral range of the reference spectrum.

    Return
    ------
    out: NirdustSpectrum, NirdustSpectrum

    """
    scaling = scaling.lower()
    if scaling not in ["downscale", "upscale"]:
        raise ValueError(
            "Unknown scaling mode. Must be 'downscale' or 'upscale'."
        )

    first_disp = first_sp.spectral_dispersion
    second_disp = second_sp.spectral_dispersion

    dispersion_difference = (first_disp - second_disp).value

    # Larger numerical dispersion means lower resolution!
    if dispersion_difference > 0:
        # Check type of rescaling
        if scaling == "downscale":
            second_sp = _rescale(second_sp, reference_sp=first_sp)
        else:
            first_sp = _rescale(first_sp, reference_sp=second_sp)

    elif dispersion_difference < 0:
        if scaling == "downscale":
            first_sp = _rescale(first_sp, reference_sp=second_sp)
        else:
            second_sp = _rescale(second_sp, reference_sp=first_sp)

    else:
        # they have the same dispersion, is that equivalent
        # to equal spectral_axis?
        pass

    if clean:
        first_sp, second_sp = _clean_and_match(first_sp, second_sp)

    return first_sp, second_sp


# ==============================================================================
# FIND LINE INTERVALS FROM AUTHOMATIC LINE FITTING
# ==============================================================================


def _make_window(center, delta):
    """Create window array."""
    return np.array([center - delta, center + delta])


def line_spectrum(
    spectrum,
    low_lim_ns=20650,
    upper_lim_ns=21000,
    noise_factor=3,
    window=50,
):
    """Construct the line spectrum.

    Uses various Specutils features to fit the continuum of the spectrum,
    subtract it and find the emission and absorption lines in the spectrum.
    Then fits all the lines with gaussian models to construct the line
    spectrum.

    Parameters
    ----------
    spectrum: NirdustSpectrum object
        A spectrum stored in a NirdustSpectrum class object.

    low_lim_ns: float
        Lower limit of the spectral region defined to measure the
        noise level. Default is 20650 (wavelenght in Angstroms).

    upper_lim_ns: float
        Lower limit of the spectral region defined to measure the
        noise level. Default is 21000 (wavelenght in Angstroms).

    noise_factor: float
        Same parameter as in find_lines_threshold from Specutils.
        Factor multiplied by the spectrum’s``uncertainty``, used for
        thresholding. Default is 3.

    window: float
        Same parameter as in fit_lines from specutils.fitting. Regions of
        the spectrum to use in the fitting. If None, then the whole
        spectrum will be used in the fitting. Window is used in the
        Gaussian fitting of the spectral lines. Default is 50 (Angstroms).

    Return
    ------
    out: flux axis, list, list
        Returns in the first position a flux axis of the same lenght as the
        original spectrum containing the fitted lines. In the second position,
        returns the intervals where those lines were finded determined by
        3-sigma values around the center of the line. In the third position
        returns an array with the quality of the fitting for each line.

    """
    # values in correct units
    low_lim_ns = u.Quantity(low_lim_ns, u.AA)
    upper_lim_ns = u.Quantity(upper_lim_ns, u.AA)
    window = u.Quantity(window, u.AA)

    # By defaults this fits a Chebyshev of order 3 to the flux
    model = fit_generic_continuum(
        spectrum.spec1d_, fitter=fitting.LinearLSQFitter()
    )
    continuum = model(spectrum.spectral_axis)
    new_flux = spectrum.spec1d_ - continuum

    noise_region_def = SpectralRegion(low_lim_ns, upper_lim_ns)
    noise_reg_spectrum = noise_region_uncertainty(new_flux, noise_region_def)
    lines = find_lines_threshold(noise_reg_spectrum, noise_factor=noise_factor)

    line_sign = {"emission": 1.0, "absorption": -1.0}
    line_spectrum = np.zeros(len(new_flux.spectral_axis))
    line_intervals = []

    for line in lines:
        amp = line_sign[line["line_type"]]
        center = line["line_center"].value

        gauss_model = models.Gaussian1D(amplitude=amp, mean=center)
        gauss_fit = fit_lines(new_flux, gauss_model, window=window)
        intensity = gauss_fit(new_flux.spectral_axis)
        interval = _make_window(center, 3 * gauss_fit.stddev)

        line_spectrum += intensity.value
        line_intervals.append(interval)

    line_spectrum = u.Quantity(line_spectrum)
    line_intervals = u.Quantity(line_intervals, u.AA)

    line_fitting_quality = 0.0
    return line_spectrum, line_intervals, line_fitting_quality


# ==============================================================================
# ISOLATE THE DUST COMPONENT
# ==============================================================================


def dust_component(nuclear_spectrum, external_spectrum):
    """Isolate the dust component via stellar population substraction.

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
        Instance of NirdustSpectrum containing the nuclear spectrum.

    external_spectrum: NirdustSpectrum object
        Instance of NirdustSpectrum containing the external spectrum.

    Return
    ------
    out: NirsdustSpectrum object
        Returns a new instance of the class NirdustSpectrum containing the
        nuclear spectrum ready for blackbody fitting.
    """
    normalized_nuc = nuclear_spectrum.normalize()
    normalized_ext = external_spectrum.normalize()

    flux_resta = (
        normalized_nuc.spec1d_.flux - normalized_ext.spec1d_.flux
    ) + 1

    new_spectral_axis = nuclear_spectrum.spec1d_.spectral_axis

    kwargs = attr.asdict(normalized_nuc, filter=_filter_internals)
    kwargs.update(
        flux=flux_resta,
        spectral_axis=new_spectral_axis,
    )

    return NirdustSpectrum(**kwargs)
