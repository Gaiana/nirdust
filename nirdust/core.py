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

"""Core functionalities for Nirdust."""


# ==============================================================================
# IMPORTS
# ==============================================================================

from collections.abc import Mapping

from astropy import units as u
from astropy.modeling import fitting

import attr

import numpy as np

import specutils as su
import specutils.manipulation as sm
from specutils.fitting import fit_generic_continuum
from specutils.manipulation import noise_region_uncertainty
from specutils.spectra import SpectralRegion
from specutils.spectra import Spectrum1D

# ==============================================================================
# UTILITIES
# ==============================================================================


def _filter_internals(attribute, value):
    """Filter internal attributes of a class."""
    return not (attribute.name.startswith("_") or attribute.name.endswith("_"))


def public_members_asdict(object):
    """Thin wrapper around attr.asdict, that ignore all private members."""
    kwargs = attr.asdict(object, filter=_filter_internals, recurse=False)
    kwargs["metadata"] = dict(kwargs["metadata"])
    return kwargs


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
        content = repr(set(self._md)) if self._md else "{}"
        return f"metadata({content})"

    def __dir__(self):
        """x.__dir__() <==> dir(x)."""
        return super().__dir__() + list(self._md)


@attr.s(frozen=True, repr=False)
class NirdustSpectrum:
    """Class containing a spectrum to operate with Nirdust.

    Stores the flux and wavelength axis of the spectrum in a Spectrum1D object
    and provides various methods for obtaining the dust component and perform
    blackbody fitting.
    The `flux` parameter recieves either flux-calibrated intensity or non
    flux-calibrated intensity, in both cases Nirdust will assign 'ADU' units to
    it.

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
        of the fits file or any arbitrary mapping. Internally NirdustSpectrum
        wraps the object inside a convenient metadata object usefull to
        access the keys as attributes.

    Attributes
    ----------
    spec1d_: `specutils.Spectrum1D` object
        Contains the wavelength axis and the flux axis of the spectrum in
        unities of Å and ADU respectively.

    noise: float.
        A value of the uncertainty representative of all the spectral range.
        If the value of noise is not provided, Nirdust will compute it by
        default using `noise_region_uncertainty` from `Astropy` inside the
        region 20650 - 21000 Å. The user can re-compute noise using the class
        method `compute_noise`.
    """

    spectral_axis = attr.ib(converter=u.Quantity)
    flux = attr.ib(converter=u.Quantity)

    z = attr.ib(default=0)
    metadata = attr.ib(factory=dict, converter=_NDSpectrumMetadata)

    spec1d_ = attr.ib(init=False)
    noise = attr.ib()

    @spec1d_.default
    def _spec1d_default(self):
        return su.Spectrum1D(
            flux=self.flux,
            spectral_axis=self.spectral_axis,  # redshift=self.z,
        )

    @noise.default
    def _noise_default(self):

        low_lim_default = u.Quantity(20650, u.AA)
        upper_lim_default = u.Quantity(21000, u.AA)

        # By defaults this fits a Chebyshev of order 3 to the fluxMETA
        model = fit_generic_continuum(
            self.spec1d_, fitter=fitting.LinearLSQFitter()
        )

        continuum = model(self.spectral_axis)
        new_flux = self.spec1d_ - continuum

        noise_region_def = SpectralRegion(low_lim_default, upper_lim_default)
        noise_value = noise_region_uncertainty(
            new_flux, noise_region_def
        ).uncertainty.array[1]

        return noise_value

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
        """Return an attribute from `specutils.Spectrum1D` class.

        Parameters
        ----------
        a: attribute from `spectrum1D` class.

        Returns
        -------
        out: a

        """
        return getattr(self.spec1d_, a)

    def __getitem__(self, slice):
        """Define the method for getting a slice of a `NirdustSpectrum` object.

        Parameters
        ----------
        slice: pair of indexes given with the method [].

        Return
        ------
        out: `NirsdustSpectrum` object
            Return a new instance of the class `NirdustSpectrum` sliced by the
            given indexes.
        """
        spec1d = self.spec1d_.__getitem__(slice)
        flux = spec1d.flux
        spectral_axis = spec1d.spectral_axis

        kwargs = public_members_asdict(self)
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
        """Assume linearity to compute the spectral dispersion."""
        a, b = self.spectral_range
        return (b - a) / (self.spectral_length - 1)

    def compute_noise(self, low_lim=20650, upper_lim=21000):
        """Compute noise for the spectrum.

        Uses `noise_region_uncertainty` from Astropy to compute the noise
        of the spectrum inside a `Spectral Region` given by the `low_lim`
        and `upper_lim` parameters.

        Parameters
        ----------
        low_lim: float
            A float containing the lower limit for the region where the noise
            will be computed. Must be in Å. Default is 20650.

        upper_lim: float
            A float containing the lower limit for the region where the noise
            will be computed. Must be in Å. Default is 20650.

        Return
        ------
        `NirdustSpectrum` object
            A new instance of `NirdustSpectrum` class with the new noise
            parameter.

        """
        
        if (low_lim < self.spectral_range[0].value) or (low_lim > self.spectral_range[1].value):
            raise ValueError("Region parameter out of spectrum bounds")
        
        if low_lim >= upper_lim:
            raise ValueError("low_lim parameter must be lower than upper_lim")
        
        low_lim_q = u.Quantity(low_lim, u.AA)
        upper_lim_q = u.Quantity(upper_lim, u.AA)

        # By defaults this fits a Chebyshev of order 3 to the flux
        model = fit_generic_continuum(
            self.spec1d_, fitter=fitting.LinearLSQFitter()
        )

        continuum = model(self.spectral_axis)
        new_flux = self.spec1d_ - continuum

        noise_region_def = SpectralRegion(low_lim_q, upper_lim_q)
        noise_value = noise_region_uncertainty(
            new_flux, noise_region_def
        ).uncertainty.array[1]

        region = {'nr_low_lim':low_lim_q.value, 'nr_upper_lim': upper_lim_q.value}
                
        kwargs = public_members_asdict(self)
        meta = kwargs['metadata']
        meta.update(region)
        kwargs.update(noise=noise_value, metadata=meta)
       

        return NirdustSpectrum(**kwargs)


    def mask_spectrum(self, line_intervals=None, mask=None):
        """Mask spectrum to remove spectral lines.

        Recives either a boolean mask containing `False` values in the line
        positions or a list with the line positions as given by the
        `line_spectrum` method of the `NirdustSpectrum` class. This method uses
        one of those imputs to remove points from the spectrum.

        Parameters
        ----------
        line_intervals: python iterable
            Any iterable object with pairs containing the beginning and end of
            the region were the spectral lines are. The second return of
            `line_spectrum` is valid.

        mask: boolean array
            array with same length as the spectrum containing boolean values
            with `False` values in the indexes that should be masked.

        Return
        ------
        `NirdustSpectrum` object
            A new instance of `NirdustSpectrum` class with the especified
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

        kwargs = public_members_asdict(self)

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
        out: `NirsdustSpectrum` object
            Return a new instance of class `NirdustSpectrum` cut in wavelength.
        """
        region = su.SpectralRegion(mini * u.AA, maxi * u.AA)
        cutted_spec1d = sm.extract_region(self.spec1d_, region)

        kwargs = public_members_asdict(self)
        kwargs.update(
            flux=cutted_spec1d.flux,
            spectral_axis=cutted_spec1d.spectral_axis,
        )
        return NirdustSpectrum(**kwargs)

    def convert_to_frequency(self):
        """Convert the spectral axis to frequency in units of Hz.

        Returns
        -------
        out: object `NirsdustSpectrum`
            New instance of the `NirdustSpectrun` class containing the spectrum
            with a frquency axis in units of Hz.
        """
        new_axis = self.spectral_axis.to(u.Hz, equivalencies=u.spectral())

        kwargs = public_members_asdict(self)
        kwargs.update(spectral_axis=new_axis)
        return NirdustSpectrum(**kwargs)

    def normalize(self):
        """Normalize the spectrum to the unity using the mean value.

        Returns
        -------
        out: `NirsdustSpectrum` object
            New instance of the `NirdustSpectrun` class with the flux
            normalized to unity.
        """
        normalized_flux = self.flux / np.mean(self.flux)

        kwargs = public_members_asdict(self)
        kwargs.update(flux=normalized_flux)
        return NirdustSpectrum(**kwargs)
