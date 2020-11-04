# ==============================================================================
# DOCS
# ==============================================================================

"""Nirdust test suite."""


# ==============================================================================
# IMPORTS
# ==============================================================================
from astropy import units as u
from astropy.io import fits

import attr

import numpy as np

import specutils as su

# ==============================================================================
# CONSTANTS
# ==============================================================================


# ==============================================================================
# CLASSES
# ==============================================================================


@attr.s(frozen=True)
class NirdustSpectrum:

    header = attr.ib(repr=False)
    z = attr.ib()
    spectrum_length = attr.ib()
    dispersion_key = attr.ib()
    first_wavelength = attr.ib()
    dispersion_type = attr.ib()
    spec1d = attr.ib(repr=False)
    axis_to_freq = attr.ib(repr=False)

    def __getattr__(self, a):
        return getattr(self.spec1d, a)

    def __getitem__(self, slice):
        spec1d = self.spec1d.__getitem__(slice)
        axis_to_freq = spec1d.spectral_axis.to(u.Hz)
        kwargs = attr.asdict(self)
        kwargs.update(
            spec1d=spec1d,
            axis_to_freq=axis_to_freq,
        )
        return NirdustSpectrum(**kwargs)


# ==============================================================================
# LOAD SPECTRA
# ==============================================================================


def make_spectrum(
    flux,
    header,
    z=0,
    dispersion_key="CD1_1",
    first_wavelength="CRVAL1",
    dispersion_type="CTYPE1",
    **kwargs,
):

    if header[dispersion_type] != "LINEAR":
        raise ValueError("dispersion must be LINEAR")

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
    axis_to_freq = spec1d.spectral_axis.to(u.Hz)

    return NirdustSpectrum(
        header=header,
        z=z,
        spectrum_length=spectrum_length,
        dispersion_key=dispersion_key,
        first_wavelength=first_wavelength,
        dispersion_type=dispersion_type,
        spec1d=spec1d,
        axis_to_freq=axis_to_freq,
    )


def read_spectrum(file_name, extension, z, **kwargs):

    with fits.open(file_name) as spectrum:

        fluxx = spectrum[extension].data
        header = fits.getheader(file_name)

    single_spectrum = make_spectrum(flux=fluxx, header=header, z=z, **kwargs)

    return single_spectrum


# ==============================================================================
# PREPARE SPECTRA FOR FITTING
# ==============================================================================


def nirdust_prepare(nuclear_spectrum):

    ...
