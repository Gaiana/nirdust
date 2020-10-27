# ==============================================================================
# DOCS
# ==============================================================================

"""Nirdust test suite."""


# ==============================================================================
# IMPORTS
# ==============================================================================
import glob
import os
import pathlib

from astropy import units as u
from astropy.io import fits

import numpy as np

import specutils as su

# ==============================================================================
# CONSTANTS
# ==============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))


# ==============================================================================
# CLASSES
# ==============================================================================


class NuclearSpectrum(su.Spectrum1D):
    def __init__(
        self,
        flux,
        header,
        radius=None,
        spectral_axis=None,
        data=None,
        flux_unit=None,
        wav_unit=None,
        wcs=None,
        velocity_convention=None,
        rest_value=None,
        redshift=None,
        radial_velocity=None,
        bin_specification=None,
        uncertainty=None,
        mask=None,
        meta=None,
        dispersion_key="CD1_1",
        first_wavelength="CRVAL1",
        dispersion_type="CTYPE1",
    ):
        self.header = header
        self.spectrum_length = len(flux)
        self.radius = radius

        if self.header[dispersion_type] != "LINEAR":
            raise ValueError("dispersion must be LINEAR")

        super().__init__(
            flux=flux * u.adu,
            spectral_axis=(
                self.header[first_wavelength]
                + self.header[dispersion_key]
                * np.arange(0, self.spectrum_length)
            )
            * u.AA,
            wcs=wcs,
            velocity_convention=velocity_convention,
            rest_value=rest_value,
            redshift=redshift,
            radial_velocity=radial_velocity,
            bin_specification=bin_specification,
            uncertainty=uncertainty,
            mask=mask,
            meta=meta,
        )


# class Off_NuclearSpectrum(NuclearSpectrum):
#    ...


# ==============================================================================
# LOAD SPECTRA
# ==============================================================================


def read_single_spectrum(file_name, extension, **kwargs):  # IO

    with fits.open(file_name) as spectrum:

        fluxx = spectrum[extension].data
        header = fits.getheader(file_name)

    single_spectrum = NuclearSpectrum(flux=fluxx, header=header, **kwargs)

    return single_spectrum


def read_sample(names, extension, **kwargs):
    spectra = []
    for path in glob.glob(names):
        spectrum = read_single_spectrum(path, extension)
        spectra.append(spectrum)

    return spectra
