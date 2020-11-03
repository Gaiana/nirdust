# ==============================================================================
# DOCS
# ==============================================================================

"""Nirdust test suite."""


# ==============================================================================
# IMPORTS
# ==============================================================================
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


class NirdustSpectrum(su.Spectrum1D):
    def __init__(
        self,
        flux,
        header,
        z=0,
        dispersion_key="CD1_1",
        first_wavelength="CRVAL1",
        dispersion_type="CTYPE1",
        axis_to_frec=None,
        **kwargs,
    ):

        self.header = header
        self.z = z
        self.spectrum_length = len(flux)
        spectral_axis = (
            (
                self.header[first_wavelength]
                + self.header[dispersion_key]
                * np.arange(0, self.spectrum_length)
            )
            / (1 + self.z)
            * u.AA
        )

        if self.header[dispersion_type] != "LINEAR":
            raise ValueError("dispersion must be LINEAR")

        super().__init__(
            flux=flux * u.adu, spectral_axis=spectral_axis, **kwargs
        )

        self.axis_to_frec = self.spectral_axis.to(u.Hz)


# ==============================================================================
# LOAD SPECTRA
# ==============================================================================


def read_spectrum(file_name, extension, z, **kwargs):

    with fits.open(file_name) as spectrum:

        fluxx = spectrum[extension].data
        header = fits.getheader(file_name)

    single_spectrum = NirdustSpectrum(flux=fluxx, header=header, z=z, **kwargs)

    return single_spectrum


# ==============================================================================
# PREPARE SPECTRA FOR FITTING
# ==============================================================================


def nirdust_prepare(nuclear_spectrum):

    ...
