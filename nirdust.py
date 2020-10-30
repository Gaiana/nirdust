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
        dispersion_key="CD1_1",
        first_wavelength="CRVAL1",
        dispersion_type="CTYPE1",
        **kwargs,
    ):

        self.header = header
        self.spectrum_length = len(flux)
        spectral_axis = (
            self.header[first_wavelength]
            + self.header[dispersion_key] * np.arange(0, self.spectrum_length)
        ) * u.AA

        if self.header[dispersion_type] != "LINEAR":
            raise ValueError("dispersion must be LINEAR")

        super().__init__(
            flux=flux * u.adu, spectral_axis=spectral_axis, **kwargs
        )


# ==============================================================================
# LOAD SPECTRA
# ==============================================================================


def read_spectrum(file_name, extension, **kwargs):

    with fits.open(file_name) as spectrum:

        fluxx = spectrum[extension].data
        header = fits.getheader(file_name)

    single_spectrum = NirdustSpectrum(flux=fluxx, header=header, **kwargs)

    return single_spectrum
