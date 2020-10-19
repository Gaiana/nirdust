# ==============================================================================
# DOCS
# ==============================================================================

"""Nirdust test suite."""


# ==============================================================================
# IMPORTS
# ==============================================================================

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import specutils as su
import os
import pathlib
from pathlib import Path
from astropy import units as u


# ==============================================================================
# CLASSES
# ==============================================================================


class NuclearSpectrum(su.Spectrum1D):
    def __init__(
        self,
        spectrum_file,
        extension,
        radius=None,
        flux=None,
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
    ):
        self.spectrum = fits.open(spectrum_file)
        self.fluxx = self.spectrum[extension].data
        self.header = fits.getheader(spectrum_file)
        self.radius = radius

        super().__init__(
            flux=self.spectrum[extension].data * u.adu,
            spectral_axis=(
                self.header["CRVAL1"]
                + self.header["CD1_1"]
                * np.arange(0, len(self.spectrum[extension].data))
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

        self.plot = plt.plot(self.spectral_axis, self.flux)
        self.close = self.spectrum.close(spectrum_file)


class Off_NuclearSpectrum(NuclearSpectrum):
    ...


# ==============================================================================
# LOAD SPECTRA
# ==============================================================================

PATH = os.path.abspath(os.path.dirname(__file__))


def from_folder(nuclear_spectra_folder):

    files_nuclei = pathlib.Path(PATH) / "nuclear_spectra"

    nuclear_list = []

    for name in os.listdir(files_nuclei):
        single_spectrum = NuclearSpectrum(files_nuclei / name, 0)
        nuclear_list.append(single_spectrum)
    return nuclear_list


def from_list():

    files_nuclei = pathlib.Path(PATH) / "nuclear_spectra"

    c_list = Path(files_nuclei / "customized_list.txt")

    with c_list.open() as f:
        lines = f.readlines()

    nuclear_list = []
    for line in lines:
        fila = line.rstrip()
        single_spectrum = NuclearSpectrum(files_nuclei / fila, 0)
        nuclear_list.append(single_spectrum)
    return nuclear_list


def from_filename(file_name):

    files_nuclei = pathlib.Path(PATH) / "nuclear_spectra"

    single_spectrum = NuclearSpectrum(files_nuclei / file_name, 0)
    return single_spectrum


def load_off_nuclear_spectrum(file_name):

    off_nuclear_spectrum = pathlib.Path(PATH) / "off_nuclear_spectrum"

    single_spectrum = Off_NuclearSpectrum(off_nuclear_spectrum / file_name, 0)
    return single_spectrum

# ==============================================================================
#
# ==============================================================================
