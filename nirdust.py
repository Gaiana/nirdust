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
import specutils.manipulation as sm

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
    frequency_axis = attr.ib(repr=False)

    def __getattr__(self, a):
        return getattr(self.spec1d, a)

    def __getitem__(self, slice):
        spec1d = self.spec1d.__getitem__(slice)
        frequency_axis = spec1d.spectral_axis.to(u.Hz)
        kwargs = attr.asdict(self)
        kwargs.update(
            spec1d=spec1d,
            frequency_axis=frequency_axis,
        )
        return NirdustSpectrum(**kwargs)

    def cut_edges(self, mini, maxi):
        region = su.SpectralRegion(mini * u.AA, maxi * u.AA)
        cutted_spec1d = sm.extract_region(self.spec1d, region)
        kwargs = attr.asdict(self)
        kwargs.update(
            spec1d=cutted_spec1d,
        )
        return NirdustSpectrum(**kwargs)

    def _convert_to_frequency(self):
        new_axis = self.spec1d.spectral_axis.to(u.GHz)
        kwargs = attr.asdict(self)
        kwargs.update(
            frequency_axis=new_axis,
        )
        return NirdustSpectrum(**kwargs)

    def _normalization(self):
        normalized_flux = self.spec1d.flux / np.mean(self.spec1d.flux)
        new_spec1d = su.Spectrum1D(normalized_flux, self.spec1d.spectral_axis)
        kwargs = attr.asdict(self)
        kwargs.update(spec1d=new_spec1d)
        return NirdustSpectrum(**kwargs)


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

    with fits.open(file_name) as fits_spectrum:

        fluxx = fits_spectrum[extension].data
        header = fits.getheader(file_name)

    single_spectrum = spectrum(flux=fluxx, header=header, z=z, **kwargs)

    return single_spectrum


# ==============================================================================
# PREPARE SPECTRA FOR FITTING
# ==============================================================================


def Nirdustprepare(nuclear_spectrum, external_spectrum, mini, maxi):

    step1_nuc = nuclear_spectrum.cut_edges(mini, maxi)
    step1_ext = external_spectrum.cut_edges(mini, maxi)

    step2_nuc = step1_nuc._convert_to_frequency()
    step2_ext = step1_ext._convert_to_frequency()

    step3_nuc = step2_nuc._normalization()
    step3_ext = step2_ext._normalization()

    dif = len(step3_nuc.spec1d.spectral_axis) - len(
        step3_ext.spec1d.spectral_axis
    )

    if dif == 0:

        flux_resta = (step3_nuc.spec1d.flux - step3_ext.spec1d.flux) + 1

    elif dif < 0:

        new_step3_ext = step3_ext[-dif:]
        flux_resta = (step3_nuc.spec1d.flux - new_step3_ext.spec1d.flux) + 1

    else:

        new_step3_nuc = step3_nuc[dif:]
        flux_resta = (new_step3_nuc.spec1d.flux - step3_ext.spec1d.flux) + 1

    prepared_spectrum = su.Spectrum1D(flux_resta, step2_nuc.frequency_axis)

    kwargs = attr.asdict(step3_nuc)
    kwargs.update(spec1d=prepared_spectrum)

    return NirdustSpectrum(**kwargs)
