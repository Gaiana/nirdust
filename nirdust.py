# ==============================================================================
# DOCS
# ==============================================================================

"""Nirdust test suite."""


# ==============================================================================
# IMPORTS
# ==============================================================================
from astropy import units as u
from astropy.io import fits
from astropy.modeling.models import custom_model

import attr

import numpy as np

import specutils as su
import specutils.manipulation as sm

# ==============================================================================
# CONSTANTS
# ==============================================================================

# initial guessing for the dust temperature and black-body scale
initial_temp = 1200

# ==============================================================================
# FUNCTIONS
# ==============================================================================


@custom_model
def normalized_blackbody(nu, T):
    """Normalize black-body function."""
    from astropy.constants import h, k_B, c

    cv = c.value
    kv = k_B.value
    hv = h.value

    bb = 2 * hv * nu ** 3 / (cv ** 2 * (np.exp(hv * nu / (kv * T)) - 1))
    mean = np.mean(bb)

    return bb / mean


# ==============================================================================
# CLASSES
# ==============================================================================


@attr.s(frozen=True)
class NirdustSpectrum:
    """
    Creat the class type NirdustSectrum.

    Prepares the spectrum with specific methods, stores it in each instance of
    execution and then adjusts it.

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
        """Return objets after apply the "a" funcion."""
        return getattr(self.spec1d, a)

    def __getitem__(self, slice):
        """Define the method that cuts the spectrum.

        Parameters
        ----------
        slice: object
            The object contain a NirdustSpectrum

        Return
        ------
        out: objets NirsdustSpectrum
            Return a new instance of the class nirdustspectrum.
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
            mini: minimum wavelength to be cut
            maxi: maximum wavelength to be cut

        Return
        ------
        out: objets NirsdustSpectrum
            Return a new instance of class nirdustspectrum cut in wavelength.
        """
        region = su.SpectralRegion(mini * u.AA, maxi * u.AA)
        cutted_spec1d = sm.extract_region(self.spec1d, region)
        kwargs = attr.asdict(self)
        kwargs.update(
            spec1d=cutted_spec1d,
        )
        return NirdustSpectrum(**kwargs)

    def convert_to_frequency(self):
        """Convert the spectral axis to frequency in units of GHz.

        Return
        ------
        out: objets NirsdustSpectrum
            New instance of the nirdustSpectrun classe containing spectrum in
            frequencies.
        """
        new_axis = self.spec1d.spectral_axis.to(u.Hz)
        kwargs = attr.asdict(self)
        kwargs.update(
            frequency_axis=new_axis,
        )
        return NirdustSpectrum(**kwargs)

    def _normalize(self):
        """Normalize the spectrum to the mean value.

        Return
        ------
        out: objets NirsdustSpectrum
            New instance of the nirdustSpectrun classe whose flow is normalized
            to the average value.
        """
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
    """Instantiate NirdustSpectrum from fits parameters.

    Parameters
    ----------
    flux: Intensity for each pixel in arbitrary units
    header: Header of spectrum
    z: Redshift
    dispersion_key: keywords that gives dispersion in Å/pix
    first_wavelength: keywords that gives wavelength of first pixel
    dispersion_type: keywords that gives order the dispersion function

    Return
    ------
    out: objets NirsdustSpectrum
        Return a class of nerdustspectrum with the entered parameters.
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
    """Read a spectrum in fits format.

    Parameters
    ----------
    file_name: Path to where the fits file is stored
    extension: File extension fits
    z: Redshift

    Return
    ------
    out: objets NirsdustSpectrum
        Return a new instance of the class nirdustspectrum with the stored
        spectrum.
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
    """Prepare the nuclear spectrum for black-body fitting.

    The operations applied to prepare the nuclear spectrum are:

    1) normalization to the mean value of the flux axis for both spectra
    3) substraction of the external spectrum flux from the nuclear spectrum
    flux.

    Parameters
    ----------
    nuclear_spectrum: instance of NirdusSpectrum containing the nuclear
    spectrum.

    external_spectrum: instance of NirdusSpectrum containing the external
    spectrum.

    mini: lower limit of wavelenght to cut spectra

    maxi: upper limit of wavelenght to cut spectra


    Return
    ------
    out: objet NirsdustSpectrum
        Return a new instance of the class NirdustSpectrum containing the
        nuclear spectrum ready for black-body fitting.
    """
    normalized_nuc = nuclear_spectrum._normalize()
    normalized_ext = external_spectrum._normalize()

    dif = len(normalized_nuc.spec1d.spectral_axis) - len(
        normalized_ext.spec1d.spectral_axis
    )

    if dif == 0:

        flux_resta = (
            normalized_nuc.spec1d.flux - normalized_ext.spec1d.flux
        ) + 1

    elif dif < 0:

        new_ext = normalized_ext[-dif:]
        flux_resta = (normalized_nuc.spec1d.flux - new_ext.spec1d.flux) + 1

    elif dif > 0:

        new_nuc = normalized_nuc[dif:]
        flux_resta = (new_nuc.spec1d.flux - normalized_ext.spec1d.flux) + 1

    substracted_1d_spectrum = su.Spectrum1D(
        flux_resta, normalized_nuc.spectral_axis
    )

    kwargs = attr.asdict(normalized_nuc)
    kwargs.update(spec1d=substracted_1d_spectrum)

    return NirdustSpectrum(**kwargs)
