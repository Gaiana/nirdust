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
        """Return objets after apply the "a" funcion"""
        
        return getattr(self.spec1d, a)

    def __getitem__(self, slice):
        """Defines the method that cuts the spectrum.

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
        """This funcion cuts the spectrum in wavelength range.

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

    def _convert_to_frequency(self):
        """Converts the spectral axis to frequency in units of GHz.

        Return
        ------
        out: objets NirsdustSpectrum
            New instance of the nirdustSpectrun classe containing spectrum in
            frequencies.
        """ 
        new_axis = self.spec1d.spectral_axis.to(u.GHz)
        kwargs = attr.asdict(self)
        kwargs.update(
            frequency_axis=new_axis,
        )
        return NirdustSpectrum(**kwargs)

    def _normalization(self):
        """This function normalizes the spectrum to the mean value.

         
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
    """This funtion takes the parameters of the spectrum and passes them on to 
    nerdustspectrum class.

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
        Return a new instance of the class nirdustspectrum with the stored spectrum.
    """

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

    step2_nuc = step1_nuc._normalization()
    step2_ext = step1_ext._normalization()

    dif = len(step2_nuc.spec1d.spectral_axis) - len(
        step2_ext.spec1d.spectral_axis
    )

    if dif == 0:

        flux_resta = (step2_nuc.spec1d.flux - step2_ext.spec1d.flux) + 1

    elif dif < 0:

        new_step2_ext = step2_ext[-dif:]
        flux_resta = (step2_nuc.spec1d.flux - new_step2_ext.spec1d.flux) + 1

    elif dif > 0:

        new_step2_nuc = step2_nuc[dif:]
        flux_resta = (new_step2_nuc.spec1d.flux - step2_ext.spec1d.flux) + 1

    substracted_1d_spectrum = su.Spectrum1D(
        flux_resta, step2_nuc.spectral_axis
    )
    kwargs = attr.asdict(step2_nuc)
    kwargs.update(spec1d=substracted_1d_spectrum)

    return NirdustSpectrum(**kwargs)._convert_to_frequency()


# ==============================================================================
# FIT SPECTRUM
# ==============================================================================
