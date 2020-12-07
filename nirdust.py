# ==============================================================================
# DOCS
# ==============================================================================

"""Nirdust test suite."""


# ==============================================================================
# IMPORTS
# ==============================================================================
from astropy import units as u
from astropy.io import fits
from astropy.modeling import fitting
from astropy.modeling.models import custom_model

import attr

import numpy as np

import specutils as su
import specutils.manipulation as sm

# ==============================================================================
# FUNCTIONS
# ==============================================================================


@custom_model
def normalized_blackbody(nu, T=None):
    """Normalize black-body function.

    The equetion for calculating the black body function is the same as in
    the astropy blackbody model except that the "scale" parameter is
    eliminated, i.e. is allways equal to 1.

    The normalization is performed by dividing the black body flux by its
    numerical mean.

    Parameters
    ----------
    nu: frequency axis in units of Hz.
    T: temperature of the black body.

    Return
    ------
    out: astropy model for a normalized blackbody.

    """
    from astropy.constants import h, k_B, c

    cv = c.value
    kv = k_B.value
    hv = h.value

    bb = 2 * hv * nu ** 3 / (cv ** 2 * (np.exp(hv * nu / (kv * T)) - 1))
    mean = np.mean(bb)

    return bb / mean


def blackbody_fitter(nirspec, T):
    """Fits Black-body function to spectrum.

    Parameters
    ----------
    nirspec: a NirdustSpectrum instance containing a prepared spectrum for
    blackbody fitting. Note: the spectrum must be cuted, normalized and
    corrected for stellar population contribution.

    T: temperature of the black body.

    Return
    ------
    out: tuple, best fitted model, a dictionary containing the parameters
    of the best fitted model.


    """
    bb_model = normalized_blackbody(T=T)
    fitter = fitting.LevMarLSQFitter()
    fitted_model = fitter(
        bb_model, nirspec.frequency_axis.value, nirspec.flux.value
    )

    return fitted_model, fitter.fit_info


# ==============================================================================
# CLASSES
# ==============================================================================


@attr.s(frozen=True)
class NirdustSpectrum:
    """
    Class containing a spectrum to operate with nirdust.

    Stores the spectrum in a Spectrum1D object and provides various methods
    for obtaining the dust component and prepare it for black body fitting.

    Parameters
    ----------
    header: fits header
        the header of the spectrum obtained from the fits file
    z: float
        redshift of the galaxy
        Default: 0

    spectrum_length: int
        the number of items in the spectrum axis as in len() method

    dispersion_key: float
        header keyword containing the dispersion in Å/pix

    first_wavelength: float
        header keyword containing the wavelength of first pixel

    dispersion_type: str
        header keyword containing the type of the dispersion function

    spec1d: specutils.Spectrum1D object
        containis the wavelength axis and the flux axis of the spectrum in
        unities of Å and ADU respectively

    frequency_axis: SpectralAxis object
        spectral axis in units of Hz


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
        new_len = len(cutted_spec1d.flux)
        kwargs = attr.asdict(self)
        kwargs.update(spec1d=cutted_spec1d, spectrum_length=new_len)

        return NirdustSpectrum(**kwargs)

    def convert_to_frequency(self):
        """Convert the spectral axis to frequency in units of Hz.

        Return
        ------
        out: objets NirsdustSpectrum
            New instance of the nirdustSpectrun class containing spectrum with
            a frquency axis in units of Hz.
        """
        new_axis = self.spec1d.spectral_axis.to(u.Hz)
        kwargs = attr.asdict(self)
        kwargs.update(
            frequency_axis=new_axis,
        )
        return NirdustSpectrum(**kwargs)

    def normalize(self):
        """Normalize the spectrum to the mean value.

        Return
        ------
        out: objets NirsdustSpectrum
            New instance of the nirdustSpectrun class with the flux normalized
            to the average value.
        """
        normalized_flux = self.spec1d.flux / np.mean(self.spec1d.flux)
        new_spec1d = su.Spectrum1D(normalized_flux, self.spec1d.spectral_axis)
        kwargs = attr.asdict(self)
        kwargs.update(spec1d=new_spec1d)
        return NirdustSpectrum(**kwargs)

    def fit_blackbody(self, T):
        """Call blackbody_fitter and store results in a class Storage object.

        Return
        ------
        out: objets Storage
            New instance of the Storage classe that holds the resuslts of the
            blackbody fitting.
        """
        inst = blackbody_fitter(self, T)

        storage = Storage(
            temperature=inst[0].T,
            info=inst[1],
            covariance=inst[1]["param_cov"],
            fitted_blackbody=inst[0],
        )
        return storage


class Storage:
    """

    Create the class Storage.

    Storages the results obtained with fit_blackbody.


    Atributtes:
    -----------

    temperature: Quantity that stores the temperature obtainted in the best
    black body fit in Kelvin.

    info: The fit_info dictionary contains the values returned by
    scipy.optimize.leastsq for the most recent fit, including the values from
    the infodict dictionary it returns. See the scipy.optimize.leastsq
    documentation for details on the meaning of these values.

    covariance:  the covariance matrix of the parameters as a 2D numpy array.

    fitted_blackbody: the normalized_blackbody model fot the best fit.



    """

    def __init__(self, temperature, info, covariance, fitted_blackbody):

        self.temperature = temperature.value * u.K
        self.info = info
        self.covariance = covariance
        self.fitted_blackbody = fitted_blackbody


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
    header: Header of the spectrum
    z: Redshif
    dispersion_key: keyword that gives dispersion in Å/pix
    first_wavelength: keyword that contains the wavelength of the first pixel
    dispersion_type: keyword that contains the dispersion function type

    Return
    ------
    out: objets NirsdustSpectrum
        Return a instance of the class NirdustSpectrum with the entered
        parameters.
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
    """Read a spectrum in fits format and store it in a NirdustSpectrum object.

    Parameters
    ----------
    file_name: Path to where the fits file is stored
    extension: File extension fits
    z: Redshift

    Return
    ------
    out: objets NirsdustSpectrum
        Return an instance of the class NirdustSpectrum.
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
    """Stellar Population correction.

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

    1) normalization to the mean value of the flux for both spectra
    3) substraction of the external spectrum flux from the nuclear spectrum
    flux.

    Parameters
    ----------
    nuclear_spectrum: instance of NirdusSpectrum containing the nuclear
    spectrum.

    external_spectrum: instance of NirdusSpectrum containing the external
    spectrum.

    Return
    ------
    out: objet NirsdustSpectrum
        Return a new instance of the class NirdustSpectrum containing the
        nuclear spectrum ready for black-body fitting.
    """
    normalized_nuc = nuclear_spectrum.normalize()
    normalized_ext = external_spectrum.normalize()

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
