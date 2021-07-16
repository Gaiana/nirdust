#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   NirDust Project (https://github.com/Gaiana/nirdust)
# Copyright (c) 2020, 2021 Gaia Gaspar, Jose Alacoria
# License: MIT
#   Full Text: https://github.com/Gaiana/nirdust/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Nirdust Input/Output."""


# =============================================================================
# IMPORTS
# =============================================================================


from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

import numpy as np


from .core import NirdustSpectrum

# =============================================================================
# EXCEPTIONS
# =============================================================================


class HeaderKeywordError(KeyError):
    """Raised when header keyword not found."""

    pass


# =============================================================================
# LOAD SPECTRA FROM FITS
# =============================================================================


def infer_fits_science_extension(hdulist):
    """Auto detect fits science extension using the provided keywords.

    Parameters
    ----------
    hdulist: `~astropy.io.fits.HDUList`
        Object containing the FITS extensions.

    Return
    ------
    extensions: `~numpy.array`
        Array with the science extensions indeces in the hdulist.
    """
    if len(hdulist) == 1:
        return np.array([0])

    keys = {"CRVAL1"}  # keywords that are present in science extensions
    extl = []
    for ext, hdu in enumerate(hdulist):
        if keys.issubset(hdu.header.keys()):
            extl.append(ext)

    return np.array(extl)


def pix2wavelength(pix_arr, header, z=0):
    """Transform pixel to wavelength.

    This function uses header information to perform WCS transformation.

    Parameters
    ----------
    pix_arr: float or `~numpy.ndarray`
        Array of pixels values.

    header: FITS header
        Header of the spectrum.

    z: float
        Redshift of object. Use for the scale factor 1 / (1 + z).

    Return
    ------
    wavelength: `~numpy.ndarray`
        Array with the spectral axis.
    """
    wcs = WCS(header, naxis=1, relax=False, fix=False)
    wave_arr = wcs.wcs_pix2world(pix_arr, 0)[0]
    scale_factor = 1 / (1 + z)
    return wave_arr * scale_factor


def spectrum(flux, header, z=0):
    """Instantiate a NirdustSpectrum object from FITS parameters.

    Parameters
    ----------
    flux: Quantity
        Intensity for each pixel in arbitrary units.

    header: FITS header
        Header of the spectrum.

    z: float
        Redshif of the galaxy.

    Return
    ------
    spectrum: ``NirsdustSpectrum``
        Return a instance of the class NirdustSpectrum with the entered
        parameters.
    """
    # unit should be the same as first_wavelength and dispersion_key, AA ?
    pixel_axis = np.arange(len(flux))
    spectral_axis = pix2wavelength(pixel_axis, header, z) * u.AA
    return NirdustSpectrum(
        flux=flux,
        spectral_axis=spectral_axis,
        z=z,
        metadata=header,
    )


def read_fits(file_name, extension=None, z=0):
    """Read a spectrum in FITS format and store it in a NirdustSpectrum object.

    Parameters
    ----------
    file_name: str
        Path to where the fits file is stored.

    extension: int or str
        Extension of the FITS file where the spectrum is stored. If None the
        extension will be automatically identified by searching relevant
        header keywords. Default is None.

    z: float
        Redshift of the galaxy. Used to scale the spectral axis with the
        cosmological sacle factor 1/(1+z). Default is 0.

    Return
    ------
    out: NirsdustSpectrum object
        Returns an instance of the class NirdustSpectrum.
    """
    with fits.open(file_name) as hdulist:

        if extension is None:
            ext_candidates = infer_fits_science_extension(hdulist)
            if len(ext_candidates) > 1:
                raise HeaderKeywordError(
                    "More than one extension with relevant keywords. "
                    "Please specify the extension."
                )
            extension = ext_candidates[0]

        flux = hdulist[extension].data
        header = hdulist[extension].header

    return spectrum(flux, header, z)


# =============================================================================
# LOAD SPECTRA FROM TABLE
# =============================================================================


def read_table(
    file_name,
    wavelength_column=0,
    flux_column=1,
    format="ascii",
    z=0,
    **kwargs,
):
    """Read a spectrum from a table and store it in a NirdustSpectrum object.

    The table must contain two columns for the wavelength and the
    intensity/flux, the column number can be specified by parameters.  It is
    assumed that the unit of the wavelength axis is Angstroms.

    Parameters
    ----------
    file_name: str
        Path to where the fits file is stored.

    wavelength_column: int
        The positional number of the wavelengh column. Default is 0.

    flux_column: int
        The positional number of the intensity/flux column. Default is 1.

    kwargs:
        Se pasa directo a ``astropy.table.Table.read``.

    Return
    ------
    out: NirsdustSpectrum object
        Returns an instance of the class NirdustSpectrum.
    """
    table = Table.read(file_name, format=format, **kwargs)
    wavelength = table.columns[wavelength_column]
    flux = table.columns[flux_column]

    spectral_axis = wavelength * u.AA

    metadata = {
        "file_name": file_name,
        "wavelength_column": wavelength_column,
        "flux_column": flux_column,
        "format": format,
    }
    metadata.update(kwargs)

    return NirdustSpectrum(
        flux=flux,
        spectral_axis=spectral_axis,
        z=z,
        metadata=metadata,
    )
