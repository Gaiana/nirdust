#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   NirDust Project (https://github.com/Gaiana/nirdust)
# Copyright (c) 2020, 2021 Gaia Gaspar, Jose Alacoria
# License: MIT
#   Full Text: https://github.com/Gaiana/nirdust/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================

from unittest.mock import patch

from astropy.io import fits

from nirdust import io

import numpy as np

import pytest


# =============================================================================
# TESTS
# =============================================================================


def test_read_fits(mk_datapath):
    # read with no extension and wrong keyword
    file_name = mk_datapath("external_spectrum_200pc_N4945.fits")
    obj1 = io.read_fits(file_name)
    obj2 = io.read_fits(file_name, extension=0)
    np.testing.assert_almost_equal(
        obj1.spectral_axis.value, obj2.spectral_axis.value, decimal=10
    )


def test_read_table(mk_datapath):
    file_name1 = mk_datapath("NGC4945_nuclear.txt")
    file_name2 = mk_datapath("NGC4945_nuclear_noheader.txt")
    obj1 = io.read_table(file_name1)
    obj2 = io.read_table(file_name2)
    np.testing.assert_almost_equal(
        obj1.spectral_axis.value, obj2.spectral_axis.value, decimal=10
    )


def test_infer_science_extension_MEF_multiple_spectrum(mk_datapath):
    # fits with multiple extensions
    file_name = mk_datapath("external_spectrum_200pc_N4945.fits")
    with fits.open(file_name) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(data=data.copy(), header=header.copy())
    hdu2 = fits.ImageHDU(data=data.copy(), header=None)
    hdu3 = fits.ImageHDU(data=data.copy(), header=header.copy())
    hdul = fits.HDUList([hdu0, hdu1, hdu2, hdu3])

    ext_candidates = io.infer_fits_science_extension(hdul)
    assert len(ext_candidates) == 2
    np.testing.assert_array_equal(ext_candidates, np.array([1, 3]))


def test_read_fits_MEF_single_spectrum(mk_datapath):
    # fits with multiple extensions
    file_name = mk_datapath("external_spectrum_200pc_N4945.fits")
    with fits.open(file_name) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    # only one header with relevant keywords
    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(data=None)
    hdu2 = fits.ImageHDU(data=data.copy(), header=header.copy())
    hdu3 = fits.ImageHDU(data=None)
    hdul = fits.HDUList([hdu0, hdu1, hdu2, hdu3])

    with patch("astropy.io.fits.open", return_value=hdul):
        obj = io.read_fits("imaginary_file.fits")
        assert isinstance(obj, io.NirdustSpectrum)


def test_read_fits_MEF_multiple_spectrum(mk_datapath):
    # fits with multiple extensions
    file_name = mk_datapath("external_spectrum_200pc_N4945.fits")
    with fits.open(file_name) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(data=data.copy(), header=header.copy())
    hdu2 = fits.ImageHDU(data=data.copy(), header=header.copy())
    hdu3 = fits.ImageHDU(data=data.copy(), header=header.copy())
    hdul = fits.HDUList([hdu0, hdu1, hdu2, hdu3])

    with patch("astropy.io.fits.open", return_value=hdul):
        with pytest.raises(io.HeaderKeywordError):
            # Default is extension=None. this tries to detect
            # the data extension, if there are many the error is raised
            io.read_fits("imaginary_file.fits")

        # If the extension is specified it should work ok
        obj = io.read_fits("imaginary_file.fits", extension=2)
        assert isinstance(obj, io.NirdustSpectrum)


def test_calibration(mk_datapath):
    path = mk_datapath("no-calibrated_spectrum.fits")
    with pytest.raises(ValueError):
        io.read_fits(path, 0, 0)


def test_pix2wavelength(mk_datapath):
    file_name = mk_datapath("external_spectrum_400pc_N4945.fits")

    with fits.open(file_name) as hdul:
        header = hdul[0].header

    pix_0_wav = header["CRVAL1"]
    pix_disp = header["CD1_1"]

    pix_array = np.arange(50.0)
    z = 0.1

    expected = (pix_0_wav + pix_disp * pix_array) / (1 + z)
    result = io.pix2wavelength(pix_array, header, z)

    np.testing.assert_almost_equal(result, expected, decimal=10)
