#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   NIRDust Project (https://github.com/Gaiana/nirdust)
# Copyright (c) 2020, 2021 Gaia Gaspar, Jose Alacoria
# License: MIT
#   Full Text: https://github.com/Gaiana/nirdust/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================


from astropy import units as u
from astropy.modeling import models

from nirdust import core, preprocessing

import numpy as np


import pytest


# =============================================================================
# LINE SPECTRUM
# =============================================================================


def test_line_positions(NGC4945_continuum_rest_frame):

    sp_axis = NGC4945_continuum_rest_frame.spectral_axis
    g1 = models.Gaussian1D(0.6, 21200, 10)
    g2 = models.Gaussian1D(-0.3, 22400, 15)

    rng = np.random.default_rng(75)

    y = (
        g1(sp_axis.value)
        + g2(sp_axis.value)
        + rng.normal(0.0, 0.02, sp_axis.shape)
    )
    y_tot = (y + 0.0001 * sp_axis.value + 1000) * u.adu

    snth_line_spectrum = core.NirdustSpectrum(
        flux=y_tot,
        spectral_axis=sp_axis,
        z=0,
    ).compute_noise(23400, 24000)

    expected_positions = (
        np.array(
            [
                [g1.mean - 3 * g1.stddev, g1.mean + 3 * g1.stddev],
                [g2.mean - 3 * g2.stddev, g2.mean + 3 * g2.stddev],
            ]
        )
        * u.Angstrom
    )

    positions = preprocessing.line_spectrum(snth_line_spectrum, 5, window=80)[
        1
    ]

    np.testing.assert_almost_equal(
        positions.value * 0.001, expected_positions.value * 0.001, decimal=0
    )


def test_number_of_lines(NGC4945_continuum_rest_frame):

    sp_axis = NGC4945_continuum_rest_frame.spectral_axis
    g1 = models.Gaussian1D(0.6, 21200, 10)
    g2 = models.Gaussian1D(-0.3, 22000, 15)

    rng = np.random.default_rng(75)

    y = (
        g1(sp_axis.value)
        + g2(sp_axis.value)
        + rng.normal(0.0, 0.03, sp_axis.shape)
    )
    y_tot = (y + 0.0001 * sp_axis.value + 1000) * u.adu

    snth_line_spectrum = core.NirdustSpectrum(
        flux=y_tot,
        spectral_axis=sp_axis,
        z=0,
    ).compute_noise(23000, 24000)

    positions = preprocessing.line_spectrum(snth_line_spectrum, 5, window=80)[
        1
    ]

    assert len(positions[0]) == 2


def test_line_spectrum(NGC4945_continuum_rest_frame):

    sp_axis = NGC4945_continuum_rest_frame.spectral_axis

    g1 = models.Gaussian1D(0.6, 21200, 10)
    g2 = models.Gaussian1D(-0.3, 22000, 15)

    rng = np.random.default_rng(75)

    y = (
        g1(sp_axis.value)
        + g2(sp_axis.value)
        + rng.normal(0.0, 0.03, sp_axis.shape)
    )
    y_tot = (y + 0.0001 * sp_axis.value + 1000) * u.adu

    snth_line_spectrum = core.NirdustSpectrum(
        flux=y_tot,
        spectral_axis=sp_axis,
        z=0,
    ).compute_noise(23000, 24000)

    lines = preprocessing.line_spectrum(snth_line_spectrum, 5, window=80)[
        0
    ].flux

    all_zeros = any(lines)

    assert all_zeros is True


def test_spectral_dispersion(NGC4945_continuum_rest_frame):

    sp = NGC4945_continuum_rest_frame

    dispersion = sp.spectral_dispersion.value
    expected = sp.metadata.CD1_1

    np.testing.assert_almost_equal(dispersion, expected, decimal=14)


# ===============================================================================
# MASK SPECTRUM
# =============================================================================


def test_mask_spectrum_1(NGC4945_continuum_rest_frame):

    spectrum = NGC4945_continuum_rest_frame

    with pytest.raises(ValueError):
        spectrum.mask_spectrum(line_intervals=None, mask=None)


def test_mask_spectrum_2(NGC4945_continuum_rest_frame):

    spectrum = NGC4945_continuum_rest_frame
    line_intervals = ((20000, 20050), (21230, 21280)) * u.Angstrom
    mask = (True, True, False, False)

    with pytest.raises(ValueError):
        spectrum.mask_spectrum(line_intervals, mask)


def test_mask_spectrum_3(NGC4945_continuum_rest_frame):

    spectrum = NGC4945_continuum_rest_frame
    line_intervals = ((20000, 20050), (21230, 21280)) * u.Angstrom

    line_indexes = np.searchsorted(spectrum.spectral_axis, line_intervals)
    mask = np.ones(len(spectrum.flux), dtype=bool)

    for i, j in line_indexes:
        mask[i : j + 1] = False  # noqa

    new_flux = spectrum.flux[mask]

    masked_flux = spectrum.mask_spectrum(line_intervals).flux

    np.testing.assert_array_equal(new_flux, masked_flux, verbose=True)


def test_mask_spectrum_4(NGC4945_continuum_rest_frame):

    spectrum = NGC4945_continuum_rest_frame

    mask = np.ones(spectrum.spectral_length, dtype=bool)
    mask[100] = False

    new_flux = spectrum.flux[mask]

    masked_flux = spectrum.mask_spectrum(mask=mask).flux

    np.testing.assert_array_equal(new_flux, masked_flux, verbose=True)


def test_mask_spectrum_5(NGC4945_continuum_rest_frame):

    spectrum = NGC4945_continuum_rest_frame

    mask = np.ones(1200, dtype=bool)
    mask[100] = False

    with pytest.raises(ValueError):
        spectrum.mask_spectrum(mask=mask)


# =============================================================================
# MATCH SPECTRAL AXES
# =============================================================================


def test_spectrum_resampling_downscale():

    rng = np.random.default_rng(75)

    g1 = models.Gaussian1D(0.6, 21200, 10)
    g2 = models.Gaussian1D(-0.3, 22000, 15)

    axis = np.arange(1500, 2500, 1) * u.Angstrom

    y = g1(axis.value) + g2(axis.value) + rng.normal(0.0, 0.03, axis.shape)
    y_tot = (y + 0.0001 * axis.value + 1000) * u.adu

    low_disp_sp = core.NirdustSpectrum(
        flux=y_tot,
        spectral_axis=axis,
        z=0,
    )

    # same as axis but half the points, hence twice the dispersion
    new_axis = np.arange(1500, 2500, 2) * u.Angstrom
    new_flux = np.ones(len(new_axis)) * u.adu

    high_disp_sp = core.NirdustSpectrum(
        flux=new_flux,
        spectral_axis=new_axis,
        z=0,
    )

    # check without cleaning nan values
    f_sp, s_sp = preprocessing.match_spectral_axes(
        low_disp_sp, high_disp_sp, scaling="downscale", clean=False
    )

    assert len(f_sp.flux) == len(s_sp.flux)
    assert len(s_sp.flux) == 500

    # check cleaning nan values.
    # we know only 1 nan occurs for these spectrums
    f_sp, s_sp = preprocessing.match_spectral_axes(
        low_disp_sp, high_disp_sp, scaling="downscale", clean=True
    )

    assert len(f_sp.flux) == len(s_sp.flux)
    assert len(s_sp.flux) == 499


def test_spectrum_resampling_upscale():

    g1 = models.Gaussian1D(0.6, 21200, 10)
    g2 = models.Gaussian1D(-0.3, 22000, 15)

    axis = np.arange(1500, 2500, 1) * u.Angstrom

    rng = np.random.default_rng(75)

    y = g1(axis.value) + g2(axis.value) + rng.normal(0.0, 0.03, axis.shape)
    y_tot = (y + 0.0001 * axis.value + 1000) * u.adu

    low_disp_sp = core.NirdustSpectrum(
        flux=y_tot,
        spectral_axis=axis,
        z=0,
    )

    # same as axis but half the points, hence twice the dispersion
    new_axis = np.arange(1500, 2500, 2) * u.Angstrom
    new_flux = np.ones(len(new_axis)) * u.adu

    high_disp_sp = core.NirdustSpectrum(
        flux=new_flux,
        spectral_axis=new_axis,
        z=0,
    )

    # check without cleaning nan values
    f_sp, s_sp = preprocessing.match_spectral_axes(
        low_disp_sp, high_disp_sp, scaling="upscale", clean=False
    )

    assert len(f_sp.flux) == len(s_sp.flux)
    assert len(s_sp.flux) == 1000

    # check cleaning nan values.
    # we know only 1 nan occurs for these spectrums
    f_sp, s_sp = preprocessing.match_spectral_axes(
        low_disp_sp, high_disp_sp, scaling="upscale", clean=True
    )

    assert len(f_sp.flux) == len(s_sp.flux)
    assert len(s_sp.flux) == 999


def test_spectrum_resampling_invalid_scaling():
    with pytest.raises(ValueError):
        preprocessing.match_spectral_axes(
            None,
            None,
            scaling="equal",
        )


def test_spectrum_resampling_upscale_second_if():

    g1 = models.Gaussian1D(0.6, 21200, 10)
    g2 = models.Gaussian1D(-0.3, 22000, 15)

    axis = np.arange(1500, 2500, 1) * u.Angstrom

    rng = np.random.default_rng(75)

    y = g1(axis.value) + g2(axis.value) + rng.normal(0.0, 0.03, axis.shape)
    y_tot = (y + 0.0001 * axis.value + 1000) * u.adu

    low_disp_sp = core.NirdustSpectrum(
        flux=y_tot,
        spectral_axis=axis,
        z=0,
    )

    # same as axis but half the points, hence twice the dispersion
    new_axis = np.arange(1500, 2500, 2) * u.Angstrom
    new_flux = np.ones(len(new_axis)) * u.adu

    high_disp_sp = core.NirdustSpectrum(
        flux=new_flux,
        spectral_axis=new_axis,
        z=0,
    )

    # check without cleaning nan values
    f_sp, s_sp = preprocessing.match_spectral_axes(
        low_disp_sp, high_disp_sp, scaling="upscale", clean=False
    )

    assert len(f_sp.flux) == len(s_sp.flux)
    assert len(s_sp.flux) == 1000

    # check cleaning nan values.
    # we know only 1 nan occurs for these spectrums
    f_sp, s_sp = preprocessing.match_spectral_axes(
        high_disp_sp, low_disp_sp, scaling="upscale", clean=True
    )

    assert len(f_sp.flux) == len(s_sp.flux)
    assert len(s_sp.flux) == 999


def test_match_spectral_axes_first_if(NGC4945_continuum_rest_frame):
    # tests the case where the first spectrum is the largest

    first_sp = NGC4945_continuum_rest_frame
    second_sp = NGC4945_continuum_rest_frame.cut_edges(22000, 23000)

    new_first_sp, new_second_sp = preprocessing.match_spectral_axes(
        first_sp, second_sp
    )

    assert new_first_sp.spectral_length == new_second_sp.spectral_length


def test_match_spectral_axes_second_if(NGC4945_continuum_rest_frame):

    first_sp = NGC4945_continuum_rest_frame.cut_edges(22000, 23000)
    second_sp = NGC4945_continuum_rest_frame

    new_first_sp, new_second_sp = preprocessing.match_spectral_axes(
        first_sp, second_sp
    )

    assert new_first_sp.spectral_length == new_second_sp.spectral_length
