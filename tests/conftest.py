#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   NirDust Project (https://github.com/Gaiana/nirdust)
# Copyright (c) 2020, 2021 Gaia Gaspar, Jose Alacoria
# License: MIT
#   Full Text: https://github.com/Gaiana/nirdust/LICENSE

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import pathlib

from astropy import units as u

import nirdust as nd

import numpy as np

import pytest


# ==============================================================================
# CONSTANTS
# ==============================================================================

PATH = os.path.abspath(os.path.dirname(__file__))

TEST_DATA_PATH = pathlib.Path(PATH) / "data"

# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture(scope="session")
def test_data_path():
    def make(filename):
        return TEST_DATA_PATH / filename

    return make


@pytest.fixture(scope="session")
def NGC4945_continuum(test_data_path):
    file_name = test_data_path("cont03.fits")
    spect = nd.read_spectrum(file_name, 0, z=0.00188)
    return spect


@pytest.fixture(scope="session")
def NGC4945_continuum_rest_frame(test_data_path):
    file_name = test_data_path("cont03.fits")
    spect = nd.read_spectrum(file_name, 0, z=0)
    return spect


@pytest.fixture(scope="session")
def NGC4945_external_continuum_400pc(test_data_path):
    file_name = test_data_path("external_spectrum_400pc_N4945.fits")
    spect = nd.read_spectrum(file_name, 0, z=0.00188)
    return spect


@pytest.fixture(scope="session")
def NGC4945_external_continuum_200pc(test_data_path):
    file_name = test_data_path("external_spectrum_200pc_N4945.fits")
    spect = nd.read_spectrum(file_name, 0, z=0.00188)
    return spect


@pytest.fixture(scope="session")
def NGC4945_external_with_lines_200pc(test_data_path):
    file_name = test_data_path("External_with_lines.fits")
    spect = nd.read_spectrum(file_name, 0, z=0.00188)
    return spect


@pytest.fixture(scope="session")
def NGC4945_nuclear_with_lines(test_data_path):
    file_name = test_data_path("NuclearNGC4945.fits")
    spect = nd.read_spectrum(file_name, 0, z=0.00188)
    return spect


@pytest.fixture(scope="session")
def NGC3998_sp_lower_resolution(test_data_path):
    file_name = test_data_path("ngc3998-sfin.fits")
    spect = nd.read_spectrum(file_name, 1, z=0.00350)
    return spect


@pytest.fixture(scope="session")
def continuum01(test_data_path):
    file_name = test_data_path("cont01.fits")
    return nd.read_spectrum(file_name, 0, z=0.00188)


@pytest.fixture(scope="session")
def snth_spectrum_1000(NGC4945_continuum_rest_frame):

    real_spectrum = NGC4945_continuum_rest_frame
    freq_axis = real_spectrum.spec1d_.spectral_axis.to(u.Hz)
    sinthetic_model = nd.NormalizedBlackBody(1000)
    sinthetic_flux = sinthetic_model(freq_axis.value)

    mu, sigma = 0, 0.1
    nd_random = np.random.default_rng(50)
    gaussian_noise = nd_random.normal(mu, sigma, len(freq_axis))
    noisy_model = sinthetic_flux * (1 * u.adu + gaussian_noise * u.adu)

    dispersion = 3.51714285129581
    first_wave = 18940.578099674

    spectrum_length = len(real_spectrum.flux)
    spectral_axis = (
        first_wave + dispersion * np.arange(0, spectrum_length)
    ) * u.AA

    return nd.NirdustSpectrum(
        flux=noisy_model,
        spectral_axis=spectral_axis,
        z=0,
    )
