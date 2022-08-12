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

import os
import pathlib

from astropy import units as u
from astropy.io import fits
from astropy.modeling.models import BlackBody

import nirdust as nd

import numpy as np

import pytest


# =============================================================================
# CONSTANTS
# =============================================================================

PATH = os.path.abspath(os.path.dirname(__file__))

TEST_DATA_PATH = pathlib.Path(PATH) / "data"

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def mk_datapath():
    def make(filename):
        return TEST_DATA_PATH / filename

    return make


@pytest.fixture(scope="session")
def header_of(mk_datapath):
    def make(filename):
        file_name = mk_datapath(filename)
        with fits.open(file_name) as hdul:
            return hdul[0].header

    return make


@pytest.fixture(scope="session")
def NGC4945_continuum(mk_datapath):
    file_name = mk_datapath("cont03.fits")
    spect = nd.read_fits(file_name, 0, z=0.00188)
    return spect


@pytest.fixture(scope="session")
def NGC4945_continuum_rest_frame(mk_datapath):
    file_name = mk_datapath("cont03.fits")
    spect = nd.read_fits(file_name, 0, z=0)
    return spect


@pytest.fixture(scope="session")
def NGC4945_external_continuum_400pc(mk_datapath):
    file_name = mk_datapath("external_spectrum_400pc_N4945.fits")
    spect = nd.read_fits(file_name, 0, z=0.00188)
    return spect


@pytest.fixture(scope="session")
def NGC4945_external_continuum_200pc(mk_datapath):
    file_name = mk_datapath("external_spectrum_200pc_N4945.fits")
    spect = nd.read_fits(file_name, 0, z=0.00188)
    return spect


@pytest.fixture(scope="session")
def NGC4945_external_with_lines_200pc(mk_datapath):
    file_name = mk_datapath("External_with_lines.fits")
    spect = nd.read_fits(file_name, 0, z=0.00188)
    return spect


@pytest.fixture(scope="session")
def NGC4945_nuclear_with_lines(mk_datapath):
    file_name = mk_datapath("NuclearNGC4945.fits")
    spect = nd.read_fits(file_name, 0, z=0.00188)
    return spect


@pytest.fixture(scope="session")
def NGC3998_sp_lower_resolution(mk_datapath):
    file_name = mk_datapath("ngc3998-sfin.fits")
    spect = nd.read_fits(file_name, 1, z=0.00350)
    return spect


@pytest.fixture(scope="session")
def continuum01(mk_datapath):
    file_name = mk_datapath("cont01.fits")
    return nd.read_fits(file_name, 0, z=0.00188)


# =============================================================================
# SYNTHETIC MODEL FIXTURES
# =============================================================================


def gaussian_noise(signal, snr, seed):
    """Add gaussian noise to signal given a SNR value."""
    rng = np.random.default_rng(seed=seed)
    sigma = np.mean(signal) / snr
    noise = rng.normal(0, sigma, len(signal))
    return signal + noise


def add_noise(spectrum, snr, seed):
    flux = gaussian_noise(spectrum.flux.value, snr, seed)

    return nd.NirdustSpectrum(
        spectral_axis=spectrum.spectral_axis,
        flux=u.Quantity(flux, u.adu),
        z=spectrum.z,
    )


@pytest.fixture
def with_noise():
    def foo(spectrum, snr, seed):
        return add_noise(spectrum, snr, seed)

    return foo


@pytest.fixture
def true_params():
    return {"T": 750 * u.K, "alpha": 15.0, "beta": 8.3, "gamma": -3.3}


@pytest.fixture
def synth_spectral_axis():
    return np.linspace(20000.0, 24000, 200) * u.AA


@pytest.fixture
def synth_blackbody(synth_spectral_axis, true_params):

    sinthetic_model = BlackBody(true_params["T"])
    sinthetic_flux = sinthetic_model(synth_spectral_axis)
    sinthetic_flux *= u.adu

    return nd.NirdustSpectrum(
        spectral_axis=synth_spectral_axis,
        flux=sinthetic_flux,
        z=0,
    )


@pytest.fixture
def synth_nuclear(synth_blackbody, true_params):

    # BlackBody model
    bb = (10 ** true_params["beta"]) * synth_blackbody.flux.value

    # Linear model
    def tp_line(x, x1, x2, y1, y2):
        return (y2 - y1) / (x2 - x1) * (x - x1) + y1

    wave = synth_blackbody.spectral_axis.value
    delta_bb = bb[-1] - bb[0]
    y1_line, y2_line = bb[0] + 6 / 5 * delta_bb, bb[0] + 2 / 5 * delta_bb

    nuclear = tp_line(wave, wave[0], wave[-1], y1_line, y2_line)
    nuclear *= u.adu

    return nd.NirdustSpectrum(
        spectral_axis=synth_blackbody.spectral_axis,
        flux=nuclear,
        z=0,
    )


@pytest.fixture
def synth_external(synth_nuclear, true_params):

    flux = synth_nuclear.flux.value / true_params["alpha"]
    flux *= u.adu

    return nd.NirdustSpectrum(
        spectral_axis=synth_nuclear.spectral_axis,
        flux=flux,
        z=0,
    )


@pytest.fixture
def synth_external_noised(synth_external):
    return add_noise(synth_external, snr=1000, seed=35)


@pytest.fixture
def synth_total(synth_nuclear, synth_blackbody, true_params):

    bb_flux = 10 ** true_params["beta"] * synth_blackbody.flux.value
    flux = synth_nuclear.flux.value + bb_flux + 10 ** true_params["gamma"]
    flux *= u.adu

    return nd.NirdustSpectrum(
        spectral_axis=synth_nuclear.spectral_axis,
        flux=flux,
        z=0,
    )


@pytest.fixture
def synth_total_noised(synth_total):
    return add_noise(synth_total, snr=1000, seed=234)


#################################################
#################################################
#################################################
#################################################
#################################################
#################################################
#################################################

# @pytest.fixture
# def axis():
#     return np.linspace(20100, 23000, 150) * u.AA

# @pytest.fixture
# def synth_blackbody(axis, true_params):

#     sinthetic_model = BlackBody(true_params["T"])
#     sinthetic_flux = sinthetic_model(axis)
#     sinthetic_flux *= u.adu

#     return nd.NirdustSpectrum(
#         spectral_axis=axis,
#         flux=sinthetic_flux,
#         z=0,
#     )


# @pytest.fixture
# def synth_nuclear(synth_blackbody, true_params):

#     # BlackBody model
#     bb = (10**true_params["beta"]) * synth_blackbody.flux.value

#     # Linear model
#     def tp_line(x, x1, x2, y1, y2):
#         return (y2 - y1) / (x2 - x1) * (x - x1) + y1

#     wave = synth_blackbody.spectral_axis.to(u.AA).value
#     delta_bb = bb[-1] - bb[0]
#     y1_line, y2_line = bb[0] + 6 / 5 * delta_bb, bb[0] + 2 / 5 * delta_bb

#     nuclear = tp_line(wave, wave[0], wave[-1], y1_line, y2_line)
#     nuclear *= u.adu

#     return nd.NirdustSpectrum(
#         spectral_axis=synth_blackbody.spectral_axis,
#         flux=nuclear,
#         z=0,
#     )


# @pytest.fixture
# def synth_external(synth_nuclear, true_params):

#     flux = synth_nuclear.flux.value / true_params["alpha"]
#     flux *= u.adu

#     return nd.NirdustSpectrum(
#         spectral_axis=synth_nuclear.spectral_axis,
#         flux=flux,
#         z=0,
#     )


# @pytest.fixture
# def synth_external_noised(synth_external):
#     return add_noise(synth_external, snr=1000, seed=35)


# @pytest.fixture
# def synth_total(synth_nuclear, synth_blackbody, true_params):

#     bb_flux = 10 ** true_params["beta"] * synth_blackbody.flux.value
#     flux = synth_nuclear.flux.value + bb_flux + 10 ** true_params["gamma"]
#     flux *= u.adu

#     return nd.NirdustSpectrum(
#         spectral_axis=synth_nuclear.spectral_axis,
#         flux=flux,
#         z=0,
#     )


# @pytest.fixture
# def synth_total_noised(synth_total):
#     return add_noise(synth_total, snr=1000, seed=234)
