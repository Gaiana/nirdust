# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import pathlib

import pytest

import nirdust as nd


# ==============================================================================
# CONSTANTS
# ==============================================================================

PATH = os.path.abspath(os.path.dirname(__file__))

TEST_PATH = pathlib.Path(PATH) / "test_data"

# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture(scope="session")
def cont03():
    path = TEST_PATH / "cont03.fits"
    #spect = nd.NuclearSpectrum(path, 0)
    spect = nd.from_filename(path)
    return spect


# ==============================================================================
# TESTS
# ==============================================================================

def test_match(cont03):
    spectrum = cont03
    assert spectrum.spectral_axis.shape == spectrum.flux.shape


def test_header(cont03):
    spectrum = cont03
    assert isinstance(spectrum.header['EXPTIME'], (float, int))
    assert spectrum.header['EXPTIME'] >= 0.


def test_wav_axis(cont03):
    spectrum = cont03
    assert spectrum.header['CD1_1'] >= 0.
    assert spectrum.header['CTYPE1'] == 'LINEAR'
