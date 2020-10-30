# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import pathlib

import nirdust as nd

import pytest

# ==============================================================================
# CONSTANTS
# ==============================================================================

PATH = os.path.abspath(os.path.dirname(__file__))

TEST_PATH = pathlib.Path(PATH) / "test_data"

# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture(scope="session")
def NGC4945_continuum():
    file_name = TEST_PATH / "cont03.fits"
    spect = nd.read_spectrum(file_name, 0)
    return spect


# ==============================================================================
# TESTS
# ==============================================================================


def test_match(NGC4945_continuum):
    spectrum = NGC4945_continuum
    assert spectrum.spectral_axis.shape == spectrum.flux.shape


def test_header(NGC4945_continuum):
    spectrum = NGC4945_continuum
    assert isinstance(spectrum.header["EXPTIME"], (float, int))
    assert spectrum.header["EXPTIME"] >= 0.0


def test_wav_axis(NGC4945_continuum):
    spectrum = NGC4945_continuum
    assert spectrum.header["CRVAL1"] >= 0.0
    assert spectrum.header["CTYPE1"] == "LINEAR"
