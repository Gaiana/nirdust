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
def cont03():
    file_name = TEST_PATH / "cont03.fits"
    spect = nd.read_single_spectrum(file_name, 0)
    return spect


@pytest.fixture(scope="session")
def sample1():
    files = str(TEST_PATH / "cont*.fits")
    lista = nd.read_sample(files, 0)
    return lista


# ==============================================================================
# TESTS
# ==============================================================================


def test_match(cont03):
    spectrum = cont03
    assert spectrum.spectral_axis.shape == spectrum.flux.shape


def test_header(cont03):
    spectrum = cont03
    assert isinstance(spectrum.header["EXPTIME"], (float, int))
    assert spectrum.header["EXPTIME"] >= 0.0


def test_wav_axis(cont03):
    spectrum = cont03
    assert spectrum.header["CRVAL1"] >= 0.0
    assert spectrum.header["CTYPE1"] == "LINEAR"


def test_sample_size(sample1):
    spec_list = sample1
    assert len(spec_list) == 6
