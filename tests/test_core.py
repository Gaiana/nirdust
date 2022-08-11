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

from astropy import units as u

from nirdust import core

import numpy as np

import pytest

import specutils as su

# =============================================================================
# TEST METADATA
# =============================================================================


def test_metadata_creation():
    md = core._NDSpectrumMetadata({"alfa": 1})
    assert md["alfa"] == md.alfa == 1
    assert len(md) == 1


def test_metadata_creation_empty():
    md = core._NDSpectrumMetadata({})
    assert len(md) == 0


def test_metadata_key_notfound():
    md = core._NDSpectrumMetadata({"alfa": 1})
    assert md["alfa"] == md.alfa == 1
    with pytest.raises(KeyError):
        md["bravo"]


def test_metadata_attribute_notfound():
    md = core._NDSpectrumMetadata({"alfa": 1})
    assert md["alfa"] == md.alfa == 1
    with pytest.raises(AttributeError):
        md.bravo


def test_metadata_iter():
    md = core._NDSpectrumMetadata({"alfa": 1})
    assert list(iter(md)) == ["alfa"]


def test_metadata_repr():
    md = core._NDSpectrumMetadata({"alfa": 1})
    assert repr(md) == "metadata({'alfa'})"


def test_metadata_dir():
    md = core._NDSpectrumMetadata({"alfa": 1})
    assert "alfa" in dir(md)


def test_metadata_creation_fitst_header_fits_header(header_of):
    header = header_of("external_spectrum_200pc_N4945.fits")
    md = core._NDSpectrumMetadata(header)
    assert md["AMPDAC21"] == md.AMPDAC21 == 180.0
    assert len(md) == len(header)


def test_metadata_key_notfound_fits_header(header_of):
    header = header_of("external_spectrum_200pc_N4945.fits")
    md = core._NDSpectrumMetadata(header)
    assert md["AMPDAC21"] == md.AMPDAC21 == 180.0
    with pytest.raises(KeyError):
        md["bravo"]


def test_metadata_attribute_notfound_fits_header(header_of):
    header = header_of("external_spectrum_200pc_N4945.fits")
    md = core._NDSpectrumMetadata(header)
    assert md["AMPDAC21"] == md.AMPDAC21 == 180.0
    with pytest.raises(AttributeError):
        md.bravo


def test_metadata_iter_fits_header(header_of):
    header = header_of("external_spectrum_200pc_N4945.fits")
    md = core._NDSpectrumMetadata(header)
    assert list(iter(md)) == list(iter(header))


def test_metadata_repr_fits_header(header_of):
    header = header_of("external_spectrum_200pc_N4945.fits")
    md = core._NDSpectrumMetadata(header)
    assert repr(md) == f"metadata({set(header)})"


def test_metadata_dir_fits_header(header_of):
    header = header_of("external_spectrum_200pc_N4945.fits")
    md = core._NDSpectrumMetadata(header)
    for elem in header:
        assert elem in dir(md)


# =============================================================================
# TESTS
# =============================================================================


def test_property_freq_axis(NGC4945_continuum):
    x = NGC4945_continuum.spectral_axis.to(u.Hz, equivalencies=u.spectral())
    np.testing.assert_allclose(x.value, NGC4945_continuum.frequency_axis.value)


def test_spectrum_repr(NGC4945_external_continuum_200pc):
    result = repr(NGC4945_external_continuum_200pc)
    expected = "NirdustSpectrum(z=0.00188, spectral_length=1751, spectral_range=[18905.11-25048.53] Angstrom)"  # noqa
    assert result == expected


def test_spectrum_dir(NGC4945_external_continuum_200pc):
    obj1 = NGC4945_external_continuum_200pc
    result = dir(obj1)
    expected = dir(obj1.spec1d_)
    assert not set(expected).difference(result)


def test_match(NGC4945_continuum):
    spectrum = NGC4945_continuum
    assert spectrum.spectral_axis.shape == spectrum.flux.shape


def test_fits_header(NGC4945_continuum):
    spectrum = NGC4945_continuum
    assert isinstance(spectrum.metadata["EXPTIME"], (float, int))
    assert spectrum.metadata["EXPTIME"] >= 0.0
    assert spectrum.metadata["EXPTIME"] == spectrum.metadata.EXPTIME


def test_wav_axis(NGC4945_continuum):
    spectrum = NGC4945_continuum
    assert spectrum.metadata.CRVAL1 >= 0.0
    assert spectrum.metadata.CTYPE1 == "LINEAR"


def test_redshift_correction(NGC4945_continuum):
    spectrum = NGC4945_continuum
    assert (
        spectrum.spectral_axis[0]
        == (spectrum.metadata.CRVAL1 / (1 + 0.00188)) * u.AA
    )


def test_convert_to_frequency(NGC4945_continuum):
    spectrum = NGC4945_continuum
    freq = spectrum.convert_to_frequency().spectral_axis
    np.testing.assert_almost_equal(
        freq.value.mean(), 137313317328585.02, decimal=7
    )


def test_getattr(NGC4945_continuum):
    spectrum = NGC4945_continuum

    # primero tiramos el dir original a spectrum y a spec1d
    original_dir = dir(super(type(spectrum), spectrum))
    spec1d_dir = dir(spectrum.spec1d_)

    # garantizamos que redshift no este oculto por un atributo de spectum
    # y que si este en spectrum.spect1d_
    assert "redshift" not in original_dir and "redshift" in spec1d_dir

    # ahora si probamos el funcionamiento
    np.testing.assert_array_equal(spectrum.spec1d_.redshift, spectrum.redshift)


def test_slice(NGC4945_continuum):
    spectrum = NGC4945_continuum
    result = spectrum[20:40]
    np.testing.assert_array_equal(
        result.spec1d_.flux, spectrum.spec1d_[20:40].flux
    )


def test_len(NGC4945_continuum):
    sp = NGC4945_continuum
    assert len(sp) == len(sp.spectral_axis)


def test_unit():
    a = np.arange(0, 100, 1) * u.Angstrom
    f = np.arange(0, 200, 2) * u.adu

    sp = core.NirdustSpectrum(a, f)

    assert sp.unit == "adu", "Angstrom"


def test_cut_edges(NGC4945_continuum):
    spectrum = NGC4945_continuum
    region = su.SpectralRegion(20000 * u.AA, 23000 * u.AA)
    expected = su.manipulation.extract_region(spectrum.spec1d_, region)
    result = spectrum.cut_edges(20000, 23000)
    np.testing.assert_array_equal(result.spec1d_.flux, expected.flux)


def test_normalize(NGC4945_continuum):
    spectrum = NGC4945_continuum
    normalized_spectrum = spectrum.normalize()
    mean = np.mean(normalized_spectrum.spec1d_.flux)

    assert mean == u.Quantity(1.0)


def test_compute_noise(NGC4945_continuum):
    wave = np.arange(23000, 25000, 0.1) * u.Angstrom

    flux = 10 * u.adu
    rng = np.random.default_rng(5)
    gnoise = rng.normal(0, 0.5, len(wave))
    noisy_thing = flux + gnoise * u.adu

    noise_sp = core.NirdustSpectrum(wave, noisy_thing)

    low_lim = wave[1].value
    up_lim = wave[-2].value

    noise = noise_sp.compute_noise(low_lim, up_lim).noise

    expected = np.std(noisy_thing).value

    np.testing.assert_almost_equal(noise, expected, decimal=5)


def test_metadata_update_regions():

    wave = np.arange(22000, 25000, 3.5) * u.Angstrom
    flux = 10 * u.adu * np.ones(len(wave))
    spectrum = core.NirdustSpectrum(flux=flux, spectral_axis=wave)

    n_spectrum = spectrum.compute_noise(24000, 24500)

    dic = core.public_members_asdict(n_spectrum)
    meta = dic["metadata"]

    assert meta["nr_low_lim"] == 24000
    assert meta["nr_upper_lim"] == 24500


def test_value_error_compute_noise():

    wave = np.arange(22000, 25000, 3.5) * u.Angstrom
    flux = 10 * u.adu * np.ones(len(wave))
    spectrum = core.NirdustSpectrum(flux=flux, spectral_axis=wave)

    with pytest.raises(ValueError):
        spectrum.compute_noise(21000, 24500)


def test_value_error_compute_noise2():

    wave = np.arange(22000, 25000, 3.5) * u.Angstrom
    flux = 10 * u.adu * np.ones(len(wave))
    spectrum = core.NirdustSpectrum(flux=flux, spectral_axis=wave)

    with pytest.raises(ValueError):
        spectrum.compute_noise(24500, 24000)
