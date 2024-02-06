#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   NIRDust Project (https://github.com/Gaiana/nirdust)
# Copyright (c) 2020, 2021 Gaia Gaspar, Jose Alacoria
# License: MIT
#   Full Text: https://github.com/Gaiana/nirdust/LICENSE

# ==============================================================================
# DOCS
# ==============================================================================

"""Collection of preprocessing utilities."""


# ==============================================================================
# IMPORTS
# ==============================================================================

import warnings

from astropy import units as u
from astropy.modeling import fitting, models
from astropy.nddata import StdDevUncertainty

import numpy as np

from specutils.fitting import find_lines_threshold
from specutils.fitting import fit_generic_continuum
from specutils.fitting import fit_lines
from specutils.manipulation import FluxConservingResampler
from specutils.spectra import Spectrum1D

from . import core


# ==============================================================================
# RESAMPLE SPECTRA TO MATCH SPECTRAL RESOLUTIONS
# ==============================================================================


def _rescale(sp, reference_sp):
    """Resample a given spectrum to a reference spectrum.

    Notes
    -----
    `nan` values may occur at the edges where the resampler is forced
    to extrapolate.
    """
    input_sp1d = sp.spec1d_
    resample_axis = reference_sp.spectral_axis

    resampler = FluxConservingResampler(extrapolation_treatment="nan_fill")
    output_sp1d = resampler(input_sp1d, resample_axis)

    kwargs = core.public_members_asdict(sp)
    kwargs.update(
        flux=output_sp1d.flux,
        spectral_axis=output_sp1d.spectral_axis,
    )
    return core.NirdustSpectrum(**kwargs)


def _clean_and_match(sp1, sp2):
    """Clean `nan` values and apply the same mask to both spectrums."""
    # nan values occur in the flux variable
    # check for invalid values in both spectrums
    mask = np.isfinite(sp1.flux) & np.isfinite(sp2.flux)

    sp_list = []
    for sp in [sp1, sp2]:
        kw = core.public_members_asdict(sp)
        kw.update(flux=sp.flux[mask], spectral_axis=sp.spectral_axis[mask])
        sp_list.append(core.NirdustSpectrum(**kw))

    return sp_list


def match_spectral_axes(
    first_sp,
    second_sp,
    scaling="downscale",
    clean=True,
):
    """Resample the higher resolution spectrum.

    `Spectrum_resampling` uses the `spectral_axis` of one imput spectrum to
    resample the `spectral_axis` of the otherone, depending on the `scaling`
    parameter.
    To do so this function uses the `FluxConservingResampler` class of
    `Specutils`. The order of the input spectra is arbitrary and the order in
    the output is the same as in the input. It is recomended to run this
    function after the class methods 'cut_edges' and 'mask_spectrum'.

    Parameters
    ----------
    first_sp: `NirdustSpectrum` object

    second_sp: `NirdustSpectrum` object

    scaling: string
        If `downscale` the higher resolution spectrum will be resampled to
        match the lower resolution spectrum. If `upscale` the lower resolution
        spectrum will be resampled to match the higher resolution spectrum.

    clean: bool
        Flag to indicate if the spectrums have to be cleaned by `nan` values
        after the rescaling procedure. `nan` values occur at the edges of the
        resampled spectrum when it is forced to extrapolate beyond the
        spectral range of the reference spectrum.

    Return
    ------
    out: `NirdustSpectrum`, `NirdustSpectrum`

    """
    scaling = scaling.lower()
    if scaling not in ["downscale", "upscale"]:
        raise ValueError(
            "Unknown scaling mode. Must be 'downscale' or 'upscale'."
        )

    first_disp = first_sp.spectral_dispersion
    second_disp = second_sp.spectral_dispersion

    dispersion_difference = (first_disp - second_disp).value

    # Larger numerical dispersion means lower resolution!
    if dispersion_difference > 0:
        # Check type of rescaling
        if scaling == "downscale":
            second_sp = _rescale(second_sp, reference_sp=first_sp)
        else:
            first_sp = _rescale(first_sp, reference_sp=second_sp)

    elif dispersion_difference < 0:
        if scaling == "downscale":
            first_sp = _rescale(first_sp, reference_sp=second_sp)
        else:
            second_sp = _rescale(second_sp, reference_sp=first_sp)

    # else:
    #     # they have the same dispersion, is that equivalent
    #     # to equal spectral_axis?
    #     pass

    if clean:
        first_sp, second_sp = _clean_and_match(first_sp, second_sp)

    return first_sp, second_sp


# ==============================================================================
# FIND LINE INTERVALS FROM AUTHOMATIC LINE FITTING
# ==============================================================================


def _make_window(center, delta):
    """Create window array."""
    return np.array([center - delta, center + delta])


def line_spectrum(
    spectrum,
    noise_factor=3,
    window=50,
):
    """Construct the line spectrum.

    Uses various `Specutils` features to fit the continuum of the spectrum,
    subtract it and find the emission and absorption lines.
    Then fits all the lines with gaussian models to construct the line
    spectrum using `astropy.models.Gaussian1D`.

    Parameters
    ----------
    spectrum: `NirdustSpectrum` object
        A spectrum stored in a `NirdustSpectrum` class object.

    noise_factor: float
        Same parameter as in `specutils.fitting.find_lines_threshold`.
        Factor multiplied by the spectrum’s`uncertainty`, used for
        thresholding. Default is 3.
    window: float
        Same parameter as in `specutils.fitting.fit_lines`. Width of the region
        around each line of the spectrum to use in the fitting. If None, then
        the whole spectrum will be used in the fitting. `Window` is used in the
        Gaussian fitting of the spectral lines. Default is 50 (Å).

    Return
    ------
    out: NirdustSpectrum, Quantity
        Returns in the first element a NirdustSpectrum of the same lenght as
        the original spectrum containing the fitted lines. In the 2nd position,
        returns the intervals where those lines were found determined by
        3-sigma values around the center of the line.
    """
    # values in correct units
    window = u.Quantity(window, u.AA)

    # By defaults this fits a Chebyshev of order 3 to the flux
    with warnings.catch_warnings():  # Ignore warnings
        warnings.simplefilter("ignore")
        model = fit_generic_continuum(
            spectrum.spec1d_, fitter=fitting.LinearLSQFitter()
        )
    continuum = model(spectrum.spectral_axis)
    new_flux = spectrum.spec1d_ - continuum

    noise = spectrum.noise * np.ones(len(spectrum.flux))
    uncertainty = StdDevUncertainty(noise)
    noise_spectrum = Spectrum1D(
        new_flux.flux, spectrum.spectral_axis, uncertainty=uncertainty
    )
    lines = find_lines_threshold(noise_spectrum, noise_factor=noise_factor)

    line_sign = {"emission": 1.0, "absorption": -1.0}
    line_spectrum = np.zeros(len(new_flux.spectral_axis))
    line_intervals = []

    for line in lines:
        amp = line_sign[line["line_type"]]
        center = line["line_center"].value

        gauss_model = models.Gaussian1D(amplitude=amp, mean=center)
        gauss_fit = fit_lines(new_flux, gauss_model, window=window)
        intensity = gauss_fit(new_flux.spectral_axis)
        interval = _make_window(center, 3 * gauss_fit.stddev.value)

        line_spectrum += intensity.value
        line_intervals.append(interval)

    line_spectrum = u.Quantity(line_spectrum)
    line_nd_spectrum = core.NirdustSpectrum(
        flux=line_spectrum, spectral_axis=spectrum.spectral_axis
    )
    line_intervals = u.Quantity(line_intervals, u.AA)

    return line_nd_spectrum, line_intervals
