#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   NirDust Project (https://github.com/Gaiana/nirdust)
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

from astropy import units as u
from astropy.modeling import fitting, models

import numpy as np

from specutils.fitting import find_lines_threshold
from specutils.fitting import fit_generic_continuum
from specutils.fitting import fit_lines
from specutils.manipulation import FluxConservingResampler
from specutils.manipulation import noise_region_uncertainty
from specutils.spectra import SpectralRegion

from . import core


# ==============================================================================
# RESAMPLE SPECTRA TO MATCH SPECTRAL RESOLUTIONS
# ==============================================================================


def _rescale(sp, reference_sp):
    """Resample a given spectrum to a reference spectrum.

    The first spectrum will be resampled to have the same spectral_axis as
    the reference spectrum. The resampling algorithm is the specutils method
    FluxConservingResampler.

    Notes
    -----
    nan values may occur at the edges where the resampler is forced
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
    """Clean nan values and apply the same mask to both spectrums."""
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

    Spectrum_resampling uses the spectral_axis of the lower resolution
    spectrum to resample the higher resolution one. To do so this function
    uses the FluxConservingResampler() class of 'Specutils'. The order of the
    input spectra is arbitrary and the order in the output is the same as in
    the input. Only the higher resolution spectrum will be modified, the lower
    resolution spectrum will be unaltered. It is recommended to run
    spectrum_resampling after 'cut_edges'.

    Parameters
    ----------
    first_sp: NirdustSpectrum object

    second_sp: NirdustSpectrum object

    scaling: string
        If 'downscale' the higher resolution spectrum will be resampled to
        match the lower resolution spectrum. If 'upscale' the lower resolution
        spectrum will be resampled to match the higher resolution spectrum.

    clean: bool
        Flag to indicate if the spectrums have to be cleaned by nan values
        after the rescaling procedure. nan values occur at the edges of the
        resampled spectrum when it is forced to extrapolate beyond the
        spectral range of the reference spectrum.

    Return
    ------
    out: NirdustSpectrum, NirdustSpectrum

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

    else:
        # they have the same dispersion, is that equivalent
        # to equal spectral_axis?
        pass

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
    low_lim_ns=20650,
    upper_lim_ns=21000,
    noise_factor=3,
    window=50,
):
    """Construct the line spectrum.

    Uses various Specutils features to fit the continuum of the spectrum,
    subtract it and find the emission and absorption lines in the spectrum.
    Then fits all the lines with gaussian models to construct the line
    spectrum.

    Parameters
    ----------
    spectrum: NirdustSpectrum object
        A spectrum stored in a NirdustSpectrum class object.

    low_lim_ns: float
        Lower limit of the spectral region defined to measure the
        noise level. Default is 20650 (wavelenght in Angstroms).

    upper_lim_ns: float
        Lower limit of the spectral region defined to measure the
        noise level. Default is 21000 (wavelenght in Angstroms).

    noise_factor: float
        Same parameter as in find_lines_threshold from Specutils.
        Factor multiplied by the spectrum’s``uncertainty``, used for
        thresholding. Default is 3.

    window: float
        Same parameter as in fit_lines from specutils.fitting. Regions of
        the spectrum to use in the fitting. If None, then the whole
        spectrum will be used in the fitting. Window is used in the
        Gaussian fitting of the spectral lines. Default is 50 (Angstroms).

    Return
    ------
    out: flux axis, list, list
        Returns in the first position a flux axis of the same lenght as the
        original spectrum containing the fitted lines. In the second position,
        returns the intervals where those lines were finded determined by
        3-sigma values around the center of the line. In the third position
        returns an array with the quality of the fitting for each line.

    """
    # values in correct units
    low_lim_ns = u.Quantity(low_lim_ns, u.AA)
    upper_lim_ns = u.Quantity(upper_lim_ns, u.AA)
    window = u.Quantity(window, u.AA)

    # By defaults this fits a Chebyshev of order 3 to the flux
    model = fit_generic_continuum(
        spectrum.spec1d_, fitter=fitting.LinearLSQFitter()
    )
    continuum = model(spectrum.spectral_axis)
    new_flux = spectrum.spec1d_ - continuum

    noise_region_def = SpectralRegion(low_lim_ns, upper_lim_ns)
    noise_reg_spectrum = noise_region_uncertainty(new_flux, noise_region_def)
    lines = find_lines_threshold(noise_reg_spectrum, noise_factor=noise_factor)

    line_sign = {"emission": 1.0, "absorption": -1.0}
    line_spectrum = np.zeros(len(new_flux.spectral_axis))
    line_intervals = []

    for line in lines:
        amp = line_sign[line["line_type"]]
        center = line["line_center"].value

        gauss_model = models.Gaussian1D(amplitude=amp, mean=center)
        gauss_fit = fit_lines(new_flux, gauss_model, window=window)
        intensity = gauss_fit(new_flux.spectral_axis)
        interval = _make_window(center, 3 * gauss_fit.stddev)

        line_spectrum += intensity.value
        line_intervals.append(interval)

    line_spectrum = u.Quantity(line_spectrum)
    line_intervals = u.Quantity(line_intervals, u.AA)

    line_fitting_quality = 0.0
    return line_spectrum, line_intervals, line_fitting_quality


# ==============================================================================
# ISOLATE THE DUST COMPONENT
# ==============================================================================


def dust_component(nuclear_spectrum, external_spectrum):
    """Isolate the dust component via stellar population substraction.

    The spectral continuum of Type 2 Seyfert galaxies in the K band
    (19.-2.5 $mu$m) is composed by the stellar population component and the
    hot dust component. The first one is the sum of the Planck functions of
    all the stars in the host galaxy and can be represented by a spectrum
    extracted at a prudential distance from the nucleus, where the emission
    is expected to be dominated by the stellar population. In sp_correction
    this is introduced in the parameter "external spectrum". The stellar
    population dominated spectrum must be substracted from the nuclear
    spectrum in order to obtain the hot dust component in the nuclear
    spectrum. The excess obtained from the substraction is expected to have
    blackbody-shape.

    The operations applied to prepare the nuclear spectrum for fitting are:

    1) normalization using the mean value of the flux for both spectra
    2) substraction of the external spectrum flux from the nuclear spectrum
       flux.

    Parameters
    ----------
    nuclear_spectrum: NirdustSpectrum object
        Instance of NirdustSpectrum containing the nuclear spectrum.

    external_spectrum: NirdustSpectrum object
        Instance of NirdustSpectrum containing the external spectrum.

    Return
    ------
    out: NirsdustSpectrum object
        Returns a new instance of the class NirdustSpectrum containing the
        nuclear spectrum ready for blackbody fitting.
    """
    normalized_nuc = nuclear_spectrum.normalize()
    normalized_ext = external_spectrum.normalize()

    flux_resta = (
        normalized_nuc.spec1d_.flux - normalized_ext.spec1d_.flux
    ) + 1

    new_spectral_axis = nuclear_spectrum.spec1d_.spectral_axis

    kwargs = core.public_members_asdict(normalized_nuc)
    kwargs.update(
        flux=flux_resta,
        spectral_axis=new_spectral_axis,
    )

    return core.NirdustSpectrum(**kwargs)