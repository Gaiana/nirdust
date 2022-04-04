import nirdust as nd
import matplotlib.pyplot as plt

plt.ion()

import astropy.units as u
from astropy.io import fits
from astropy.modeling import models
import numpy as np


# ==========================================
# TEST NOISE


def gaussian_noise(signal, snr, seed):
    rng = np.random.default_rng(seed)
    if snr is None:
        return np.zeros_like(signal)

    sigma = np.mean(signal) / snr
    noise = rng.normal(0, sigma, len(signal))
    print("mean qq ", np.mean(signal), sigma)
    return noise


def make_nuclear(wave, bb):
    # Linear model
    def tp_line(x, x1, x2, y1, y2):
        return (y2 - y1) / (x2 - x1) * (x - x1) + y1

    # nuclear = np.zeros_like(wave)
    delta_bb = bb[-1] - bb[0]
    y1_line = bb[0] + 6 / 5 * delta_bb
    y2_line = bb[0] + 2 / 5 * delta_bb

    # or just linear
    nuclear = tp_line(wave, wave[0], wave[-1], y1_line, y2_line)
    return nuclear


vega = fits.open("vega_1.fits")[0].data
spectral_axis = np.linspace(20100.0, 23000, len(vega)) * u.AA


def true_model(
    T, alpha, log_beta, log_gamma, snr=None, seed=None, validate=True
):

    # True Parameters
    true_T = u.Quantity(T, u.K)
    true_alpha = alpha
    true_beta = 10 ** log_beta
    true_gamma = 10 ** log_gamma

    # BlackBody model
    model = models.BlackBody(true_T)
    bb = true_beta * model(spectral_axis).value

    wave = spectral_axis.value
    # nuclear = vega * 1e14
    nuclear = make_nuclear(wave, bb)

    # Total model
    xternal = nuclear / true_alpha
    flux = nuclear + bb + true_gamma

    if validate:
        if true_gamma > 0.05 * flux.min():
            raise ValueError(f"Gamma: {true_gamma} > {0.05 * flux.min()}")

        if nuclear.mean() < bb.mean():
            raise ValueError("Nuclear < BB")

    noisy_model = flux + gaussian_noise(flux, snr, seed)
    noisy_external = xternal + gaussian_noise(xternal, snr, seed)

    spectrumT = nd.NirdustSpectrum(
        flux=noisy_model * u.adu, spectral_axis=spectral_axis
    )
    externalT = nd.NirdustSpectrum(
        flux=noisy_external * u.adu, spectral_axis=spectral_axis
    )
    return spectrumT, externalT
