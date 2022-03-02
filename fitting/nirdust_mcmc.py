import nirdust as nd
import matplotlib.pyplot as plt

plt.ion()

import astropy.units as u
from astropy.io import fits
from astropy.modeling import models
import numpy as np

import emcee
import corner
from multiprocessing import Pool
from joblib import Parallel, delayed

from true_model import true_model

# ================================================================
seed = 42
# True Parameters
true_T = 750 #* u.K
true_alpha = 15.0
true_beta = 2e8
true_gamma = 5e-4

true_theta = (true_T, true_alpha, np.log10(true_beta), np.log10(true_gamma))

cov = np.zeros((4, 4))
np.fill_diagonal(cov, [10., 1, 10., 1])
cov /= 100.

# ================================================================
noise_level = [200]
# noise_level = [50, 100, 150, 200, 250]
# m_line = [0.8, 0.5, 0.2, 0.1, 0.05]

temperature = []
for snr in noise_level:
    print(snr)

    spectrumT, externalT = true_model(
        true_T,
        true_alpha,
        np.log10(true_beta),
        np.log10(true_gamma),
        snr=snr,
        seed=seed,
    )

    moves = [
        # (emcee.moves.DEMove(), 0.5),
        (emcee.moves.DESnookerMove(), 0.5),
        # (emcee.moves.DEMove(gamma0=3), 0.5),
        (emcee.moves.GaussianMove(cov), 0.5),
    ]
    # moves=[
    #     (emcee.moves.DESnookerMove(), 0.1),
    #     (emcee.moves.DEMove(), 0.9 * 0.9),
    #     (emcee.moves.DEMove(gamma0=1.0), 0.9 * 0.1),
    # ]
    with Pool() as pool:
        fitter = nd.fit_blackbody(
            spectrumT,
            externalT,
            initial_state=[1000.0, 10.0, 9.0, -5],
            steps=20000,
            nwalkers=9,
            seed=41,
            pool=pool,
            moves=moves,
        )

    temperature.append(fitter)
fitter.plot()


c = temperature[0].chain(discard=2500).reshape((-1, 4))
corner.corner(c, bins=30, labels=["T", "alpha", "beta", "gamma"])


# ============================================================================
# TEST WITH REAL DATA
# ============================================================================

# NGC4945
sp1 = nd.read_fits("tests/data/cont03.fits", z=0.00188)
sp2 = nd.read_fits("tests/data/external_spectrum_200pc_N4945.fits", z=0.00188)

target = sp1.cut_edges(20000, 22900)
external = sp2.cut_edges(20000, 22900)

with Pool() as pool:
    # moves=[
    #     (emcee.moves.DEMove(1e-4), 0.8),
    #     (emcee.moves.DESnookerMove(1.2), 0.2),
    # ]
    moves = [
        (emcee.moves.DESnookerMove(), 0.1),
        (emcee.moves.DEMove(), 0.9 * 0.9),
        (emcee.moves.DEMove(gamma0=1.0), 0.9 * 0.1),
    ]
    fitter = nd.fit_blackbody(
        target,
        external,
        initial_state=(1000.0, 1.0, 1e10, 3e3),
        steps=20000,
        nwalkers=15,
        seed=1,
        pool=pool,
        moves=moves,
    )


# NGCXXX
nuclear_raw = nd.read_table("Kcortenuclear.txt", z=0.0037)
external_raw = nd.read_table("fcsKcorte8-3.txt", z=0.0037)

nuclear_cut = nuclear_raw.cut_edges(20160, 23000)
external_cut = external_raw.cut_edges(20160, 23000)

mask_nuclear = nd.line_spectrum(nuclear_cut, 21360, 21695)[1]
mask_external = nd.line_spectrum(external_cut, 20691, 21115)[1]

masked_nuclear = nuclear_cut.mask_spectrum(mask_nuclear)
masked_external = external_cut.mask_spectrum(mask_external)

m_nuclear, m_external = nd.match_spectral_axes(masked_nuclear, masked_external)

wave = m_nuclear.spectral_axis

with Pool() as pool:
    fit = nd.fit_blackbody(m_nuclear, m_external, steps=8000, pool=pool)


################
# PROBANDO BASINHOPPING DE SCIPY
from scipy.optimize import basinhopping


def print_callback(x, f, accepted):
    print("at minima %.4f accepted %d" % (f, int(accepted)))


# True values: 800, 10, 1e9, 1
x0 = [600.0, 20, 10, 1.0]
args = (
    spectrumT.spectral_axis,
    spectrumT.flux.value,
    externalT.flux.value,
    spectrumT.noise,
)
minimizer_kwargs = {
    "method": "TNC",
    "args": args,
    "options": {"maxfun": 100, "eps": 1e-4},
}

result = basinhopping(
    nd.model4scipy,
    x0,
    minimizer_kwargs=minimizer_kwargs,
    niter=1000,
    T=10,
    seed=42,
)
print("=====================")
print(result)
print("=====================")
