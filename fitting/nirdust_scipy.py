from functools import partial
import nirdust as nd

import matplotlib.pyplot as plt

plt.ion()

import astropy.units as u

# from astropy.io import fits
from astropy.modeling import models
import numpy as np

from scipy.optimize import basinhopping, least_squares

from true_model import true_model

# ================================================================
seed = 42
# True Parameters
true_T = 900 * u.K
true_alpha = 18.0
true_beta = 1e7
true_gamma = 1e-4
snr = 100
true_theta = (true_T, true_alpha, np.log10(true_beta), np.log10(true_gamma))

spectrumT, externalT = true_model(
    true_T,
    true_alpha,
    np.log10(true_beta),
    np.log10(true_gamma),
    snr=snr,
    seed=seed,
    validate=True
)
plt.plot(spectrumT.spectral_axis.value, spectrumT.flux.value, "-")
# ================================================================
# Physical Conditions

# bounds
bounds_T = (100.0, 2000.0)
bounds_alpha = (0, 20)
bounds_log_beta = (6, 10)
bounds_log_gamma = (-10, 0)

# constraints
def alpha_vs_beta(theta, spectral_axis, target_flux, external_flux, noise):
    # we assume that alpha*ExternalSpectrum > beta*BlackBody, in mean values
    T, alpha, log_beta, log_gamma = theta
    beta = 10 ** log_beta
    gamma = 10 ** log_gamma

    prediction = nd.target_model(
        spectral_axis,
        external_flux,
        T,
        alpha,
        beta,
        gamma,
    )

    alpha_term = np.mean(alpha * external_flux)
    beta_term = np.mean(prediction - alpha_term - gamma)

    alpha_positivity = alpha_term - beta_term
    # if alpha_positivity > 0:
    #    print(f"alpha positivity {alpha_positivity>0} : {alpha} : {log_beta}")

    # Positive output is True
    return alpha_positivity


def gamma_vs_total_flux(
    theta, spectral_axis, target_flux, external_flux, noise
):
    # we assume that gamma can account for 5 percent or less of target flux
    T, alpha, log_beta, log_gamma = theta
    gamma = 10 ** log_gamma

    gamma_positivity = 0.05 * target_flux.min() - gamma
    # if gamma_positivity > 0:
    #    print(f"gamma positivity {gamma_positivity>0} : {gamma}")

    # Positive output is True
    return gamma_positivity


# print callback function
def print_fun(x, f, accepted):
    # if f<0:
    print(f"at minimum {f} : {x[0]:.1f}, {x[1]:.2f}, {x[2]:.2f}, {x[3]:.4f}")


# Initial values
x0 = [1000.0, 8.0, 9.0, -5]

args = (
    spectrumT.spectral_axis,
    spectrumT.flux.value,
    externalT.flux.value,
    spectrumT.noise,
)

bounds = (bounds_T, bounds_alpha, bounds_log_beta, bounds_log_gamma)
constraints = (
    {"type": "ineq", "fun": alpha_vs_beta, "args": args},
    {"type": "ineq", "fun": gamma_vs_total_flux, "args": args},
)

minimizer_kwargs = {
    "method": "SLSQP",
    "args": args,
    "bounds": bounds,
    "constraints": constraints,
    "options": {"maxiter": 1000, "ftol": 1e-8},
    "jac": "3-point",
}

bh_res = basinhopping(
    nd.model4scipy,
    x0=x0,
    niter=100,
    T=100,
    stepsize=1,
    seed=seed,
    callback=print_fun,
    niter_success=500,
    minimizer_kwargs=minimizer_kwargs,
)

# =================================================================
# Reconstruct model
vfit = bh_res.x
fT, fa, fb, fg = vfit  # [0], vfit[1], vfit[2], vfit[3]

prediction_fit = nd.target_model(
    spectrumT.spectral_axis, externalT.flux.value, fT, fa, 10 ** fb, 10 ** fg
)
prediction_true = nd.target_model(
    spectrumT.spectral_axis,
    externalT.flux.value,
    true_T,
    true_alpha,
    true_beta,
    true_gamma,
)

plt.plot(
    spectrumT.spectral_axis.value,
    spectrumT.flux.value,
    "-",
    label="Noisy data",
)
plt.plot(
    spectrumT.spectral_axis.value, prediction_true, "k-", label="True data"
)
plt.plot(spectrumT.spectral_axis.value, prediction_fit, "r--", label="Fit")
plt.legend()

ll_fit = nd.model4scipy(
    vfit,
    spectrumT.spectral_axis,
    spectrumT.flux.value,
    externalT.flux.value,
    spectrumT.noise,
)
ll_true = nd.model4scipy(
    true_theta,
    spectrumT.spectral_axis,
    spectrumT.flux.value,
    externalT.flux.value,
    spectrumT.noise,
)
print(f" True: {ll_true:.3f} \n Fit: {ll_fit:.3f}")

# ============================================================================
# COMPARISON PLOT
# ============================================================================
# True Parameters
true_T = 750 * u.K
true_alpha = 15.0
true_beta = 2e8
true_gamma = 5e-4
snr = 100
true_theta = (true_T, true_alpha, np.log10(true_beta), np.log10(true_gamma))



noises = [100, 200, 300] # np.arange(50, 1050, 50)
seeds = np.arange(0, 10)

results = []
for noise in noises:
    spectrumT, externalT = true_model(
        true_T,
        true_alpha,
        np.log10(true_beta),
        np.log10(true_gamma),
        snr=noise,
        seed=42,
    )

    partial_results = []
    for seed in seeds:
        print(f"{noise=}, {seed=}")
        args = (
            spectrumT.spectral_axis,
            spectrumT.flux.value,
            externalT.flux.value,
            spectrumT.noise,
        )
        bounds = (bounds_T, bounds_alpha, bounds_log_beta, bounds_log_gamma)
        constraints = (
            {"type": "ineq", "fun": alpha_vs_beta, "args": args},
            {"type": "ineq", "fun": gamma_vs_total_flux, "args": args},
        )
        minimizer_kwargs = {
            # "method": "trust-constr",
            "method": "SLSQP",
            "args": args,
            "bounds": bounds,
            "constraints": constraints,
            "options": {"maxiter": 1000},
        }

        res = basinhopping(
            nd.model4scipy,
            x0=x0,
            minimizer_kwargs=minimizer_kwargs,
            niter=100,
            T=100,
            stepsize=1,
            seed=seed,
            # callback=print_fun,
            niter_success=500,
        )
        if res.success:
            partial_results.append(res.x)
        else:
            partial_results.append(np.zeros(4))
    
    results.append(partial_results)


# ============================================================================
# TEST WITH REAL DATA
# ============================================================================

# # NGC4945
sp1 = nd.read_fits("tests/data/cont03.fits", z=0.00188)
sp2 = nd.read_fits("tests/data/external_spectrum_200pc_N4945.fits", z=0.00188)

target = sp1.cut_edges(20000, 22900)
external = sp2.cut_edges(20000, 22900)


# Initial values
x0 = [1000.0, 8.0, 9.0, -5]

args = (
    target.spectral_axis,
    target.flux.value,
    external.flux.value,
    target.noise,
)

bounds = (bounds_T, bounds_alpha, bounds_log_beta, bounds_log_gamma)
constraints = (
    {"type": "ineq", "fun": alpha_vs_beta, "args": args},
    {"type": "ineq", "fun": gamma_vs_total_flux, "args": args},
)

minimizer_kwargs = {
    # "method": "trust-constr",
    "method": "SLSQP",
    "args": args,
    "bounds": bounds,
    "constraints": constraints,
    "options": {"maxiter": 1000},
}

bh_res = basinhopping(
    nd.model4scipy,
    x0=x0,
    minimizer_kwargs=minimizer_kwargs,
    niter=100,
    T=100,
    stepsize=1,
    seed=seed,
    callback=print_fun,
    niter_success=500,
)


# Reconstruct model
vfit = bh_res.x
fT, fa, fb, fg = vfit  # [0], vfit[1], vfit[2], vfit[3]

prediction_fit = nd.target_model(
    target.spectral_axis, external.flux.value, fT, fa, 10 ** fb, 10 ** fg
)

plt.plot(
    target.spectral_axis.value, target.flux.value, "-", label="Noisy data"
)
plt.plot(target.spectral_axis.value, prediction_fit, "r--", label="Fit")
plt.legend()
