import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import create_params, fit_report, minimize
from monteCarloDataHandler import monteCarloDataHandler
from scipy import optimize
from scipy.optimize import curve_fit, minimize
from xMaxDataHandler import xMaxDataHandler


monteCarloData = monteCarloDataHandler().getMonteCarloData()
xMaxHandler = xMaxDataHandler()
xMaxData = xMaxHandler.getXmaxData()

# get momentums with erros from xMax data
(
    meandata,
    meandata_err,
    vardata,
    vardata_err,
    skewdata,
    skewdata_err,
    kurtdata,
    kurtdata_err,
) = xMaxHandler.moments_with_errors(
    xMaxData[0][0][:48], xMaxData[0][1][:48], xMaxData[0][2][:48]
)

arr1 = monteCarloData[0][1][:48]
arr2 = monteCarloData[1][1][:48]
arr3 = monteCarloData[2][1][:48]
arr4 = monteCarloData[3][1][:48]
arr5 = monteCarloData[4][:48]

data = np.array(xMaxData[0][1][:48], dtype=float)
data_err = np.array(xMaxData[0][2][:48], dtype=float)
params = [0.25, 0.25, 0.25, 0.25]

# --- Súlyozási tényező a momentumoknak ---
W_MOMENTS = 50.0  # Ezt az értéket tudod növelni (pl. 100-ra), ha a skew/kurt még mindig távol van

def model(pars):
    a = pars[0]
    b = pars[1]
    c = pars[2]
    d = pars[3]
    return (
        a * arr1["Frac"][:48]
        + b * arr2["Frac"][:48]
        + c * arr3["Frac"][:48]
        + d * arr4["Frac"][:48]
    )

def residual(pars):
    this_model = model(pars)
    mcmean, mcmu2, mcskew, mc_excesskurt = xMaxHandler.moments_from_prob(
        xMaxData[0][0][:48], this_model
    )

    # 1. Alak illeszkedése (Chi2)
    chi = 0.0
    for i in range(len(this_model)):
        err = data_err[i] if data_err[i] > 0 else 1.0
        chi += ((data[i] - this_model[i]) / err) ** 2

    # 2. Momentum feltételek súlyozva
    chi += W_MOMENTS * ((meandata - mcmean) / meandata_err) ** 2
    chi += W_MOMENTS * ((vardata - mcmu2) / vardata_err) ** 2
    chi += W_MOMENTS * ((skewdata - mcskew) / skewdata_err) ** 2
    chi += W_MOMENTS * ((kurtdata - mc_excesskurt) / kurtdata_err) ** 2

    return chi

# Boundaries - Ha c továbbra is 0 marad, írd át a harmadik tagot pl. (0.01, 1)-re!
bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]

constr_unit = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}
constraints = [constr_unit]

out = optimize.minimize(
    residual, params, method="SLSQP", bounds=bounds, constraints=constraints
)

print("Success:", out.success)
print("Message:", out.message)

print("a =", out.x[0])
print("b =", out.x[1])
print("c =", out.x[2])
print("d =", out.x[3])
print("chi2 =", out.fun)

a_fit = out.x[0]
b_fit = out.x[1]
c_fit = out.x[2]
d_fit = out.x[3]

model_fit = model(out.x)
mcmeantest, mcmu2test, mcskewtest, mc_excesskurttest = (
    xMaxHandler.moments_from_prob(xMaxData[0][0][:48], model_fit)
)

# --- Paraméterhibák becslése Bootstrap módszerrel ---
N_BOOTSTRAP = 200
boot_fits = []

for _ in range(N_BOOTSTRAP):
    data_sampled = np.random.normal(data, data_err)

    def residual_boot(pars):
        m = model(pars)
        chi_b = np.sum(
            ((data_sampled - m) / np.where(data_err > 0, data_err, 1.0)) ** 2
        )
        m_mean, m_mu2, m_skew, m_kurt = xMaxHandler.moments_from_prob(
            xMaxData[0][0][:48], m
        )
        chi_b += W_MOMENTS * ((meandata - m_mean) / meandata_err) ** 2
        chi_b += W_MOMENTS * ((vardata - m_mu2) / vardata_err) ** 2
        chi_b += W_MOMENTS * ((skewdata - m_skew) / skewdata_err) ** 2
        chi_b += W_MOMENTS * ((kurtdata - m_kurt) / kurtdata_err) ** 2
        return chi_b

    res = optimize.minimize(
        residual_boot,
        params,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if res.success:
        boot_fits.append(res.x)

boot_fits = np.array(boot_fits)
pcov = np.cov(boot_fits, rowvar=False)
param_errors = np.std(boot_fits, axis=0)

print("\nKovarianciamátrix (pcov):")
print(pcov)
print(
    f"Paraméter hibák (a_err, b_err, c_err, d_err): {param_errors[0]:.4f}, {param_errors[1]:.4f}, {param_errors[2]:.4f}, {param_errors[3]:.4f}"
)

print(f"Model skew: ", mcskewtest, "Auger skew: ", skewdata, "Auger skewerr: ", skewdata_err)
print(f"Model kurt: ", mc_excesskurttest, "Auger kurt: ", kurtdata, "Auger kurt: ", kurtdata_err)

# --- Ábrázolás változatlanul ---
plt.errorbar(
    monteCarloData[4][:48],
    xMaxData[0][1][:48],
    yerr=xMaxData[0][2][:48],
    fmt="o",
    markersize=1,
    capsize=1,
    elinewidth=1,
    color="black",
    zorder=3,
    label="Auger Xmax data",
)
plt.plot(
    monteCarloData[4][:48],
    model_fit,
    "-",
    markersize=2,
    color="skyblue",
    label="MC composition",
)
plt.xlabel("Xmax")
plt.title("MC composition on Xmax data")
plt.legend()
plt.show()