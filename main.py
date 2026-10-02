
from monteCarloDataHandler import monteCarloDataHandler
from xMaxDataHandler import xMaxDataHandler
from lmfit import create_params, fit_report, minimize
from scipy.optimize import minimize
import scipy.optimize as optimization
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import optimize


monteCarloData = monteCarloDataHandler().getMonteCarloData()
xMaxHandler = xMaxDataHandler()
xMaxData = xMaxHandler.getXmaxData()

arrAugerComp = [[0.40, 0.03, 0.41, 0.16],[0.26, 0.21, 0.47, 0.06],[0.14, 0.00, 0.84, 0.02]]

#Bootstrap beállítások
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 12345
rng = np.random.default_rng(BOOTSTRAP_SEED)

lgE_values = ["18", "18.5", "19"]
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)

for idx in range(3):

    lgE = idx + 1
    xMax = idx
    ax = axes[idx]

    #aata
    meandata, meandata_err, vardata, vardata_err, skewdata, skewdata_err, kurtdata, kurtdata_err = xMaxHandler.moments_with_errors(xMaxData[xMax][0][:48], xMaxData[xMax][1][:48], xMaxData[xMax][2][:48])

    
    xMaxAuger = np.asarray(xMaxData[xMax][0][:48], dtype=float)

    arr1 = monteCarloData[0][lgE][:48]
    arr2 = monteCarloData[1][lgE][:48]
    arr3 = monteCarloData[2][lgE][:48]
    arr4 = monteCarloData[3][lgE][:48]
    arr5 = monteCarloData[4][:48]

    #Xmax Auger Counts
    augerData = np.asarray(xMaxData[xMax][1][:48], dtype=float)
    #Xmax Auger Counts Sqrt
    augerData_Err = np.asarray(xMaxData[xMax][2][:48], dtype=float)

    #kezdeti paraméterek a comphoz
    params = [0.10, 0.25, 0.25, 0.40]

    #Model
    #Monte-Carlo templatek
    mc1 = np.asarray(arr1["Frac"], dtype=float)
    mc2 = np.asarray(arr2["Frac"], dtype=float)
    mc3 = np.asarray(arr3["Frac"], dtype=float)
    mc4 = np.asarray(arr4["Frac"], dtype=float)

    def model(pars):
        a = pars[0]
        b = pars[1]
        c = pars[2]
        d = pars[3]
        return (
            a * mc1 + b * mc2 + c * mc3 + d * mc4)
    #Chi^2
    def residual(pars):
        this_model = model(pars)
        mask = augerData != 0
        return np.sum(
            ((augerData[mask] - this_model[mask]) / np.sqrt(augerData[mask])) ** 2)

    #mean, mu, skew, kurt
    def mean(pars):
        this_model = model(pars)
        mcmean, mcmu2, mcskew, mc_excesskurt = xMaxHandler.moments_from_prob(xMaxAuger,this_model)
        return ((meandata - mcmean) / meandata_err) ** 2

    def mu(pars):
        this_model = model(pars)
        mcmean, mcmu2, mcskew, mc_excesskurt = xMaxHandler.moments_from_prob(xMaxAuger,this_model)
        return ((vardata - mcmu2) / vardata_err) ** 2


    def skew(pars):
        this_model = model(pars)
        mcmean, mcmu2, mcskew, mc_excesskurt = xMaxHandler.moments_from_prob(xMaxAuger,this_model)
        return ((skewdata - mcskew) / skewdata_err) ** 2

    def kurt(pars):
        this_model = model(pars)
        mcmean, mcmu2, mcskew, mc_excesskurt = xMaxHandler.moments_from_prob(xMaxAuger,this_model)
        return ((kurtdata - mc_excesskurt) / kurtdata_err) ** 2

    #csak 0 és 1 között lehet a param
    bounds = [(0, 1),(0, 1),(0, 1),(0, 1)]

    constr_mean = {"type": "eq","fun": mean}
    constr_mu = {"type": "eq","fun": mu}
    constr_skew = {"type": "eq","fun": skew}
    constr_kurt = {"type": "eq","fun": kurt}

    #mindig 0 legyen
    constr_unit = {"type": "eq","fun": lambda x: np.sum(x) - 1}

    #megszorítások
    constraints = [constr_mean,constr_unit]

    #első fit
    out = optimize.minimize(
        residual,
        params,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    print("Success:", out.success)
    print("Message:", out.message)

    print("a =", out.x[0])
    print("b =", out.x[1])
    print("c =", out.x[2])
    print("d =", out.x[3])
    print("chi2 =", out.fun)

    #fittelt fractions
    a_fit = out.x[0]
    b_fit = out.x[1]
    c_fit = out.x[2]
    d_fit = out.x[3]

    model_fit = (a_fit * mc1 + b_fit * mc2 + c_fit * mc3 + d_fit * mc4)


    #moments az adatokból és a fitből
    mcmeantest, mcmu2test, mcskewtest, mc_excesskurttest = xMaxHandler.moments_from_prob(xMaxAuger,model_fit)
    model_sigma = np.sqrt(mcmu2test)
    
    print(f"Model mean: {mcmeantest}  Auger mean: {meandata}  Auger meanerr: {meandata_err}")
    print(f"Model variance: {mcmu2test}  Auger variance: {vardata}  Auger varerr: {vardata_err}")
    print(f"Model sigma: {model_sigma}")
    print(f"Model skew: {mcskewtest}  Auger skew: {skewdata}  Auger skewerr: {skewdata_err}")
    print(f"Model kurt: {mc_excesskurttest}  Auger kurt: {kurtdata}  Auger kurterr: {kurtdata_err}")

    #bootstrap hiba becslés

    bootstrap_fractions = []

    bootstrap_means = []
    bootstrap_vars = []
    bootstrap_sigmas = []
    bootstrap_skews = []
    bootstrap_kurts = []

    # Az eredeti fit legyen minden toy kezdőpontja
    bootstrap_start = out.x.copy()

    for iboot in range(N_BOOTSTRAP):
        #toy data
        toy_data = (augerData + rng.normal(0.0, augerData_Err))
        toy_data = np.clip(toy_data, 0.0, None)
        #ha üres akkor következő
        if np.sum(toy_data) <= 0:
            continue

        #chi^2 toy
        def toy_residual(pars):
            this_model = (pars[0] * mc1 + pars[1] * mc2 + pars[2] * mc3 + pars[3] * mc4)
            mask = toy_data != 0
            return np.sum(((toy_data[mask] - this_model[mask]) /np.sqrt(toy_data[mask])) ** 2)
        
        #toy fit
        toy_out = optimize.minimize(toy_residual,bootstrap_start,method="SLSQP",bounds=bounds,constraints=constraints,options={"maxiter": 100,"ftol": 1e-6})

        if not toy_out.success:
            continue
        toy_pars = toy_out.x
        #toy model
        toy_model = (toy_pars[0] * mc1 + toy_pars[1] * mc2 + toy_pars[2] * mc3 + toy_pars[3] * mc4)

        #toy mean, var, skew, kurt
        toy_mean, toy_var, toy_skew, toy_kurt = xMaxHandler.moments_from_prob(xMaxAuger, toy_model)
        toy_sigma = np.sqrt(toy_var)

        #appends
        bootstrap_fractions.append(toy_pars)
        bootstrap_means.append(toy_mean)
        bootstrap_vars.append(toy_var)
        bootstrap_sigmas.append(toy_sigma)
        bootstrap_skews.append(toy_skew)
        bootstrap_kurts.append(toy_kurt)
    #hibák
    bootstrap_fractions = np.asarray(bootstrap_fractions)
    bootstrap_means = np.asarray(bootstrap_means)
    bootstrap_vars = np.asarray(bootstrap_vars)
    bootstrap_sigmas = np.asarray(bootstrap_sigmas)
    bootstrap_skews = np.asarray(bootstrap_skews)
    bootstrap_kurts = np.asarray(bootstrap_kurts)

    #fraction hibák
    fraction_errors = np.std(bootstrap_fractions,axis=0,ddof=1)
    f_H_err = fraction_errors[0]
    f_He_err = fraction_errors[1]
    f_N_err = fraction_errors[2]
    f_Fe_err = fraction_errors[3]
    
    #momentumok hibái
    mean = mcmeantest
    mean_err = np.std(bootstrap_means,ddof=1)
    var = mcmu2test
    var_err = np.std(bootstrap_vars,ddof=1)
    sigma = model_sigma
    sigma_err = np.std(bootstrap_sigmas,ddof=1)
    skew = mcskewtest
    skew_err = np.std(bootstrap_skews,ddof=1)
    kurt = mc_excesskurttest
    kurt_err = np.std(bootstrap_kurts,ddof=1)


    print("\nChi^2")
    print(f"Mean  = {mean:.5f} +/- {mean_err:.5f}")
    print(f"Var   = {var:.5f} +/- {var_err:.5f}")
    print(f"Sigma = {sigma:.5f} +/- {sigma_err:.5f}")
    print(f"Skew  = {skew:.5f} +/- {skew_err:.5f}")
    print(f"Kurt  = {kurt:.5f} +/- {kurt_err:.5f}")
    print(f"f_H   = {a_fit:.5f} +/- {f_H_err:.5f}")
    print(f"f_He  = {b_fit:.5f} +/- {f_He_err:.5f}")
    print(f"f_N   = {c_fit:.5f} +/- {f_N_err:.5f}")
    print(f"f_Fe  = {d_fit:.5f} +/- {f_Fe_err:.5f}")
    print(f"Successful bootstrap fits: {len(bootstrap_fractions)}/{N_BOOTSTRAP}")

    #plot
    l1 = ax.errorbar(arr5, xMaxData[xMax][1][:48], yerr=xMaxData[xMax][2][:48], fmt='o', markersize=1, capsize=1, elinewidth=1, color='black', zorder=3, label='Auger Xmax data')
    l2, = ax.plot(arr5, model_fit, '-', markersize=2, color='skyblue', label='MC composition')

    ax.text(0.65, 0.60, f'$f_H$={a_fit:.2f}$\\pm${f_H_err:.2f}\n' f'$f_{{He}}$={b_fit:.2f}$\\pm${f_He_err:.2f}\n'f'$f_N$={c_fit:.2f}$\\pm${f_N_err:.2f}\n'f'$f_{{Fe}}$={d_fit:.2f}$\\pm${f_Fe_err:.2f}',transform=ax.transAxes,fontsize=14)

    #auger frac
    ax.text(0.05,0.55,f'$f_H^{{Auger}}$={arrAugerComp[xMax][0]:.2f}\n'f'$f_{{He}}^{{Auger}}$={arrAugerComp[xMax][1]:.2f}\n'f'$f_N^{{Auger}}$={arrAugerComp[xMax][2]:.2f}\n'f'$f_{{Fe}}^{{Auger}}$={arrAugerComp[xMax][3]:.2f}',transform=ax.transAxes,fontsize=14)
    ax.set_xlabel("Xmax")
    ax.set_title(f"lgE = {lgE_values[idx]}")
    ax.legend(loc='upper right')

axes[0].set_ylabel("Fraction")

plt.suptitle("MC composition on Xmax data",fontsize=16)
plt.tight_layout()
plt.show()

