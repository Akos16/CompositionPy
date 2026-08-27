from monteCarloDataHandler import monteCarloDataHandler
from xMaxDataHandler import xMaxDataHandler
from compostionHandler import compositionHandler
from lmfit import create_params, fit_report, minimize
from scipy.optimize import minimize                                             ## For the minimalization
import scipy.optimize as optimization                                           ## For the optimization
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import optimize

monteCarloData = monteCarloDataHandler().getMonteCarloData()
xMaxHandler = xMaxDataHandler()
xMaxData = xMaxHandler.getXmaxData()

arrAugerComp = [[0.40, 0.03, 0.41, 0.16],[0.26, 0.21, 0.47, 0.06],[0.14, 0.00, 0.84, 0.02]]

lgE_values = ["18", "18.5", "19"]
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
for idx in range(3):
    lgE = idx + 1
    xMax = idx
    ax = axes[idx]
#get momentums with erros from xMax data
    meandata, meandata_err, vardata, vardata_err, skewdata, skewdata_err, kurtdata, kurtdata_err = xMaxHandler.moments_with_errors(xMaxData[xMax][0][:48], xMaxData[xMax][1][:48], xMaxData[xMax][2][:48])

    arr1 = monteCarloData[0][lgE][:48]
    arr2 = monteCarloData[1][lgE][:48]
    arr3 = monteCarloData[2][lgE][:48]
    arr4 = monteCarloData[3][lgE][:48]
    arr5 = monteCarloData[4][:48]

    data = np.array(xMaxData[xMax][1][:48], dtype=float)
    params = [0.10,0.25,0.25,0.40]
    
    def model(pars):
        a = pars[0]
        b = pars[1]
        c = pars[2]
        d = pars[3]
        return a * arr1["Frac"] + b * arr2["Frac"] + c * arr3["Frac"] + d * arr4["Frac"]
    
    def residual(pars):
        this_model = model(pars)
        chi = 0.0
        for i in range(len(this_model)):
            if data[i] != 0:
                chi += ((data[i] - this_model[i])/np.sqrt(data[i])) ** 2
        return chi

    def mean(pars):
        this_model = model(pars)
        mcmean, mcmu2, mcskew, mc_excesskurt = xMaxHandler.moments_from_prob(xMaxData[xMax][0][:48], this_model)
        return ((meandata - mcmean) / meandata_err) ** 2
    def mu(pars):
        this_model = model(pars)
        mcmean, mcmu2, mcskew, mc_excesskurt = xMaxHandler.moments_from_prob(xMaxData[xMax][0][:48], this_model)
        return ((vardata - mcmu2) / vardata_err) ** 2

    def skew(pars):
        this_model = model(pars)
        mcmean, mcmu2, mcskew, mc_excesskurt = xMaxHandler.moments_from_prob(xMaxData[xMax][0][:48], this_model)
        return ((skewdata - mcskew) / skewdata_err) ** 2
    def kurt(pars):
        this_model = model(pars)
        mcmean, mcmu2, mcskew, mc_excesskurt = xMaxHandler.moments_from_prob(xMaxData[xMax][0][:48], this_model)
        return ((kurtdata - mc_excesskurt) / kurtdata_err) ** 2

    #mc, xmax skew és kurt különbsége minél kisebb legyen
    # Initial parameter values, a, b, c, d

    # Optimization
    bounds = [ #https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.Bounds.html#scipy.optimize.Bounds
        (0, 1),  # a
        (0, 1),  # b
        (0, 1),  # c
        (0, 1)   # d
    ]
    #?
    constr_mean = {"type":"eq", "fun": mean}
    constr_mu = {"type":"eq", "fun": mu}
    constr_skew = { "type": "eq", "fun": skew }
    constr_kurt = { "type": "eq", "fun": kurt }
    constr_unit = {"type":"eq", "fun": lambda x: np.sum(x) - 1}
    constraints = [ constr_unit]
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


    a_fit = out.x[0]
    b_fit = out.x[1]
    c_fit = out.x[2]
    d_fit = out.x[3]

    model_fit = (a_fit*arr1["Frac"][:48] + b_fit*arr2["Frac"][:48] + c_fit*arr3["Frac"][:48] + d_fit*arr4["Frac"][:48])
    mcmeantest, mcmu2test, mcskewtest, mc_excesskurttest = xMaxHandler.moments_from_prob(xMaxData[xMax][0][:48], model_fit)
    print(f"Model skew: ", mcskewtest, "Auger skew: ", skewdata, "Auger skewerr: ", skewdata_err)
    print(f"Model kurt: ", mc_excesskurttest, "Auger kurt: ", kurtdata, "Auger kurt: ", kurtdata_err)
    #ax.errorbar(x, y, yerr=yerr, fmt='o', markersize=1, capsize=1, elinewidth=1, color='black', zorder=3, label='Mért adat')
    #plot xMax, lgE 18
    l1 = ax.errorbar(monteCarloData[4][:48], xMaxData[xMax][1][:48], yerr=xMaxData[xMax][2][:48], fmt='o', markersize=1, capsize=1, elinewidth=1, color='black', zorder=3, label='Auger Xmax data')
    l2,= ax.plot(monteCarloData[4][:48], model_fit, '-', markersize=2, color='skyblue', label=f'MC composition')
    #H, He, N, Fe
    ax.text(0.65, 0.60, f'$f_H$={out.x[0]:.2f}\n$f_{{He}}$={out.x[1]:.2f}\n$f_N$={out.x[2]:.2f}\n$f_{{Fe}}$={out.x[3]:.2f}', transform=ax.transAxes, fontsize=14)
    ax.text(0.05, 0.55, f'$f_H^{{Auger}}$={arrAugerComp[xMax][0]:.2f}\n$f_{{He}}^{{Auger}}$={arrAugerComp[xMax][1]:.2f}\n$f_N^{{Auger}}$={arrAugerComp[xMax][2]:.2f}\n$f_{{Fe}}^{{Auger}}$={arrAugerComp[xMax][3]:.2f}', transform=ax.transAxes, fontsize=14)
    ax.set_xlabel("Xmax")
    ax.set_title(f"lgE = {lgE_values[idx]}")
    ax.legend(loc='upper right')
axes[0].set_ylabel("Fraction")
plt.suptitle("MC composition on Xmax data", fontsize=16)
plt.tight_layout()
plt.show()

