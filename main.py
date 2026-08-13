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

#get momentums with erros from xMax data
mean, mean_err, var, var_err, skew,  skew_err, kurt, kurt_err = xMaxHandler.moments_with_errors(xMaxData[0][0][:48], xMaxData[0][1][:48], xMaxData[0][2][:48])

arr1 = monteCarloData[0][1][:48]
arr2 = monteCarloData[1][1][:48]
arr3 = monteCarloData[2][1][:48]
arr4 = monteCarloData[3][1][:48]

data = np.array(xMaxData[0][1][:48], dtype=float)
params = create_params(a=0.68,b=0.11,c=0.17,d=0.04)

def residual(pars):
    """Calculate chi-square between model and data."""

    a = pars[0]
    b = pars[1]
    c = pars[2]
    d = pars[3]

    model = (a * arr1["Frac"] + b * arr2["Frac"] + c * arr3["Frac"] + d * arr4["Frac"])

    chi = 0.0

    for i in range(len(model)):
        if data[i] != 0:
            chi += ((data[i] - model[i])/np.sqrt(data[i])) ** 2 

    return chi

#mc, xmax skew és kurt különbsége minél kisebb legyen
# Initial parameter values, a, b, c, d
params = np.array([0.4, 0.03, 0.41, 0.15])

# Optimization
out = optimize.minimize(residual, params, method="SLSQP")

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
#ax.errorbar(x, y, yerr=yerr, fmt='o', markersize=1, capsize=1, elinewidth=1, color='black', zorder=3, label='Mért adat')
#plot xMax, lgE 18
plt.errorbar(monteCarloData[4][:48], xMaxData[0][1][:48], yerr=xMaxData[0][2][:48], fmt='o', markersize=1, capsize=1, elinewidth=1, color='black', zorder=3, label='Auger Xmax data')
plt.plot(monteCarloData[4][:48], model_fit, '-', markersize=2, color='skyblue', label='MC composition')
plt.xlabel("Xmax")
plt.title("MC composition on Xmax data")
plt.legend()
plt.show()

