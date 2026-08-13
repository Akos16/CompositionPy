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
xMaxData = xMaxDataHandler().getXmaxData()

def moments_from_prob(bin_centers, hist_counts):
    bin_centers = np.asarray(bin_centers, dtype=float)
    hist_counts = np.asarray(hist_counts, dtype=float)
    P = hist_counts / hist_counts.sum()
    mean = np.sum(bin_centers * P)
    mu2 = np.sum(((bin_centers - mean) ** 2) * P)
    skew = np.sum(((bin_centers - mean) ** 3) * P)
    kurt = np.sum(((bin_centers - mean) ** 4) * P)
    skewness = skew / mu2**1.5
    excess_kurt = kurt / mu2**2 - 3
    return mean, mu2, skewness, excess_kurt

mean, var, skew, kurt = moments_from_prob(xMaxData[0][0], xMaxData[0][1])
print(mean, var, skew, kurt)
#first 50 element, xmax length = 50
i = 0
xAxis = []
while i < 48:
    xAxis.append(monteCarloData[4][i])
    i += 1

arr1 = monteCarloData[0][1]
arr2 = monteCarloData[1][1]
arr3 = monteCarloData[2][1]
arr4 = monteCarloData[3][1]

data = np.array(xMaxData[0][1][:48], dtype=float)
params = create_params(a=0.68,b=0.11,c=0.17,d=0.04)



def residual(pars):
    """Calculate chi-square between model and data."""

    a = pars[0]
    b = pars[1]
    c = pars[2]
    d = pars[3]

    model = (
        a * arr1["Frac"][:48]
        + b * arr2["Frac"][:48]
        + c * arr3["Frac"][:48]
        + d * arr4["Frac"][:48]
    )

    chi = 0.0

    for i in range(len(model)):
        if data[i] != 0:
            chi += ((data[i] - model[i])/np.sqrt(data[i])) ** 2 

    return chi

#mc, xmax skew és kurt különbsége minél kisebb legyen
# Initial parameter values
params = np.array([
    0.4,  # a
    0.03,  # b
    0.41,  # c
    0.15   # d
])

# Optimization
out = optimize.minimize(
    residual,
    params,
    method="SLSQP"
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

model_fit = (
    a_fit*arr1["Frac"][:48]
    + b_fit*arr2["Frac"][:48]
    + c_fit*arr3["Frac"][:48]
    + d_fit*arr4["Frac"][:48]
)

#plot xMax, lgE 18
plt.plot(xAxis, xMaxData[0][1][:48], 'o', markersize=2, color="r", label='Auger Xmax data')
plt.plot(xAxis, model_fit, 'o', markersize=2, color='b', label='MC composition')
plt.xlabel("Xmax")
plt.title("MC composition on Xmax data")
plt.legend()
plt.show()

