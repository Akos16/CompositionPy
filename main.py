from monteCarloDataHandler import monteCarloDataHandler
from xMaxDataHandler import xMaxDataHandler
from compostionHandler import compositionHandler
from lmfit import create_params, fit_report, minimize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


monteCarloData = monteCarloDataHandler().getMonteCarloData()
xMaxData = xMaxDataHandler().getXmaxData()

#first 50 element, xmax length = 50
i = 0
xAxis = []
while i < 50:
    xAxis.append(monteCarloData[4][i])
    i += 1

arr1 = monteCarloData[0][1]
arr2 = monteCarloData[1][1]
arr3 = monteCarloData[2][1]
arr4 = monteCarloData[3][1]

data = np.array(xMaxData[0][1][:50], dtype=float)
params = create_params(a=0.398609,b=0.032087,c=0.414095,d=0.15521)

def residual(pars, data=None):
    """Model a decaying sine wave and subtract data."""
    vals = pars.valuesdict()
    a = vals['a']
    b = vals['b']
    c = vals['c']
    d = vals['d']

    model = (a*arr1["Frac"][:50] + b*arr2["Frac"][:50] + c*arr3["Frac"][:50] + d*arr4["Frac"][:50])   
    if data is None:
        return model
    return model - data

out = minimize(residual, params, kws={"data": data})

print(fit_report(out))


a_fit = out.params["a"].value
b_fit = out.params["b"].value
c_fit = out.params["c"].value
d_fit = out.params["d"].value

model_fit = (
    a_fit*arr1["Frac"][:50]
    + b_fit*arr2["Frac"][:50]
    + c_fit*arr3["Frac"][:50]
    + d_fit*arr4["Frac"][:50]
)

#plot xMax, lgE 18
plt.plot(xAxis, xMaxData[0][1], '-', color="r", label='Auger Xmax data')
plt.plot(xAxis, model_fit, '-', markersize=3, color='b', label='MC composition')
plt.xlabel("Xmax")
plt.title("MC composition on Xmax data")
plt.legend()
plt.show()

