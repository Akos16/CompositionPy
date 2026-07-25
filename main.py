from monteCarloDataHandler import monteCarloDataHandler
from xMaxDataHandler import xMaxDataHandler
from compostionHandler import compositionHandler
import matplotlib.pyplot as plt
import pandas as pd

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
#parameters
a = 0.398609
b = 0.032087
c = 0.414095
d = 0.15521
#function
model = a*arr1['Frac'] + b*arr2['Frac'] + c*arr3['Frac'] + d*arr4['Frac']

#first 50 element (model), xmax length = 50
y = 0
counter = []
while y < 50:
    counter.append(model[y])
    y += 1

#plot xMax, lgE 18
plt.plot(xAxis, xMaxData[0][1], '.', color="r", label='Auger Xmax data')
plt.plot(xAxis, counter, 'o', markersize=3, color='b', label='MC composition')
plt.xlabel("Xmax")
plt.title("MC composition on Xmax data")
plt.legend()
plt.show()

