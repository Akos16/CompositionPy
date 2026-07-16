from monteCarloDataHandler import monteCarloDataHandler
from xMaxDataHandler import xMaxDataHandler
from compostionHandler import compositionHandler
import matplotlib.pyplot as plt
import pandas as pd

monteCarloData = monteCarloDataHandler().getMonteCarloData()
xMaxData = xMaxDataHandler().getXmaxData()



i = 0
tempArray = []
while i < 50:
    tempArray.append(monteCarloData[4][i])
    i += 1



plt.plot(tempArray, xMaxData[0][1], color="r")

plt.xlabel("Xmax")
plt.title("Xmax test")
plt.legend()
plt.show()
