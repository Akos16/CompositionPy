from monteCarloDataHandler import monteCarloDataHandler
from xMaxDataHandler import xMaxDataHandler
from compostionHandler import compositionHandler
import matplotlib.pyplot as plt
import pandas as pd

monteCarloData = monteCarloDataHandler().getMonteCarloData()
xMaxData = xMaxDataHandler().getXmaxData()

print(xMaxData[0][1])