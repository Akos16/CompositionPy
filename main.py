from monteCarloDataHandler import monteCarloDataHandler
from compostionHandler import compositionHandler
import matplotlib.pyplot as plt
import pandas as pd
dataClass = monteCarloDataHandler()
compHandler = compositionHandler()

data = dataClass.getData()
print(data)
