from dataHandler import dataHandler
from compostionHandler import compositionHandler
import matplotlib.pyplot as plt
import pandas as pd
dataClass = dataHandler()
data = dataClass.getData()
compHandler = compositionHandler()
wComp = compHandler.weightedComposition(data)
df = pd.DataFrame(wComp)
df.to_csv("./Datas/out.csv")
