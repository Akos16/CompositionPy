from dataHandler import dataHandler
from compostionHandler import compositionHandler

dataClass = dataHandler()
data = dataClass.getData()
compHandler = compositionHandler()
wComp = compHandler.weightedComposition(data)
print(wComp[0])

print(f"Comp0: {wComp[1]}, TempComp0: {wComp[2]}, ParamComp0: {wComp[3]}")