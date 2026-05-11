from dataHandler import dataHandler
from compostionHandler import compositionHandler

dataClass = dataHandler()
data = dataClass.getData()
compHandler = compositionHandler()
wComp = compHandler.weightedComposition(data)
print(wComp)