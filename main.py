from dataHandler import dataHandler
from compostionHandler import compositionHandler
import matplotlib.pyplot as plt
import pandas as pd
dataClass = dataHandler()
compHandler = compositionHandler()

data = dataClass.getData()
wComp = compHandler.weightedComposition(data[0].Frac, data[1].Frac, data[2].Frac, data[3].Frac)
print(wComp)

#data[0].to_csv("./Datas/comp0.csv", index=False)


plt.plot(data[0]['Xmax'], data[0]['Frac'])
plt.xticks(data[0]['Xmax'])
plt.xlabel('Xmax')
plt.ylabel('Frac')
plt.savefig('./Datas/comp0.png')

plt.plot(data[0]['Xmax'], wComp)
plt.xticks(data[0]['Xmax'])
plt.xlabel('Xmax')
plt.ylabel('Frac')
plt.savefig('./Datas/wComp.png')
plt.show()

plt.show()