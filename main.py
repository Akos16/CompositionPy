from dataHandler import dataHandler
from compostionHandler import compositionHandler
import matplotlib.pyplot as plt
import pandas as pd
dataClass = dataHandler()
data = dataClass.getData()
#compHandler = compositionHandler()
#wComp = compHandler.weightedComposition(data)
#print(data)
#df = pd.DataFrame(wComp)
#df.to_csv("./Datas/out.csv")
print(data[0])
data[0].to_csv("./Datas/comp0.csv", index=False)
# Sample DataFrame 501, 0-tól

plt.plot(data[0]['Xmax'], data[0]['Frec'])
plt.xticks(data[0]['Xmax'])
plt.xlabel('Xmax')
plt.ylabel('Frec')
plt.savefig('./Datas/comp0.png')
plt.show()