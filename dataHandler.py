import os
import pandas as pd

class dataHandler:
    def getData(self):
        folder = "./MainDatas"
        files = [f for f in os.listdir(folder) if f.endswith(".txt")]
        n_files = len(files) - 1
        
        lgEs = ["17,5", "18", "18,5", "19"]
        #Xmax-ok
        componentsX = []
        filenameX = f"./MainDatas/component0.txt"
        df = pd.read_csv(filenameX, sep="\t", decimal = ',')
        
        componentsX = df["bin"]
        componentsY0 = []
        componentsY1 = []
        componentsY2 = []
        componentsY3 = []
        for i in range(n_files):
            filename1 = f"./MainDatas/component{i}.txt"
            df = pd.read_csv(filename1, sep="\t", decimal=",")
            match i: 
                case 0: 
                    for y in lgEs:
                        componentsY0.append(pd.DataFrame({'Xmax': componentsX.values, 'Frac': df[y].values / df[y].values.sum()}))
                case 1: 
                    for y in lgEs:
                        componentsY1.append(pd.DataFrame({'Xmax': componentsX.values, 'Frac': df[y].values / df[y].values.sum()}))
                case 2: 
                    for y in lgEs:
                        componentsY2.append(pd.DataFrame({'Xmax': componentsX.values, 'Frac': df[y].values / df[y].values.sum()}))
                case 3: 
                    for y in lgEs:
                        componentsY3.append(pd.DataFrame({'Xmax': componentsX.values, 'Frac': df[y].values / df[y].values.sum()}))
        
        return componentsY0, componentsY1, componentsY2, componentsY3, componentsX
    
    def getDataStats(self, components):
        componentsStatsMax = []
        componentsStatsMin = []
        componentsStatsLength = []
        lgEs = ["17,5", "18", "18,5", "19"]
        for i in range(4):
            for z in lgEs:
                componentsStatsMax.append(components[i][z].max())
                componentsStatsMin.append(components[i][z][components[i][z] > 0].min())
                componentsStatsLength.append(len(components[i][z]))
        return componentsStatsMax, componentsStatsMin, componentsStatsLength
