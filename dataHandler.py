import os
import pandas as pd

class dataHandler:
    def getData(self):
        folder = "./MainDatas"
        files = [f for f in os.listdir(folder) if f.endswith(".txt")]
        n_files = len(files) - 1
        
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
                    componentsY0 = pd.DataFrame({'Xmax': componentsX.values, 'Frec': df["18"].values / df["18"].values.sum()})
                case 1: 
                    componentsY1 = pd.DataFrame({'Xmax': componentsX.values, 'Frec': df["18"].values / df["18"].values.sum()})
                case 2: 
                    componentsY2 = pd.DataFrame({'Xmax': componentsX.values, 'Frec': df["18"].values / df["18"].values.sum()})
                case 3: 
                    componentsY3 = pd.DataFrame({'Xmax': componentsX.values, 'Frec': df["18"].values / df["18"].values.sum()})
        
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
