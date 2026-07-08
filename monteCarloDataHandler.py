import os
import pandas as pd

class monteCarloDataHandler:
    def readAndRebin(self, filename, start=504, step=12):
        df = pd.read_csv(filename, sep="\t", decimal=",")

        uj_sorok = []

        i = 0
        while i < len(df):
            if df.iloc[i, 0] < start:
                i += 1
                continue

            sor = {"bin": df.iloc[i, 0]}

            for col in df.columns[1:]:
                osszeg = 0
                for j in range(step):
                    if i + j < len(df):
                        osszeg += df.iloc[i + j][col]
                sor[col] = osszeg

            uj_sorok.append(sor)
            i += step

        return pd.DataFrame(uj_sorok)         

    def getMonteCarloData(self):
        folder = "./MonteCarloSimulations"
        files = [f for f in os.listdir(folder) if f.endswith(".txt")]
        n_files = len(files) - 1
        
        lgEs = ["17,5", "18", "18,5", "19"]
        #Xmax-ok
        componentsX = []
        filenameX = f"./MonteCarloSimulations/component0.txt"
        df = self.readAndRebin(filenameX)
        
        
        componentsX = df["bin"]
        componentsY0 = []
        componentsY1 = []
        componentsY2 = []
        componentsY3 = []
        for i in range(n_files):
            filename1 = f"./MonteCarloSimulations/component{i}.txt"
            df = self.readAndRebin(filename1)
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
