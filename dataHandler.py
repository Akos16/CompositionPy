import os
import pandas as pd

class dataHandler:
    def getData(self):
        folder = "./MainDatas"
        files = [f for f in os.listdir(folder) if f.endswith(".txt")]
        n_files = len(files) - 1
        components = []
        for i in range(n_files):
            filename = f"./MainDatas/component{i}.txt"
            df = pd.read_csv(filename, sep="\t", decimal=",")
            components.append(df)
        return components #composition[0]["18"][260] [0. txt]["lgE"][index, txt 1. sora a 0. elem]
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
