import os 
import pandas as pd

class xMaxDataHandler:
    def getXmaxData(self):
        folder = "./xMaxData"
        files = [f for f in os.listdir(folder) if f.endswith(".txt")]
        n_files = len(files) - 1

        file0 = []
        file1 = []
        file2 = []
        for i in range(n_files):
            filename = f"./xMaxData/XmaxDist_Ebin{i}.txt"
            df = pd.read_csv(filename, sep="\t", decimal=",")
            match i: 
                case 0: 
                        file0 = df["Xmax"].to_numpy(), df["Counts"].to_numpy(), df["CountsSqrt"].to_numpy()
                case 1: 
                        file1 = df["Xmax"].to_numpy(), df["Counts"].to_numpy(), df["CountsSqrt"].to_numpy()
                case 2: 
                        file2 = df["Xmax"].to_numpy(), df["Counts"].to_numpy(), df["CountsSqrt"].to_numpy()
        return file0, file1, file2