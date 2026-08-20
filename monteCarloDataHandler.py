import os
import pandas as pd
import numpy as np


class monteCarloDataHandler:

    def readAndRebin(self, filename, start=504, step=12):
        df = pd.read_csv(filename, sep="\t", decimal=",")

        new_lines = []

        # Minden bin-pár külön van kezelve
        for col_idx in range(0, len(df.columns), 2):

            bin_col = df.columns[col_idx]
            value_col = df.columns[col_idx + 1]

            # Csak a start-tól kezdődő adatok
            data = df[df[bin_col] >= start].copy()

            if data.empty:
                continue

            # Az eredményhez tartozó bin értékek
            bins = range(
                start,
                int(data[bin_col].max()) + 1,
                step
            )

            # Első oszlopnál létrehozzuk a sorokat
            if col_idx == 0:
                for new_bin in bins:

                    mask = (
                        (data[bin_col] >= new_bin) &
                        (data[bin_col] < new_bin + step)
                    )

                    osszeg = data.loc[mask, value_col].sum()

                    new_lines.append([new_bin, osszeg])

            else:
                # A további energiaoszlopok értékeit
                # hozzáadjuk a már meglévő sorokhoz
                for row, new_bin in zip(new_lines, bins):

                    mask = (
                        (data[bin_col] >= new_bin) &
                        (data[bin_col] < new_bin + step)
                    )

                    osszeg = data.loc[mask, value_col].sum()

                    row.append(new_bin)
                    row.append(osszeg)

        return new_lines

    def getMonteCarloData(self):

        folder = "./MonteCarloSimulations"
        files = [f for f in os.listdir(folder) if f.endswith(".txt")]
        n_files = len(files) - 1

        lgEs = ["17,5", "18", "18,5", "19"]

        # Xmax-ok
        filenameX = "./MonteCarloSimulations/component0.txt"

        data = self.readAndRebin(filenameX)

        df = pd.DataFrame(data, columns=[
            "bin", "17,5",
            "bin", "18",
            "bin", "18,5",
            "bin", "19"
        ])

        componentsX = df.iloc[:, 0]

        componentsY0 = []
        componentsY1 = []
        componentsY2 = []
        componentsY3 = []

        for i in range(n_files):

            filename1 = f"./MonteCarloSimulations/component{i}.txt"

            data = self.readAndRebin(filename1)

            df = pd.DataFrame(data, columns=[
                "bin", "17,5",
                "bin", "18",
                "bin", "18,5",
                "bin", "19"
            ])

            match i:

                case 0:
                    for y in lgEs:
                        componentsY0.append(
                            pd.DataFrame({
                                'Xmax': componentsX.values,
                                'Frac': df[y].values / df[y].values.sum()
                            })
                        )

                case 1:
                    for y in lgEs:
                        componentsY1.append(
                            pd.DataFrame({
                                'Xmax': componentsX.values,
                                'Frac': df[y].values / df[y].values.sum()
                            })
                        )

                case 2:
                    for y in lgEs:
                        componentsY2.append(
                            pd.DataFrame({
                                'Xmax': componentsX.values,
                                'Frac': df[y].values / df[y].values.sum()
                            })
                        )

                case 3:
                    for y in lgEs:
                        componentsY3.append(
                            pd.DataFrame({
                                'Xmax': componentsX.values,
                                'Frac': df[y].values / df[y].values.sum()
                            })
                        )

        return componentsY0, componentsY1, componentsY2, componentsY3, componentsX