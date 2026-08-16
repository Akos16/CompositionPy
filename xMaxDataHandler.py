import os 
import pandas as pd
import numpy as np
class xMaxDataHandler:
    def getXmaxData(self):
        folder = "./xMaxData"
        files = [f for f in os.listdir(folder) if f.endswith(".txt")]
        n_files = len(files) 

        file0 = []
        file1 = []
        file2 = []
        for i in range(n_files):
            filename = f"./xMaxData/XmaxDist_Ebin{i}.txt"
            df = pd.read_csv(filename, sep="\t", decimal=",")
            match i: 
                case 0: 
                        file0 = df["Xmax"].to_numpy(), df["Counts"].to_numpy() / df["Counts"].to_numpy().sum(), df["CountsSqrt"].to_numpy(dtype=float) / df["Counts"].to_numpy(dtype=float).sum()
                case 1: 
                        file1 = df["Xmax"].to_numpy(), df["Counts"].to_numpy() / df["Counts"].to_numpy().sum(), df["CountsSqrt"].to_numpy(dtype=float) / df["Counts"].to_numpy(dtype=float).sum()
                case 2: 
                        file2 = df["Xmax"].to_numpy(), df["Counts"].to_numpy() / df["Counts"].to_numpy().sum(), df["CountsSqrt"].to_numpy(dtype=float) / df["Counts"].to_numpy(dtype=float).sum()
        return file0, file1, file2
    def moments_from_prob(self, bin_centers, hist_counts):
        bin_centers = np.asarray(bin_centers, dtype=float)
        hist_counts = np.asarray(hist_counts, dtype=float)
        P = hist_counts / hist_counts.sum()
        mean = np.sum(bin_centers * P)
        mu2 = np.sum(((bin_centers - mean) ** 2) * P)
        skew = np.sum(((bin_centers - mean) ** 3) * P)
        kurt = np.sum(((bin_centers - mean) ** 4) * P)
        skewness = skew / mu2**1.5
        excess_kurt = kurt / mu2**2 - 3
        return mean, mu2, skewness, excess_kurt
    def moments_with_errors(self, x_or_edges, counts, count_err, ntoy=5000, seed=0):
        """
        Returns:
        mean, mean_err
        variance, var_err
        skew, skew_err
        kurtosis, kurt_err
        """

        rng = np.random.default_rng(seed)

        counts = np.asarray(counts, float)
        errs   = np.asarray(count_err, float)

        # ---- baseline values ----
        mean0, var0, skew0, kurt0 = self.moments_from_prob(x_or_edges, counts)

        mean_list = []
        var_list  = []
        skew_list = []
        kurt_list = []

        for _ in range(ntoy):

                # Gaussian toy histogram
                toy = counts + rng.normal(0.0, errs)

                # enforce positivity
                toy = np.clip(toy, 0.0, None)

                if np.sum(toy) <= 0:
                        continue

                m, v, s, k = self.moments_from_prob(x_or_edges, toy)

                mean_list.append(m)
                var_list.append(v)
                skew_list.append(s)
                kurt_list.append(k)

        mean_err = np.std(mean_list, ddof=1)
        var_err  = np.std(var_list,  ddof=1)
        skew_err = np.std(skew_list, ddof=1)
        kurt_err = np.std(kurt_list, ddof=1)

        return (mean0, mean_err, var0, var_err, skew0, skew_err, kurt0, kurt_err)