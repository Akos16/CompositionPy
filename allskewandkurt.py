
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================================
# FIGURE DIRECTORY
# ============================================================================

FIG_DIR = Path("./figs")
FIG_DIR.mkdir(parents=True, exist_ok=True)

#Auger Adat
augerData = pd.DataFrame({
    "lgE": [17.85, 17.95, 18.05, 18.15, 18.25, 18.35,18.45, 18.55, 18.65, 18.75, 18.85, 18.95,19.05, 19.14, 19.25, 19.34, 19.45, 19.62],
    "mean": [709.9, 719.9, 725.2, 736.9, 744.5, 748.0,752.2, 754.5, 756.1, 757.4, 763.6, 764.6,766.4, 767.0, 779.5, 773.1, 787.9, 779.8],
    "mean_err": [1.2, 1.4, 1.5, 1.8, 2.0, 2.0,2.1, 2.2, 2.7, 2.8, 2.9, 3.2,3.3, 3.6, 5.1, 5.0, 9.6, 5.0],
    "var": [59.6, 62.4, 59.5, 64.3, 66.4, 60.2,53.3, 53.5, 54.5, 45.8, 42.8, 43.4,39.0, 36.7, 46.4, 40.1, 53.2, 26.5],
    "var_err": [1.7, 2.1, 2.0, 2.6, 2.6, 2.8,2.9, 3.0, 3.5, 3.4, 3.6, 4.1,3.8, 3.6, 6.2, 4.8, 12.7, 4.8]
})

#poisson data
poisson_data = pd.DataFrame({
    "lgE": [18.0,18.5,19.0],
    "mean": [715.503109,743.743773,760.665924],
    "mean_err": [2.015946,2.961606,3.652190],
    "var": [57.705941,55.181325,41.214547],
    "var_err": [2.008850,2.719291,3.873927],
    "skew": [0.872687,1.013364,1.047389],
    "skew_err": [0.128302,0.161715,0.317238],
    "kurt": [1.465072,1.942328,3.232990],
    "kurt_err": [0.419528,0.541491,1.212128]
})
#mc chi^2
chi2_data = pd.DataFrame({
    "lgE": [18.0,18.5,19.0],
    "mean": [715.6240686703887,742.9591891517671,762.5751286030218],
    "mean_err": [0.00045,0.00051,0.00047],
    "var": [3465.6077502827534,3052.0669403395295,1794.1530071439524],
    "var_err": [251.65305,335.70242,608.11669],
    "sigma": [58.86941268844758,55.245515115161425,42.3574433499468],
    "sigma_err": [2.20820,3.10070,5.94273],
    "skew": [1.1463398263237694,1.048093075501054,0.879875137414128],
    "skew_err": [0.04309,0.05878,0.18375],
    "kurt": [2.1624185121755612,1.8990284320299022,1.4351568869263982],
    "kurt_err": [0.09129,0.11375,0.46798],
    "f_H": [0.3828760256671434,0.3128533188152391,0.0],
    "f_H_err": [0.06710,0.10225,0.18100],
    "f_He": [0.49248038325431664,0.5903809655454867,0.9258880514095215],
    "f_He_err": [0.12563,0.17462,0.33192],
    "f_N": [0.08479133992559,0.041419554357772295,0.07411194859047858],
    "f_N_err": [0.06330,0.07723,0.16175],
    "f_Fe": [0.03985225115295,0.05534616128150207,0.0],
    "f_Fe_err": [0.01846,0.02909,0.06423]
})

#skew és kurt a szakdogából
my_data = pd.DataFrame({
    "lgE": [17.85,17.95,18.05,18.15,18.25,18.35,18.60,19.30],
    "skew": [0.6545267102542,0.6745059058334941,0.7630452929430325,1.038290038107693,0.990543672910973,1.1342456676713384,1.4135175397049822,1.7531892594033571],
    "skew_err": [0.1238647874286272,0.125710475782923,0.11694648037783201,0.22307181756130617,0.1556729433803088,0.1783884504762278,0.26613811489976136,0.7591671164111837],
    "kurt": [0.7500574752724325,0.6770084992108178,0.6986291151541688,2.5932883390588763,1.408393448319341,2.080512468062783,4.616204997261213,9.585748661695416],
    "kurt_err": [0.37130452593811186,0.36882628861589467,0.3378649884217547,0.9319468866653724,0.55220049069602,0.6458569685486962,1.3239053232806413,4.6240627280781945]
})

#szakdoga chi^2
thesis_data = pd.DataFrame({
    "lgE": [17.85,17.95,18.05,18.15,18.25,18.35,18.60,19.30],
    "chi2": [25.26351029255698,41.50593853655931,14.324816312701216,21.40169519006541,28.783991480059008,33.57569201357938,17.399330184996085,15.26352683854746],
    "ndf": [47] * 8,
    "chi2red": [0.5375214955863187,0.8831050752459428,0.30478332580215356,0.4553552168099024,0.6124253506395534,0.7143764258208378,0.3701985145743848,0.32475589018186085]
})

#mean
plt.figure(figsize=(8, 6))

#Auger data
plt.errorbar(augerData["lgE"],augerData["mean"],yerr=augerData["mean_err"],fmt="s",markersize=5,elinewidth=2,capsize=3,label="Intézményi eredeti teljes adat")
# Poisson
plt.errorbar(poisson_data["lgE"],poisson_data["mean"],yerr=poisson_data["mean_err"],fmt="D",markersize=7,elinewidth=2,capsize=3,label="Poisson illesztés")
# Chi2
plt.errorbar(chi2_data["lgE"],chi2_data["mean"],yerr=chi2_data["mean_err"],fmt="o",markersize=7,elinewidth=2,capsize=3,label=r"$\chi^2$ illesztés")

plt.grid(True,linestyle="--",alpha=0.5)
plt.tick_params(axis="both",labelsize=14)
plt.ylabel(r"$\langle X_{\max} \rangle$ [g/cm$^2$]",fontsize=16)
plt.xlabel(r"$\log(E_{proton})$",fontsize=16)

plt.legend(fontsize="large", loc="best")
plt.tight_layout()
plt.savefig(FIG_DIR / "mean_vs_lgE.png", dpi=300)
plt.show()

#var
plt.figure(figsize=(8, 6))
#Auger data
plt.errorbar(augerData["lgE"], augerData["var"], yerr=augerData["var_err"], fmt="s", markersize=5, elinewidth=2, capsize=3, label="Intézményi eredeti teljes adat")
#Poisson
plt.errorbar(poisson_data["lgE"], poisson_data["var"], yerr=poisson_data["var_err"], fmt="D", markersize=7, elinewidth=2, capsize=3, label="Poisson illesztés")
#Chi^2
plt.errorbar(chi2_data["lgE"], chi2_data["sigma"], yerr=chi2_data["sigma_err"], fmt="o", markersize=7, elinewidth=2, capsize=3, label=r"$\chi^2$ illesztés")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tick_params(axis="both", labelsize=14)
plt.ylabel(r"$\sigma$ [g/cm$^2$]", fontsize=16)
plt.xlabel(r"$\log(E_{proton})$", fontsize=16)
plt.legend(fontsize="large", loc="best")
plt.tight_layout()
plt.savefig(FIG_DIR / "sigma_vs_lgE.png", dpi=300)
plt.show()

#skewness
plt.figure(figsize=(8, 6))
#szakdoga
plt.errorbar(my_data["lgE"], my_data["skew"], yerr=my_data["skew_err"], fmt="s", markersize=6, elinewidth=2, capsize=3, label="Saját eredmény")
#Poisson
plt.errorbar(poisson_data["lgE"], poisson_data["skew"], yerr=poisson_data["skew_err"], fmt="D", markersize=7, elinewidth=2, capsize=3, label="Poisson illesztés")
#Chi^2
plt.errorbar(chi2_data["lgE"], chi2_data["skew"], yerr=chi2_data["skew_err"], fmt="o", markersize=7, elinewidth=2, capsize=3, label=r"$\chi^2$ illesztés")
#Gumbel tankönyvi
plt.axhline(y=1.14, color="red", linestyle="--", linewidth=1.5, label=r"$\gamma_1$(Gumbel) = 1.14")

plt.grid(True, linestyle="--", alpha=0.5)
plt.tick_params(axis="both", labelsize=14)
plt.ylabel(r"$\gamma_1$", fontsize=16)
plt.xlabel(r"$\log(E_{proton})$", fontsize=16)
plt.legend(fontsize="large", loc="best")
plt.tight_layout()
plt.savefig(FIG_DIR / "skew_vs_lgE.png", dpi=300)
plt.show()

#kurtosis
plt.figure(figsize=(8, 6))

#szakdoga
plt.errorbar(my_data["lgE"], my_data["kurt"], yerr=my_data["kurt_err"], fmt="s", markersize=6, elinewidth=2, capsize=3, label="Saját eredmény")
#Poisson
plt.errorbar(poisson_data["lgE"], poisson_data["kurt"], yerr=poisson_data["kurt_err"], fmt="D", markersize=7, elinewidth=2, capsize=3, label="Poisson illesztés")
#Chi^2
plt.errorbar(chi2_data["lgE"], chi2_data["kurt"], yerr=chi2_data["kurt_err"], fmt="o", markersize=7, elinewidth=2, capsize=3, label=r"$\chi^2$ illesztés")
#Gumbel tankönyvi
plt.axhline(y=2.4, color="red", linestyle="--", linewidth=1.5, label=r"$\beta_2$(Gumbel) = 2.4")

plt.grid(True, linestyle="--", alpha=0.5)
plt.tick_params(axis="both", labelsize=14)
plt.ylabel(r"$\beta_2$", fontsize=16)
plt.xlabel(r"$\log(E_{proton})$", fontsize=16)
plt.legend(fontsize="large", loc="best")
plt.tight_layout()
plt.savefig(FIG_DIR / "kurt_vs_lgE.png", dpi=300)
plt.show()

#chi^2
plt.figure(figsize=(8, 6))
plt.plot(thesis_data["lgE"], thesis_data["chi2red"], "o-", markersize=7, label=r"$\chi^2$/ndf")
plt.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, label=r"$\chi^2$/ndf = 1")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tick_params(axis="both", labelsize=14)
plt.ylabel(r"$\chi^2$/ndf", fontsize=16)
plt.xlabel(r"$\log(E_{proton})$", fontsize=16)
plt.legend(fontsize="large", loc="best")
plt.tight_layout()
plt.savefig(FIG_DIR / "chi2_vs_lgE.png", dpi=300)
plt.show()

