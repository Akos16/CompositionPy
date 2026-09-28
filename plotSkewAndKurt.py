import matplotlib.pyplot as plt
import numpy as np

# Adatok megadása
lgE = np.array([18.0, 18.5, 19.0])

# 1. chi2 (értékek és hibáik)
chi2_skew = np.array([0.71878, 1.30176, 1.60763])
chi2_skew_err = np.array([0.12133, 0.23100, 0.54794])
chi2_kurt = np.array([0.68782, 3.60194, 7.45591])
chi2_kurt_err = np.array([0.35333, 1.05266, 3.20966])

# 2. Monte-Carlo chi2
mc_skew = np.array([1.0819869638028892, 0.9892106648394198, 0.9045422069466365])
mc_kurt = np.array([2.0060479219411045, 1.8311394736142939, 1.4860175578730166])

# 3. Poisson Mc
poisson_skew = np.array([0.7450793, 1.1814918, 1.4565396])
poisson_kurt = np.array([1.1025584, 2.7184287, 7.7624341])

# Plot létrehozása: 1 ábrán 2 részgrafikon (subplot)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 1. Részábra: Skewness (pontok összekötés nélkül)
ax1.errorbar(lgE, chi2_skew, yerr=chi2_skew_err, fmt='o', color='blue', label='Auger adat', capsize=5)
ax1.plot(lgE, mc_skew, 'o', color='orange', label='Monte-Carlo chi2')
ax1.plot(lgE, poisson_skew, 'o', color='green', label='Poisson Mc')
ax1.set_title('Skewness (Ferdesség)')
ax1.set_xlabel('lgE')
ax1.set_ylabel('Skewness')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# 2. Részábra: Kurtosis (pontok összekötés nélkül)
ax2.errorbar(lgE, chi2_kurt, yerr=chi2_kurt_err, fmt='o', color='blue', label='Auger adat', capsize=5)
ax2.plot(lgE, mc_kurt, 'o', color='orange', label='Monte-Carlo chi2')
ax2.plot(lgE, poisson_kurt, 'o', color='green', label='Poisson Mc')
ax2.set_title('Kurtosis (Csúcsosság)')
ax2.set_xlabel('lgE')
ax2.set_ylabel('Kurtosis')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.show()