import matplotlib.pyplot as plt
import numpy as np

# Energiaszintek
lgE = np.array([18.0, 18.5, 19.0])

# 1. Eredeti Auger adatok
auger_H  = np.array([0.40, 0.26, 0.14])
auger_He = np.array([0.03, 0.21, 0.00])
auger_N  = np.array([0.41, 0.47, 0.84])
auger_Fe = np.array([0.16, 0.06, 0.02])

# 2. Poisson MC adatok (érték + hiba)
p_H,  p_H_err  = np.array([0.1461597, 0.2349757, 0.0437722]), np.array([0.0413181, 0.0608978, 0.0300447])
p_He, p_He_err = np.array([0.5146711, 0.1788756, 0.0000000]), np.array([0.0749034, 0.1037224, 0.0513584])
p_N,  p_N_err  = np.array([0.0760032, 0.3658201, 0.9118170]), np.array([0.0638503, 0.0867676, 0.0767701])
p_Fe, p_Fe_err = np.array([0.2631659, 0.2203286, 0.0444108]), np.array([0.0293993, 0.0425000, 0.0513000])

# 3. MC chi2 adatok
mc_H  = np.array([0.39417570342381586, 0.22404119607556078, 0.0])
mc_He = np.array([0.5263495113403392,  0.7126064294399276,  0.7939209390224525])
mc_N  = np.array([0.012860302401582153, 1.0639319293146628e-18, 0.20607906097754758])
mc_Fe = np.array([0.06661448283426268, 0.0633523744845116,   5.472256355736407e-18])

# Plot struktúra létrehozása (4 panel egymás alatt)
fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)

# Elemek konfigurációja fentről lefelé: Fe, N, He, H
elements = [
    ('f_{Fe}', auger_Fe, p_Fe, p_Fe_err, mc_Fe, axes[0]),
    ('f_{N}',  auger_N,  p_N,  p_N_err,  mc_N,  axes[1]),
    ('f_{He}', auger_He, p_He, p_He_err, mc_He, axes[2]),
    ('f_{H}',  auger_H,  p_H,  p_H_err,  mc_H,  axes[3]),
]

c_auger = 'blue'
c_poisson = 'red'
c_mc = 'green'

for title, auger, p_val, p_err, mc_val, ax in elements:
    # 1. Auger (kék korong)
    ax.plot(lgE, auger, 'o', color=c_auger, label='Auger (eredeti)', markersize=7)
    for x, y in zip(lgE, auger):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(-16, 6), fontsize=8, color=c_auger, fontweight='bold')
        
    # 2. Poisson MC (piros háromszög + hibasáv)
    ax.errorbar(lgE, p_val, yerr=p_err, fmt='^', color=c_poisson, label='Poisson MC', capsize=4, markersize=7)
    for x, y in zip(lgE, p_val):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(8, -4), fontsize=8, color=c_poisson, fontweight='bold')

    # 3. MC chi2 (narancs négyzet)
    ax.plot(lgE, mc_val, 's', color=c_mc, label='MC chi2', markersize=6)
    for x, y in zip(lgE, mc_val):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(-16, -12), fontsize=8, color=c_mc, fontweight='bold')

    ax.set_ylabel(f'${title}$', fontsize=12)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(True, linestyle='--', alpha=0.5)

# Jelmagyarázat a legfelső panelre
axes[0].legend(loc='upper right', ncol=3)

# Tengelybeállítások
axes[3].set_xlabel('$\lg(E / \mathrm{eV})$', fontsize=12)
axes[3].set_xticks([18.0, 18.5, 19.0])

plt.tight_layout()
plt.subplots_adjust(hspace=0.1)  # Panolok közötti hézag csökkentése
plt.show()