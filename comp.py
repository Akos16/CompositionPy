import matplotlib.pyplot as plt
import numpy as np

lgE = np.array([18.0, 18.5, 19.0])

#Auger data
auger_H  = np.array([0.40, 0.26, 0.14])
auger_He = np.array([0.03, 0.21, 0.00])
auger_N  = np.array([0.41, 0.47, 0.84])
auger_Fe = np.array([0.16, 0.06, 0.02])

#poisson
p_H,  p_H_err  = np.array([0.11857485448710794, 0.15641704561622746, 0.06431853737170141]), np.array([0.06159687597271662, 0.0858548034782818, 0.05582469377131368])
p_He, p_He_err = np.array([0.4877060313479121, 0.31255033021138384, 0.0000000]), np.array([0.11547654022103891, 0.1537221824839191, 0.05519610662754132])
p_N,  p_N_err  = np.array([0.17410845354310223, 0.38614208289082874, 0.9356814626296636]), np.array([0.09663670758271414, 0.1350602500430179, 0.07982345656130518])
p_Fe, p_Fe_err = np.array([0.21961066062187773, 0.14489054128155998, 0.00]), np.array([0.04154195847447198, 0.06479882130228942, 0.043867017203899704])

#mc
mc_H  = np.array([0.38287602214272937, 0.3128534246832627, 2.6020852139652106e-18])
mc_He = np.array([0.4924803933234386,  0.5903807730053451,  0.9258880514159702])
mc_N  = np.array([0.08479132978207324, 0.041419645029541194, 0.07411194858402985])
mc_Fe = np.array([0.03985225475175884, 0.05534615728185102,   8.893845946170154e-19])

#plot 4 y
fig, axes = plt.subplots(4, 1, figsize=(5, 7), sharex=True)

elements = [
    ('f_{Fe}', auger_Fe, p_Fe, p_Fe_err, mc_Fe, axes[0]),
    ('f_{N}',  auger_N,  p_N,  p_N_err,  mc_N,  axes[1]),
    ('f_{He}', auger_He, p_He, p_He_err, mc_He, axes[2]),
    ('f_{H}',  auger_H,  p_H,  p_H_err,  mc_H,  axes[3]),
]


for title, auger, p_val, p_err, mc_val, ax in elements:
    #Auger
    ax.plot(lgE, auger, 'o', color='blue', label='Auger (eredeti)', markersize=7)
    for x, y in zip(lgE, auger):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(-16, 6), fontsize=8, color='blue', fontweight='bold')
        
    #Possion
    ax.errorbar(lgE, p_val, yerr=p_err, fmt='^', color='red', label='Poisson MC', capsize=4, markersize=7)
    for x, y in zip(lgE, p_val):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(8, -4), fontsize=8, color='red', fontweight='bold')

    #MC
    ax.plot(lgE, mc_val, 's', color='green', label='MC chi2', markersize=6)
    for x, y in zip(lgE, mc_val):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(-16, -12), fontsize=8, color='green', fontweight='bold')

    ax.set_ylabel(f'${title}$', fontsize=12)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(True, linestyle='--', alpha=0.5)

axes[0].legend(loc='upper right', ncol=3)
axes[3].set_xlabel('$\lg(E / \mathrm{eV})$', fontsize=12)
axes[3].set_xticks([18.0, 18.5, 19.0])
plt.tight_layout()
plt.subplots_adjust(hspace=0.1)  
plt.show()