import itertools
import os
from compostionHandler import compositionHandler
from lmfit import create_params, fit_report, minimize
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from monteCarloDataHandler import monteCarloDataHandler
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.optimize import minimize
import scipy.optimize as optimization
from xMaxDataHandler import xMaxDataHandler

monteCarloData = monteCarloDataHandler().getMonteCarloData()
xMaxHandler = xMaxDataHandler()
xMaxData = xMaxHandler.getXmaxData()

arrAugerComp = [
    [0.40, 0.03, 0.41, 0.16],
    [0.26, 0.21, 0.47, 0.06],
    [0.14, 0.00, 0.84, 0.02],
]
lgE_values = ["18", "18.5", "19"]

# Kimeneti mappa a hőtérképeknek
output_dir = "heatmaps_ab_ba_log"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# 1. ILLESZTÉS ÉS ALAP GRAFIKONOK
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
fitted_params_per_lgE = []

for idx in range(3):
  lgE = idx + 1
  xMax = idx
  ax = axes[idx]

  (
      meandata,
      meandata_err,
      vardata,
      vardata_err,
      skewdata,
      skewdata_err,
      kurtdata,
      kurtdata_err,
  ) = xMaxHandler.moments_with_errors(
      xMaxData[xMax][0][:48], xMaxData[xMax][1][:48], xMaxData[xMax][2][:48]
  )

  arr1 = monteCarloData[0][lgE][:48]
  arr2 = monteCarloData[1][lgE][:48]
  arr3 = monteCarloData[2][lgE][:48]
  arr4 = monteCarloData[3][lgE][:48]
  arr5 = monteCarloData[4][:48]

  data = np.array(xMaxData[xMax][1][:48], dtype=float)
  params = [0.10, 0.25, 0.25, 0.40]

  def model(pars):
    a = pars[0]
    b = pars[1]
    c = pars[2]
    d = pars[3]
    return (
        a * arr1["Frac"]
        + b * arr2["Frac"]
        + c * arr3["Frac"]
        + d * arr4["Frac"]
    )

  def residual(pars):
    this_model = model(pars)
    chi = 0.0
    for i in range(len(this_model)):
      if data[i] != 0:
        chi += ((data[i] - this_model[i]) / np.sqrt(data[i])) ** 2
    return chi

  def mean(pars):
    this_model = model(pars)
    mcmean, mcmu2, mcskew, mc_excesskurt = xMaxHandler.moments_from_prob(
        xMaxData[xMax][0][:48], this_model
    )
    return ((meandata - mcmean) / meandata_err) ** 2

  def mu(pars):
    this_model = model(pars)
    mcmean, mcmu2, mcskew, mc_excesskurt = xMaxHandler.moments_from_prob(
        xMaxData[xMax][0][:48], this_model
    )
    return ((vardata - mcmu2) / vardata_err) ** 2

  def skew(pars):
    this_model = model(pars)
    mcmean, mcmu2, mcskew, mc_excesskurt = xMaxHandler.moments_from_prob(
        xMaxData[xMax][0][:48], this_model
    )
    return ((skewdata - mcskew) / skewdata_err) ** 2

  def kurt(pars):
    this_model = model(pars)
    mcmean, mcmu2, mcskew, mc_excesskurt = xMaxHandler.moments_from_prob(
        xMaxData[xMax][0][:48], this_model
    )
    return ((kurtdata - mc_excesskurt) / kurtdata_err) ** 2

  bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]
  constr_unit = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
  constraints = [constr_unit]

  out = optimize.minimize(
      residual, params, method="SLSQP", bounds=bounds, constraints=constraints
  )

  print(f"--- lgE = {lgE_values[idx]} ---")
  print("Success:", out.success)
  print("Message:", out.message)
  print("a =", out.x[0])
  print("b =", out.x[1])
  print("c =", out.x[2])
  print("d =", out.x[3])
  print("chi2 =", out.fun)

  fitted_params_per_lgE.append(out.x)

  a_fit, b_fit, c_fit, d_fit = out.x
  model_fit = (
      a_fit * arr1["Frac"][:48]
      + b_fit * arr2["Frac"][:48]
      + c_fit * arr3["Frac"][:48]
      + d_fit * arr4["Frac"][:48]
  )
  mcmeantest, mcmu2test, mcskewtest, mc_excesskurttest = (
      xMaxHandler.moments_from_prob(xMaxData[xMax][0][:48], model_fit)
  )

  l1 = ax.errorbar(
      monteCarloData[4][:48],
      xMaxData[xMax][1][:48],
      yerr=xMaxData[xMax][2][:48],
      fmt="o",
      markersize=1,
      capsize=1,
      elinewidth=1,
      color="black",
      zorder=3,
      label="Auger Xmax data",
  )
  l2 = ax.plot(
      monteCarloData[4][:48],
      model_fit,
      "-",
      markersize=2,
      color="skyblue",
      label=f"MC composition",
  )

  ax.text(
      0.65,
      0.60,
      f"$f_H$={out.x[0]:.2f}\n$f_{{He}}$={out.x[1]:.2f}\n$f_N$={out.x[2]:.2f}\n$f_{{Fe}}$={out.x[3]:.2f}",
      transform=ax.transAxes,
      fontsize=14,
  )
  ax.text(
      0.05,
      0.55,
      f"$f_H^{{Auger}}$={arrAugerComp[xMax][0]:.2f}\n$f_{{He}}^{{Auger}}$={arrAugerComp[xMax][1]:.2f}\n$f_N^{{Auger}}$={arrAugerComp[xMax][2]:.2f}\n$f_{{Fe}}^{{Auger}}$={arrAugerComp[xMax][3]:.2f}",
      transform=ax.transAxes,
      fontsize=14,
  )
  ax.set_xlabel("Xmax")
  ax.set_title(f"lgE = {lgE_values[idx]}")
  ax.legend(loc="upper right")

axes[0].set_ylabel("Fraction")
plt.suptitle("MC composition on Xmax data", fontsize=16)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 2. LOGARITMIKUS HEATMAP MENTÉSE ÉS VIZSGÁLATA
# ---------------------------------------------------------
param_names = [r"$f_H$", r"$f_{He}$", r"$f_N$", r"$f_{Fe}$"]
param_chars = ["a", "b", "c", "d"]
index_pairs = list(itertools.combinations(range(4), 2))

grid_size = 50
param_range = np.linspace(0.0, 1.0, grid_size)
X, Y = np.meshgrid(param_range, param_range)
default_params = [0.10, 0.25, 0.25, 0.40]

for idx in range(3):
  lgE = idx + 1
  xMax = idx

  arr1 = monteCarloData[0][lgE][:48]
  arr2 = monteCarloData[1][lgE][:48]
  arr3 = monteCarloData[2][lgE][:48]
  arr4 = monteCarloData[3][lgE][:48]
  data = np.array(xMaxData[xMax][1][:48], dtype=float)

  def residual_local(pars):
    this_model = (
        pars[0] * arr1["Frac"]
        + pars[1] * arr2["Frac"]
        + pars[2] * arr3["Frac"]
        + pars[3] * arr4["Frac"]
    )
    chi = 0.0
    for i in range(len(this_model)):
      if data[i] != 0:
        chi += ((data[i] - this_model[i]) / np.sqrt(data[i])) ** 2
    return chi

  bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]
  constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

  fig, axes = plt.subplots(2, 3, figsize=(18, 10))
  axes = axes.flatten()
  fig.suptitle(
      rf"Log-Scale $\chi^2$ Heatmaps & Start Sensitivity ($xy$ vs $yx$) - lgE ="
      f" {lgE_values[idx]}",
      fontsize=16,
  )

  print(f"\n=================== lgE = {lgE_values[idx]} ===================")

  for p_idx, (ix, iy) in enumerate(index_pairs):
    name_x, name_y = param_chars[ix], param_chars[iy]

    # 1. Normál sorrendű kezdőértékkel való illesztés (xy)
    init_xy = list(default_params)
    out_xy = optimize.minimize(
        residual_local,
        init_xy,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    # 2. Megcserélt kezdőértékkel való illesztés (yx)
    init_yx = list(default_params)
    init_yx[ix], init_yx[iy] = init_yx[iy], init_yx[ix]
    out_yx = optimize.minimize(
        residual_local,
        init_yx,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    print(f"Pár {name_x}{name_y}:")
    print(
        f"  {name_x}{name_y} start -> chi2: {out_xy.fun:.4f} | pos:"
        f" ({out_xy.x[ix]:.3f}, {out_xy.x[iy]:.3f})"
    )
    print(
        f"  {name_y}{name_x} start -> chi2: {out_yx.fun:.4f} | pos:"
        f" ({out_yx.x[ix]:.3f}, {out_yx.x[iy]:.3f})"
    )

    # 2D rács kiszámítása
    grid_chi = np.zeros((grid_size, grid_size))
    base_pars = out_xy.x.copy()

    for r_idx, y_val in enumerate(param_range):
      for c_idx, x_val in enumerate(param_range):
        current_pars = base_pars.copy()
        current_pars[ix] = x_val
        current_pars[iy] = y_val
        grid_chi[r_idx, c_idx] = residual_local(current_pars)

    # A logaritmikus skálázáshoz kiküszöböljük az esetleges nullákat
    min_chi = np.min(grid_chi[grid_chi > 0]) if np.any(grid_chi > 0) else 1e-5
    grid_chi_clean = np.where(grid_chi <= 0, min_chi, grid_chi)

    ax = axes[p_idx]

    # LOGARITMIKUS SZÍNSKÁLA (LogNorm)
    norm = colors.LogNorm(
        vmin=grid_chi_clean.min(), vmax=grid_chi_clean.max()
    )

    c = ax.pcolormesh(
        X, Y, grid_chi_clean, cmap="viridis", norm=norm, shading="auto"
    )

    # Kontúrvonalak logaritmikus beosztással
    try:
      ax.contour(
          X,
          Y,
          grid_chi_clean,
          norm=norm,
          locator=ticker.LogLocator(),
          colors="white",
          alpha=0.3,
          linewidths=0.7,
      )
    except Exception:
      ax.contour(
          X,
          Y,
          grid_chi_clean,
          levels=10,
          colors="white",
          alpha=0.3,
          linewidths=0.7,
      )

    # Normál kezdőpontú illesztés minimuma (Piros pont)
    ax.scatter(
        out_xy.x[ix],
        out_xy.x[iy],
        color="red",
        s=70,
        zorder=5,
        marker="o",
        label=f"min ({name_x}{name_y}): {out_xy.fun:.2f}",
    )

    # Felcserélt kezdőpontú illesztés minimuma (Ciklány X)
    ax.scatter(
        out_yx.x[ix],
        out_yx.x[iy],
        color="cyan",
        s=70,
        zorder=6,
        marker="X",
        label=f"min ({name_y}{name_x}): {out_yx.fun:.2f}",
    )

    fig.colorbar(c, ax=ax, label=r"$\log_{10}(\chi^2)$")
    ax.set_xlabel(f"{param_names[ix]} ({name_x})", fontsize=11)
    ax.set_ylabel(f"{param_names[iy]} ({name_y})", fontsize=11)
    ax.set_title(f"Pár: {name_x} vs {name_y}", fontsize=12)
    ax.legend(loc="upper right", fontsize=8)

  plt.tight_layout()

  # Kép mentése
  save_path = os.path.join(
      output_dir, f"heatmap_log_lgE_{lgE_values[idx]}.png"
  )
  plt.savefig(save_path, dpi=300, bbox_inches="tight")
  print(f"Ábra elmentve: {save_path}")

  plt.close(fig)