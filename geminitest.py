import os
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.ndimage import gaussian_filter1d
# Feltételezve, hogy ez a modul elérhető a te környezetedben:
from monteCarloDataHandler import monteCarloDataHandler

SPECIES = ("H", "He", "N", "Fe")

def aligned_fit_table(data_path, mc_data, mc_energy_index):
    """Join data and MC by their numerical Xmax coordinate."""
    data = pd.read_csv(data_path, sep="\t")[["Xmax", "Counts"]]
    table = data.copy()
    
    for species_index, species_name in enumerate(SPECIES):
        component = mc_data[species_index][mc_energy_index][["Xmax", "Frac"]].copy()
        component = component.rename(columns={"Frac": species_name})
        table = table.merge(
            component,
            on="Xmax",
            how="inner",
            validate="one_to_one",
        )
        
    if table.empty:
        raise ValueError("Data and MC have no common Xmax bins")
        
    retained = table["Xmax"].unique()
    dropped = data.loc[~data["Xmax"].isin(retained)]
    
    if np.any(dropped["Counts"].to_numpy() > 0):
        raise ValueError(
            "The common data/MC fit range would discard observed events at "
            f"Xmax={dropped.loc[dropped['Counts'] > 0, 'Xmax'].to_list()}"
        )
        
    return table.sort_values("Xmax").reset_index(drop=True)

def prepare_arrays(table, resolution_sigma=None):
    """Return raw data counts and normalized species templates."""
    counts = table["Counts"].to_numpy(dtype=float)
    templates = table[list(SPECIES)].to_numpy(dtype=float).T
    
    if resolution_sigma is not None:
        xmax = table["Xmax"].to_numpy(dtype=float)
        bin_width = np.median(np.diff(xmax))
        sigma_bins = resolution_sigma / bin_width
        templates = gaussian_filter1d(
            templates,
            sigma=sigma_bins,
            axis=1,
            mode="constant",
        )
        
    row_sums = templates.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("At least one species template is empty")
        
    templates = templates / row_sums
    unsupported = (counts > 0) & (templates.sum(axis=0) <= 0)
    
    if np.any(unsupported):
        bad_xmax = table.loc[unsupported, "Xmax"].to_list()
        raise ValueError(
            "Observed events occur in unsupported MC bins at Xmax="
            f"{bad_xmax}. Convolve with the detector response, increase "
            "the MC sample, or include finite-MC uncertainty."
        )
        
    return counts, templates

def model_probabilities(fractions, templates):
    probability = np.asarray(fractions) @ templates
    total = probability.sum()
    if total <= 0:
        return np.full_like(probability, np.nan)
    return probability / total

def expected_counts(fractions, counts, templates):
    return counts.sum() * model_probabilities(fractions, templates)

def poisson_deviance(fractions, counts, templates):
    expected = expected_counts(fractions, counts, templates)
    if np.any(~np.isfinite(expected)):
        return np.inf
    if np.any((counts > 0) & (expected <= 0)):
        return np.inf
        
    terms = expected - counts
    nonzero = counts > 0
    terms[nonzero] += counts[nonzero] * np.log(
        counts[nonzero] / expected[nonzero]
    )
    return 2.0 * terms.sum()

def fit_fractions(counts, templates, start=None):
    if start is None:
        start = np.full(4, 0.25)
        
    result = optimize.minimize(
        poisson_deviance,
        x0=np.asarray(start, dtype=float),
        args=(counts, templates),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 4,
        constraints={
            "type": "eq",
            "fun": lambda fractions: fractions.sum() - 1.0,
        },
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    
    if not result.success:
        raise RuntimeError(result.message)
    return result

def parametric_bootstrap(counts, templates, fitted, n_toys=1000, seed=0):
    """Estimate fraction intervals while treating the templates as fixed."""
    rng = np.random.default_rng(seed)
    n_events = int(counts.sum())
    probability = model_probabilities(fitted, templates)
    toy_fits = []
    
    for _ in range(n_toys):
        toy_counts = rng.multinomial(n_events, probability)
        try:
            toy_result = fit_fractions(toy_counts, templates, fitted)
        except RuntimeError:
            continue
        toy_fits.append(toy_result.x)
        
    toy_fits = np.asarray(toy_fits)
    return {
        "median": np.median(toy_fits, axis=0),
        "lower": np.quantile(toy_fits, 0.16, axis=0),
        "upper": np.quantile(toy_fits, 0.84, axis=0),
        "covariance": np.cov(toy_fits, rowvar=False),
    }

def calculate_moments(counts, xmax):
    """Számolja az átlagot, varianciát, ferdeséget (skewness) és lapultságot (kurtosis)."""
    total = np.sum(counts)
    if total == 0:
        return np.nan, np.nan, np.nan, np.nan
        
    mean = np.sum(counts * xmax) / total
    variance = np.sum(counts * (xmax - mean)**2) / total
    std_dev = np.sqrt(variance)
    
    if std_dev > 0:
        skewness = np.sum(counts * (xmax - mean)**3) / (total * std_dev**3)
        # Excess kurtosis (normál eloszlásé 0)
        kurtosis = np.sum(counts * (xmax - mean)**4) / (total * std_dev**4) - 3.0
    else:
        skewness, kurtosis = np.nan, np.nan
        
    return mean, variance, skewness, kurtosis

# ==========================================
# Fő végrehajtó blokk
# ==========================================
if __name__ == "__main__":
    # Kimeneti mappa létrehozása (ha nem létezik, létrehozza)
    output_dir = "output_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Az eredmények a(z) '{output_dir}' mappába lesznek mentve...")

    # Energia-index konvenció: 0 -> 17.5, 1 -> 18.0, 2 -> 18.5, 3 -> 19.0.
    mc_data = monteCarloDataHandler().getMonteCarloData()
    table = aligned_fit_table(
        "xMaxData/XmaxDist_Ebin0.txt",
        mc_data,
        mc_energy_index=1,
    )

    counts, templates = prepare_arrays(table, resolution_sigma=40.0)
    fit = fit_fractions(counts, templates)

    # 1. Illesztés eredményeinek mentése
    fit_results_path = os.path.join(output_dir, "fit_results.txt")
    with open(fit_results_path, "w", encoding="utf-8") as f:
        f.write(f"Success: {fit.success}\n")
        f.write(f"Poisson deviance: {fit.fun:.5f}\n")
        f.write("Fractions:\n")
        for species, fraction in zip(SPECIES, fit.x):
            f.write(f"f_{species} = {fraction:.5f}\n")

    # 2. Skewness és Kurtosis diagnosztika számítása és mentése
    xmax_vals = table["Xmax"].to_numpy(dtype=float)
    exp_counts = expected_counts(fit.x, counts, templates)
    
    _, _, data_skew, data_kurt = calculate_moments(counts, xmax_vals)
    _, _, fit_skew, fit_kurt = calculate_moments(exp_counts, xmax_vals)
    
    moments_path = os.path.join(output_dir, "moments_diagnostics.txt")
    with open(moments_path, "w", encoding="utf-8") as f:
        f.write("--- Post-Fit Moments Diagnostics ---\n")
        f.write(f"Adat Skewness: {data_skew:.4f} | Modell Skewness: {fit_skew:.4f}\n")
        f.write(f"Adat Kurtosis: {data_kurt:.4f} | Modell Kurtosis: {fit_kurt:.4f}\n")

    # 3. Bootstrap bizonytalanság számítása és mentése
    print("Bizonytalanságok számolása (Bootstrap futtatása)...")
    uncertainty = parametric_bootstrap(
        counts,
        templates,
        fitted=fit.x,
        n_toys=1000,
        seed=12345,
    )
    
    uncertainty_path = os.path.join(output_dir, "uncertainties.txt")
    with open(uncertainty_path, "w", encoding="utf-8") as f:
        f.write("--- Uncertainties ---\n")
        for index, species in enumerate(SPECIES):
            f.write(
                f"f_{species} = {fit.x[index]:.4f} "
                f"(-{fit.x[index] - uncertainty['lower'][index]:.4f}, "
                f"+{uncertainty['upper'][index] - fit.x[index]:.4f})\n"
            )
            
    print("Kész! A fájlok sikeresen mentve.")