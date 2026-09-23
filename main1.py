import numpy as np
import pandas as pd
from scipy import optimize
from scipy.ndimage import gaussian_filter1d
from monteCarloDataHandler import monteCarloDataHandler

SPECIES = ("H", "He", "N", "Fe")

def aligned_fit_table(data_path, mc_data, mc_energy_index):
    """Join data and MC by their numerical Xmax coordinate."""
    data = pd.read_csv(data_path, sep="\t")[["Xmax", "Counts"]]
    table = data.copy()
    for species_index, species_name in enumerate(SPECIES):
        component = mc_data[species_index][mc_energy_index][
            ["Xmax", "Frac"]
        ].copy()
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
    
    # --- VÉGSŐ JAVÍTÁS: Egy pici 'padlót' adunk a sablonoknak, hogy sose legyen üres bin ---
    templates = np.clip(templates, 1e-10, None)
    templates = templates / templates.sum(axis=1, keepdims=True)
    # ----------------------------------------------------------------------------------------

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
    # A biztonság kedvéért magát a várható értéket is védjük a nullától
    expected = expected_counts(fractions, counts, templates)
    expected = np.clip(expected, 1e-10, None)
    
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
        options={"ftol": 1e-6, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return result

def parametric_bootstrap(counts, templates, fitted, n_toys=1000, seed=0):
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

# Energy-index convention in getMonteCarloData:
# 0 -> 17.5, 1 -> 18.0, 2 -> 18.5, 3 -> 19.0.
mc_data = monteCarloDataHandler().getMonteCarloData()
table = aligned_fit_table(
    "xMaxData/XmaxDist_Ebin0.txt",
    mc_data,
    mc_energy_index=1,  # <--- Itt most az 1-es indexet (lgE=18.0) használjuk!
)

counts, templates = prepare_arrays(table, resolution_sigma=20.0)
fit = fit_fractions(counts, templates)

print("success:", fit.success)
print("Poisson deviance:", fit.fun)

print("\n--- Végeredmény (hibahatárokkal) ---")
uncertainty = parametric_bootstrap(
    counts,
    templates,
    fitted=fit.x,
    n_toys=1000, 
    seed=12345,
)

for index, species in enumerate(SPECIES):
    print(
        f"f_{species} = {fit.x[index]:.4f} "
        f"(-{fit.x[index] - uncertainty['lower'][index]:.4f}, "
        f"+{uncertainty['upper'][index] - fit.x[index]:.4f})"
    )