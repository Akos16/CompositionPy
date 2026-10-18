from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
from monteCarloDataHandler import monteCarloDataHandler


#Constansok a mappákhoz
DATA_DIR = Path("./xMaxData")
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

#lgE
LGES = ["18", "18.5", "19"]

# EXPERIMENTAL ENERGY MAPPING
DATA_ENERGY_INDEX = {"18": 0, "18.5": 1, "19": 2}

# MC ENERGY MAPPING
MC_ENERGY_INDEX = {"18": 1, "18.5": 2, "19": 3}
#componens nevek
COMPONENT_NAMES = ["H", "He", "N", "Fe"]
#504-es Xmax is kell
EXCLUDED_XMAX = set()

# RESOLUTION
RESOLUTION_SIGMA = 20.0
RESOLUTION_SIGMAS = [15.0, 20.0, 25.0]

#Bootstrap beállítások
DO_BOOTSTRAP = True
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 12345

#kezdeti paraméterek a comphoz
PARAMS = np.array([0.25, 0.25, 0.25, 0.25])

#Auger adatok beolvasása
def read_data_file(energy_bin):
    filename = DATA_DIR / f"XmaxDist_Ebin{energy_bin}.txt"
    if not filename.exists():
        raise FileNotFoundError(f"Missing experimental file:\n{filename}")
    df = pd.read_csv(filename, sep="\t", decimal=",")
    x = df["Xmax"].to_numpy(dtype=float)
    counts = df["Counts"].to_numpy(dtype=float)
    return x, counts

#Monte-Carlo adatok beolvasása
def load_mc_templates():
    return monteCarloDataHandler().getMonteCarloData()

#fit range
def select_fit_range(x, counts):
    mask = ~np.isin(x.astype(int), list(EXCLUDED_XMAX))
    return x[mask], counts[mask]

# ALIGN MC
def align_templates(x_data, mc, energy_index, sigma):
    mc_x = np.asarray(mc[4], dtype=float)
    template_matrix = []
    dx = np.median(np.diff(mc_x))
    sigma_bins = sigma / dx if sigma > 0 else 0.0

    for component in range(4):
        frame = mc[component][energy_index]
        frame_x = frame["Xmax"].to_numpy(dtype=float)
        frame_p = frame["Frac"].to_numpy(dtype=float)
        lookup = dict(zip(frame_x, frame_p))
        p = np.array([lookup.get(value, 0.0) for value in x_data], dtype=float)
        # Gaussian smearing
        if sigma > 0:
            p = gaussian_filter1d(p, sigma_bins, mode="constant", cval=0.0)
        
        # Normalize MC template
        total = p.sum()

        if total <= 0:
            raise RuntimeError(f"MC component {COMPONENT_NAMES[component]} has zero support.")

        p /= total
        template_matrix.append(p)

    return np.asarray(template_matrix)

# EXPECTED COUNTS
def poisson_expected(counts, templates, fractions):
    probability = fractions @ templates
    return counts.sum() * probability

# POISSON DEVIANCE
def poisson_deviance(fractions, counts, templates):
    if np.any(fractions < 0):
        return np.inf

    if not np.isclose(fractions.sum(), 1.0, atol=1e-8):
        return np.inf

    expected = poisson_expected(counts, templates, fractions)

    bad = (counts > 0) & (expected <= 0)

    if np.any(bad):
        return np.inf

    positive = counts > 0

    deviance = 2.0 * (expected.sum() - counts.sum() + np.sum(counts[positive] * np.log(counts[positive] / expected[positive])))

    return float(deviance)

#fit
def fit_poisson(counts, templates):
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    constraint = {"type": "eq", "fun": lambda f: np.sum(f) - 1.0}

    result = minimize(poisson_deviance, PARAMS, args=(counts, templates), method="SLSQP", bounds=bounds, constraints=[constraint], options={"ftol": 1e-10, "maxiter": 500})

    if not result.success:
        print("WARNING:", result.message)

    if not np.isfinite(result.fun):
        raise RuntimeError("Poisson fit failed.")

    return result

#momentumok
def moments_from_counts(x, counts):
    total = counts.sum()

    if total <= 0:
        return np.nan, np.nan, np.nan, np.nan

    p = counts / total
    mean = np.sum(x * p)
    centered = x - mean
    variance = np.sum(centered**2 * p)

    if variance <= 0:
        return mean, 0.0, np.nan, np.nan

    mu3 = np.sum(centered**3 * p)
    mu4 = np.sum(centered**4 * p)
    skewness = mu3 / variance**1.5
    excess_kurtosis = mu4 / variance**2 - 3.0

    return mean, variance, skewness, excess_kurtosis

# DEVIANCE RESIDUALS
def poisson_deviance_residuals(counts, expected):
    residuals = np.zeros_like(counts, dtype=float)
    positive = counts > 0

    residuals[positive] = np.sign(counts[positive] - expected[positive]) * np.sqrt(2.0 * (counts[positive] * np.log(counts[positive] / expected[positive]) - (counts[positive] - expected[positive])))

    zero = counts == 0
    residuals[zero] = -np.sqrt(2.0 * expected[zero])

    return residuals

# BOOTSTRAP
def bootstrap_fit(x, counts, templates):
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    fractions_list = []
    moments_list = []
    sigma_list = []

    # Bootstrap loop
    for _ in range(N_BOOTSTRAP):

        # Generate Poisson toy data
        toy_counts = rng.poisson(counts)

        try:
            result = fit_poisson(toy_counts, templates)
        except RuntimeError:
            continue

        # Best-fit fractions
        fractions = result.x

        # Expected counts for toy fit
        expected = poisson_expected(toy_counts, templates, fractions)

        # Moments of fitted model
        moments = moments_from_counts(x, expected)
        variance = moments[1]

        if not np.isfinite(variance) or variance < 0:
            continue

        sigma = np.sqrt(variance)

        fractions_list.append(fractions)
        moments_list.append(moments)
        sigma_list.append(sigma)

    # Convert to arrays
    fractions_array = np.asarray(fractions_list)
    moments_array = np.asarray(moments_list)
    sigma_array = np.asarray(sigma_list)

    if len(fractions_array) < 2:
        raise RuntimeError("Bootstrap failed.")

    #fractions error
    fraction_errors = fractions_array.std(axis=0, ddof=1)

    
    #moments errors, mena, var, skew, kurt
    moment_errors = moments_array.std(axis=0, ddof=1)

    #sigma err
    sigma_error = sigma_array.std(ddof=1)

    return fraction_errors, moment_errors, sigma_error, sigma_array

# ANALYSE ONE ENERGY
def analyse_energy(lgE, mc, sigma, do_bootstrap=False):

    # Experimental file index
    data_index = DATA_ENERGY_INDEX[lgE]

    # MC energy index
    mc_index = MC_ENERGY_INDEX[lgE]

    # Data
    x_raw, counts_raw = read_data_file(data_index)
    x, counts = select_fit_range(x_raw, counts_raw)

    # MC templates
    templates = align_templates(x, mc, mc_index, sigma)

    # Fit
    result = fit_poisson(counts, templates)
    fractions = result.x
    expected = poisson_expected(counts, templates, fractions)

    # Residuals
    residuals = poisson_deviance_residuals(counts, expected)

    # Data moments
    data_moments = moments_from_counts(x, counts)

    # Model moments
    model_moments = moments_from_counts(x, expected)

    # Data sigma
    data_sigma = np.sqrt(data_moments[1])

    # Model sigma
    model_sigma = np.sqrt(model_moments[1])

    # Bootstrap
    if do_bootstrap:
        fraction_errors, moment_errors, sigma_error, sigma_samples = bootstrap_fit(x, counts, templates)
    else:
        fraction_errors = np.full(4, np.nan)
        moment_errors = np.full(4, np.nan)
        sigma_error = np.nan
        sigma_samples = np.array([])

    # Return all information
    return {
        "lgE": lgE,
        "sigma": sigma,
        "x": x,
        "counts": counts,
        "templates": templates,
        "expected": expected,
        "fractions": fractions,
        "fraction_errors": fraction_errors,
        "deviance": result.fun,
        "residuals": residuals,
        "data_moments": np.asarray(data_moments),
        "model_moments": np.asarray(model_moments),
        "data_sigma": data_sigma,
        "model_sigma": model_sigma,
        "model_sigma_error": sigma_error,
        "sigma_samples": sigma_samples,
        "model_moment_errors": moment_errors,
    }

# MAIN
def main():

    print("=" * 80)
    print("FAST POISSON COMPOSITION FIT")
    print("=" * 80)
    
    #mc beolvasás
    mc = load_mc_templates()

    # Nominal fits
    nominal_results = []

    for lgE in LGES:

        print()
        print(f"Fitting lgE = {lgE}")

        result = analyse_energy(lgE, mc, RESOLUTION_SIGMA, do_bootstrap=DO_BOOTSTRAP)
        nominal_results.append(result)

        print()
        print(f"lgE = {lgE}")
        print("  fractions:", np.round(result["fractions"], 5))
        print("  deviance:", f"{result['deviance']:.4f}")
        print("  data mean:", f"{result['data_moments'][0]:.6f}")
        print("  model mean:", f"{result['model_moments'][0]:.6f}", "+/-", f"{result['model_moment_errors'][0]:.6f}" if DO_BOOTSTRAP else "nan")
        print("  data variance:", f"{result['data_moments'][1]:.6f}")
        print("  model variance:", f"{result['model_moments'][1]:.6f}", "+/-", f"{result['model_moment_errors'][1]:.6f}" if DO_BOOTSTRAP else "nan")
        print("  data sigma:", f"{result['data_sigma']:.6f}")
        print("  model sigma:", f"{result['model_sigma']:.6f}")
        print("  model sigma error:", f"{result['model_sigma_error']:.6f}")
        print("  data skewness:", f"{result['data_moments'][2]:.6f}")
        print("  model skewness:", f"{result['model_moments'][2]:.6f}", "+/-", f"{result['model_moment_errors'][2]:.6f}" if DO_BOOTSTRAP else "nan")
        print("  data excess kurtosis:", f"{result['data_moments'][3]:.6f}")
        print("  model excess kurtosis:", f"{result['model_moments'][3]:.6f}", "+/-", f"{result['model_moment_errors'][3]:.6f}" if DO_BOOTSTRAP else "nan")

    #resulotion scan
    scan_rows = []

    for resolution_sigma in RESOLUTION_SIGMAS:

        for lgE in LGES:

            result = analyse_energy(lgE, mc, resolution_sigma, do_bootstrap=False)

            row = {"lgE": lgE, "resolution_sigma": resolution_sigma, "deviance": result["deviance"]}

            #comp fractions
            for name, fraction in zip(COMPONENT_NAMES, result["fractions"]):
                row[f"f_{name}"] = fraction

            #moments
            data_m = result["data_moments"]
            model_m = result["model_moments"]

            row["data_mean"] = data_m[0]
            row["model_mean"] = model_m[0]
            row["data_variance"] = data_m[1]
            row["model_variance"] = model_m[1]
            row["data_sigma"] = result["data_sigma"]
            row["model_sigma"] = result["model_sigma"]
            row["data_skewness"] = data_m[2]
            row["model_skewness"] = model_m[2]
            row["data_excess_kurtosis"] = data_m[3]
            row["model_excess_kurtosis"] = model_m[3]

            scan_rows.append(row)

    scan_df = pd.DataFrame(scan_rows)
    scan_df.to_csv(RESULTS_DIR / "composition_resolution_scan.csv", index=False)

    # NOMINAL RESULTS CSV
    result_rows = []

    for result in nominal_results:

        row = {
            "lgE": result["lgE"],
            "resolution_sigma": result["sigma"],
            "poisson_deviance": result["deviance"],
            "residual_rms": np.sqrt(np.mean(result["residuals"]**2)),
        }

        # Composition fractions
        for name, fraction, error in zip(COMPONENT_NAMES, result["fractions"], result["fraction_errors"]):
            row[f"f_{name}"] = fraction
            row[f"err_f_{name}"] = error

        # Moments
        data_m = result["data_moments"]
        model_m = result["model_moments"]

        row["data_mean"] = data_m[0]
        row["model_mean"] = model_m[0]
        row["data_variance"] = data_m[1]
        row["model_variance"] = model_m[1]
        row["data_sigma"] = result["data_sigma"]
        row["model_sigma"] = result["model_sigma"]
        row["err_model_sigma"] = result["model_sigma_error"]
        row["data_skewness"] = data_m[2]
        row["model_skewness"] = model_m[2]
        row["data_excess_kurtosis"] = data_m[3]
        row["model_excess_kurtosis"] = model_m[3]

        # Bootstrap errors
        moment_errors = result["model_moment_errors"]

        row["err_model_mean"] = moment_errors[0]
        row["err_model_variance"] = moment_errors[1]
        row["err_model_skewness"] = moment_errors[2]
        row["err_model_excess_kurtosis"] = moment_errors[3]

        result_rows.append(row)

    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(RESULTS_DIR / "composition_poisson_results.csv", index=False)

    # SIGMA BOOTSTRAP DISTRIBUTIONS
    sigma_bootstrap_rows = []

    for result in nominal_results:
        for sigma_sample in result["sigma_samples"]:
            sigma_bootstrap_rows.append({"lgE": result["lgE"], "sigma_bootstrap": sigma_sample})

    sigma_bootstrap_df = pd.DataFrame(sigma_bootstrap_rows)
    sigma_bootstrap_df.to_csv(RESULTS_DIR / "sigma_bootstrap_samples.csv", index=False)

    # MOMENTS WITH ERRORS -> TXT
    moments_txt = RESULTS_DIR / "composition_moments_with_errors.txt"

    with open(moments_txt, "w", encoding="utf-8") as f:

        f.write("Xmax moments with bootstrap uncertainties\n")
        f.write("Errors are 1-sigma bootstrap standard deviations.\n")
        f.write("Kurtosis = excess kurtosis (kurtosis - 3).\n")
        f.write("Sigma = sqrt(variance).\n")
        f.write("Sigma error = standard deviation of sqrt(variance) over bootstrap toys.\n")
        f.write("\n")

        for result in nominal_results:

            lgE = result["lgE"]
            data_m = result["data_moments"]
            model_m = result["model_moments"]
            moment_errors = result["model_moment_errors"]

            f.write(f"lgE = {lgE}\n")
            f.write("-" * 70 + "\n")

            # DATA
            f.write("DATA\n")
            f.write(f"mean              = {data_m[0]:.6f}\n")
            f.write(f"variance          = {data_m[1]:.6f}\n")
            f.write(f"sigma             = {result['data_sigma']:.6f}\n")
            f.write(f"skewness          = {data_m[2]:.6f}\n")
            f.write(f"excess kurtosis   = {data_m[3]:.6f}\n")
            f.write("\n")

            # MODEL
            f.write("MODEL\n")
            f.write(f"mean              = {model_m[0]:.6f} +/- {moment_errors[0]:.6f}\n")
            f.write(f"variance          = {model_m[1]:.6f} +/- {moment_errors[1]:.6f}\n")
            f.write(f"sigma             = {result['model_sigma']:.6f} +/- {result['model_sigma_error']:.6f}\n")
            f.write(f"skewness          = {model_m[2]:.6f} +/- {moment_errors[2]:.6f}\n")
            f.write(f"excess kurtosis   = {model_m[3]:.6f} +/- {moment_errors[3]:.6f}\n")
            f.write("\n\n")

    # PRINT FINAL RESULTS
    print()
    print("=" * 80)
    print("NOMINAL RESULTS")
    print("=" * 80)

    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # PLOT 1 - ADATOK ÉS POISSON-ILLESZTÉS
    # A három panel azonos x- és y-tengelyt használ.
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 17,
                         "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "legend.fontsize": 12})
    common_xmin = min(np.min(r["x"]) for r in nominal_results)
    common_xmax = max(np.max(r["x"]) for r in nominal_results)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    for index, result in enumerate(nominal_results):
        ax = axes[index]
        x = result["x"]
        counts = result["counts"]
        expected = result["expected"]

        # Mért adatok: fekete; illesztés: piros.
        data_probability = counts / counts.sum()
        data_error = np.sqrt(counts) / counts.sum()
        ax.errorbar(x, data_probability, yerr=data_error, fmt="o",
                    color="black", ecolor="black", markersize=4,
                    capsize=2)
        model_probability = expected / expected.sum()
        ax.plot(x, model_probability, "-", color="red", linewidth=2.5,
                label="Poisson-fit")

        # Áttekinthető, valódi jelmagyarázat jobb felül, panelenként.
        ndf = len(counts) - 3  # négy arányból három független
        diagnostic = [
            Line2D([], [], color="none", label=f"Deviancia: {result['deviance']:.1f}"),
            Line2D([], [], color="none", label=f"NDF: {ndf}"),
            Line2D([], [], color="none", label=(
                f"Deviancia/NDF: {result['deviance'] / ndf:.2f}"
                if ndf > 0 else "Deviancia/NDF: –")),
        ]
        legend = ax.legend(
            handles=[
                Line2D([], [], marker="o", linestyle="none", color="black",
                       markersize=6, label="Mért adatok"),
                Line2D([], [], color="red", linewidth=2.6,
                       label="Poisson-illesztés"),
                *diagnostic,
            ],
            title=rf"$\lg(E/\mathrm{{eV}}) = {result['lgE']}$",
            loc="upper right", bbox_to_anchor=(0.985, 0.985),
            frameon=True, fancybox=False, framealpha=0.94,
            facecolor="white", edgecolor="0.78",
            fontsize=11.7, title_fontsize=15, handlelength=2.0,
            handletextpad=0.7, borderpad=0.65, labelspacing=0.5,
        )
        legend.get_title().set_fontweight("semibold")
        ax.set_xlabel(r"$X_{\max}$ [g/cm$^2$]")
        ax.set_xlim(common_xmin, common_xmax)

    axes[0].set_ylabel(r"Valószínűség $X_{\max}$-intervallumonként")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0)
    fig.savefig(RESULTS_DIR / "composition_poisson_fit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # PLOT 2 - POISSON-DEVIANCIA ELTÉRÉSEI
    fig_res, axes_res = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    fig_res.patch.set_facecolor("white")

    for index, result in enumerate(nominal_results):
        ax = axes_res[index]
        x = result["x"]
        residuals = result["residuals"]

        ax.axhline(0.0, color="red", linewidth=1.5)
        ax.axhline(2.0, color="gray", linestyle="--", linewidth=1)
        ax.axhline(-2.0, color="gray", linestyle="--", linewidth=1)
        ax.scatter(x, residuals, s=25, color="black")
        ndf = len(result["counts"]) - 3
        diagnostic = [
            Line2D([], [], color="none", label=f"Deviancia: {result['deviance']:.1f}"),
            Line2D([], [], color="none", label=f"NDF: {ndf}"),
            Line2D([], [], color="none", label=(
                f"Deviancia/NDF: {result['deviance'] / ndf:.2f}"
                if ndf > 0 else "Deviancia/NDF: –")),
        ]
        legend = ax.legend(
            handles=[
                Line2D([], [], marker="o", linestyle="none", color="black",
                       markersize=6, label="Eltérések"),
                Line2D([], [], color="red", linewidth=1.8,
                       label="Nullszint"),
                *diagnostic,
            ],
            title=rf"$\lg(E/\mathrm{{eV}}) = {result['lgE']}$",
            loc="upper right", bbox_to_anchor=(0.985, 0.985),
            frameon=True, fancybox=False, framealpha=0.94,
            facecolor="white", edgecolor="0.78",
            fontsize=11.7, title_fontsize=15, handlelength=2.0,
            handletextpad=0.7, borderpad=0.65, labelspacing=0.5,
        )
        legend.get_title().set_fontweight("semibold")
        ax.set_xlabel(r"$X_{\max}$ [g/cm$^2$]")
        ax.set_xlim(common_xmin, common_xmax)
        ax.grid(alpha=0.25)

    axes_res[0].set_ylabel("Előjeles Poisson-deviancia-eltérés")
    # Az eltérések skálája mindhárom panelen közös; a valószínűségi ábráéval
    # nem azonosítható, mert más fizikai mennyiséget és előjelet ábrázol.
    fig_res.tight_layout()
    fig_res.subplots_adjust(wspace=0)
    fig_res.savefig(RESULTS_DIR / "composition_poisson_residuals.png", dpi=300, bbox_inches="tight")
    plt.close(fig_res)

    # FINISHED
    print()
    print("=" * 80)
    print("Analysis finished successfully.")
    print("=" * 80)
    print()
    print("Output directory:")
    print(RESULTS_DIR.resolve())
    print()
    print("Output files:")
    print("  composition_poisson_results.csv")
    print("  composition_resolution_scan.csv")
    print("  composition_moments_with_errors.txt")
    print("  sigma_bootstrap_samples.csv")
    print("  composition_poisson_fit.png")
    print("  composition_poisson_residuals.png")

# RUN
if __name__ == "__main__":
    main()