import os

import numpy as np
import pandas as pd

from scipy import optimize
from scipy.ndimage import gaussian_filter1d

import sys
from datetime import datetime

OUTPUT_FILE = (
    f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)

_original_stdout = sys.stdout

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


output_file = open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
)

sys.stdout = Tee(
    _original_stdout,
    output_file,
)
# ============================================================
# CONFIGURATION
# ============================================================

SPECIES = ("H", "He", "N", "Fe")

DATA_FILE = "xMaxData/XmaxDist_Ebin0.txt"

MC_FOLDER = "./MonteCarloSimulations"

# MC energy index:
# 0 -> 17.5
# 1 -> 18.0
# 2 -> 18.5
# 3 -> 19.0
MC_ENERGY_INDEX = 1

# IMPORTANT:
# The MC is convolved BEFORE this cut is applied.
XMAX_MIN = 504.0

RESOLUTION_SIGMA = 20.0

AUGER_FRACTIONS = np.array([
    0.40,   # H
    0.03,   # He
    0.41,   # N
    0.16,   # Fe
])

BOOTSTRAP_TOYS = 1000
BOOTSTRAP_SEED = 12345


# ============================================================
# BASIC UTILITIES
# ============================================================

def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def check_fractions(fractions, name="fractions"):
    fractions = np.asarray(
        fractions,
        dtype=float,
    )

    if len(fractions) != 4:
        raise ValueError(
            f"{name} must contain four fractions."
        )

    if np.any(~np.isfinite(fractions)):
        raise ValueError(
            f"{name} contains non-finite values."
        )

    if np.any(fractions < 0):
        raise ValueError(
            f"{name} contains negative values."
        )

    if not np.isclose(
        fractions.sum(),
        1.0,
    ):
        raise ValueError(
            f"{name} does not sum to one: "
            f"{fractions.sum()}"
        )


# ============================================================
# READ DATA
# ============================================================

def read_data(data_path):

    data = pd.read_csv(
        data_path,
        sep="\t",
    )

    required = {
        "Xmax",
        "Counts",
    }

    if not required.issubset(
        data.columns
    ):
        raise ValueError(
            f"Data file must contain columns "
            f"{required}. "
            f"Found: {list(data.columns)}"
        )

    data = data[
        ["Xmax", "Counts"]
    ].copy()

    data["Xmax"] = pd.to_numeric(
        data["Xmax"],
        errors="coerce",
    )

    data["Counts"] = pd.to_numeric(
        data["Counts"],
        errors="coerce",
    )

    if data.isna().any().any():
        raise ValueError(
            "Data contain NaN/non-numeric values."
        )

    if np.any(
        data["Counts"] < 0
    ):
        raise ValueError(
            "Data contain negative event counts."
        )

    data = (
        data
        .sort_values("Xmax")
        .reset_index(drop=True)
    )

    return data


# ============================================================
# DATA RANGE DIAGNOSTICS
# ============================================================

def diagnose_data_range(data):

    print_header(
        "DATA RANGE DIAGNOSTICS"
    )

    print(
        f"Raw data Xmax range : "
        f"{data['Xmax'].min():.1f} - "
        f"{data['Xmax'].max():.1f}"
    )

    print(
        f"Raw total events    : "
        f"{int(data['Counts'].sum())}"
    )

    selected = data[
        data["Xmax"] > XMAX_MIN
    ]

    print(
        f"Events with Xmax > "
        f"{XMAX_MIN:.0f}: "
        f"{int(selected['Counts'].sum())}"
    )

    excluded = data[
        data["Xmax"] <= XMAX_MIN
    ]

    print(
        f"Events with Xmax <= "
        f"{XMAX_MIN:.0f}: "
        f"{int(excluded['Counts'].sum())}"
    )

    print()

    high = data[
        data["Xmax"] > 1068
    ]

    high_events = int(
        high["Counts"].sum()
    )

    print(
        "IMPORTANT CHECK:"
    )

    print(
        f"Events above Xmax = 1068: "
        f"{high_events}"
    )

    if high_events > 0:

        print(
            "WARNING:"
        )

        print(
            "The current fit range ends at 1068, "
            "but the data contain positive counts above 1068."
        )

        print(
            "Those events are NOT used by the current likelihood."
        )

        print()

        print(
            "Non-zero bins above 1068:"
        )

        print(
            high[
                high["Counts"] > 0
            ][
                ["Xmax", "Counts"]
            ].to_string(
                index=False
            )
        )


# ============================================================
# READ AND REBIN FULL MC
#
# IMPORTANT:
# Unlike the original monteCarloDataHandler,
# this function starts at 0 so that the complete
# MC distribution is available BEFORE convolution.
# ============================================================

def read_and_rebin_full(
    filename,
    start=0,
    step=12,
):

    df = pd.read_csv(
        filename,
        sep="\t",
        decimal=",",
    )

    new_lines = []

    for col_idx in range(
        0,
        len(df.columns),
        2,
    ):

        if col_idx + 1 >= len(
            df.columns
        ):
            break

        bin_col = df.columns[
            col_idx
        ]

        value_col = df.columns[
            col_idx + 1
        ]

        data = df[
            df[bin_col] >= start
        ].copy()

        if data.empty:
            continue

        max_value = float(
            data[bin_col].max()
        )

        bins = range(
            start,
            int(max_value) + 1,
            step,
        )

        if col_idx == 0:

            for new_bin in bins:

                mask = (
                    (data[bin_col] >= new_bin)
                    &
                    (
                        data[bin_col]
                        < new_bin + step
                    )
                )

                total = data.loc[
                    mask,
                    value_col,
                ].sum()

                new_lines.append([
                    new_bin,
                    total,
                ])

        else:

            for row, new_bin in zip(
                new_lines,
                bins,
            ):

                mask = (
                    (data[bin_col] >= new_bin)
                    &
                    (
                        data[bin_col]
                        < new_bin + step
                    )
                )

                total = data.loc[
                    mask,
                    value_col,
                ].sum()

                row.append(
                    new_bin
                )

                row.append(
                    total
                )

    return new_lines


# ============================================================
# LOAD COMPLETE MC
# ============================================================

def load_full_mc():

    print_header(
        "LOADING FULL MC DISTRIBUTIONS"
    )

    energy_labels = [
        "17,5",
        "18",
        "18,5",
        "19",
    ]

    all_components = []

    for component_index, species in enumerate(
        SPECIES
    ):

        filename = os.path.join(
            MC_FOLDER,
            f"component{component_index}.txt",
        )

        if not os.path.exists(
            filename
        ):
            raise FileNotFoundError(
                f"MC file not found: "
                f"{filename}"
            )

        raw = read_and_rebin_full(
            filename,
            start=0,
            step=12,
        )

        if not raw:
            raise ValueError(
                f"No MC data found in "
                f"{filename}"
            )

        df = pd.DataFrame(
            raw,
            columns=[
                "bin",
                "17,5",
                "bin",
                "18",
                "bin",
                "18,5",
                "bin",
                "19",
            ],
        )

        xmax = df.iloc[
            :,
            0,
        ].to_numpy(
            dtype=float
        )

        component_templates = []

        print()
        print(
            f"{species} "
            f"(component{component_index}.txt)"
        )

        for energy_index, label in enumerate(
            energy_labels
        ):

            values = pd.to_numeric(
                df[label],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

            if np.any(
                ~np.isfinite(values)
            ):
                raise ValueError(
                    f"Non-finite values in "
                    f"{filename}, "
                    f"energy {label}"
                )

            if np.any(
                values < 0
            ):
                raise ValueError(
                    f"Negative MC values in "
                    f"{filename}, "
                    f"energy {label}"
                )

            total = values.sum()

            if total <= 0:
                raise ValueError(
                    f"MC template is empty: "
                    f"{species}, "
                    f"energy {label}"
                )

            # Normalize the COMPLETE MC.
            # No Xmax cut is applied here.
            probability = (
                values / total
            )

            component_templates.append(
                pd.DataFrame({
                    "Xmax": xmax.copy(),
                    "Frac": probability,
                })
            )

            print(
                f"  energy index "
                f"{energy_index}: "
                f"Xmax = "
                f"{xmax.min():.1f} - "
                f"{xmax.max():.1f}, "
                f"sum = "
                f"{probability.sum():.12f}, "
                f"nonzero = "
                f"{np.count_nonzero(probability)}"
            )

        all_components.append(
            component_templates
        )

    return all_components


# ============================================================
# MC AXIS DIAGNOSTICS
# ============================================================

def diagnose_mc_axes(mc_data):

    print_header(
        "MC AXIS / ENERGY DIAGNOSTICS"
    )

    for species_index, species in enumerate(
        SPECIES
    ):

        print()
        print(species)

        for energy_index in range(4):

            component = mc_data[
                species_index
            ][energy_index]

            x = component[
                "Xmax"
            ].to_numpy(
                dtype=float
            )

            p = component[
                "Frac"
            ].to_numpy(
                dtype=float
            )

            print(
                f"  energy index "
                f"{energy_index}: "
                f"Xmax = "
                f"{x.min():.1f} - "
                f"{x.max():.1f}, "
                f"sum = "
                f"{p.sum():.12f}, "
                f"nonzero = "
                f"{np.count_nonzero(p)}"
            )


# ============================================================
# CHECK COMMON MC XMAX AXIS
# ============================================================

def check_mc_axes(mc_data):

    reference = mc_data[
        0
    ][
        0
    ]["Xmax"].to_numpy(
        dtype=float
    )

    print_header(
        "MC COMMON XMAX AXIS TEST"
    )

    for species_index, species in enumerate(
        SPECIES
    ):

        for energy_index in range(4):

            x = mc_data[
                species_index
            ][
                energy_index
            ]["Xmax"].to_numpy(
                dtype=float
            )

            same_length = (
                len(x)
                == len(reference)
            )

            same_values = (
                same_length
                and np.allclose(
                    x,
                    reference,
                )
            )

            print(
                f"{species:>2}, "
                f"energy {energy_index}: "
                f"{'OK' if same_values else 'MISMATCH'}"
            )

            if not same_values:

                if not same_length:

                    print(
                        "    Different number "
                        "of Xmax bins."
                    )

                else:

                    print(
                        "    Xmax coordinates differ."
                    )


# ============================================================
# BIN WIDTH TEST
# ============================================================

def diagnose_bin_width(x):

    print_header(
        "XMAX BIN WIDTH TEST"
    )

    x = np.asarray(
        x,
        dtype=float
    )

    differences = np.diff(
        x
    )

    unique = np.unique(
        np.round(
            differences,
            10,
        )
    )

    print(
        f"Number of bins : "
        f"{len(x)}"
    )

    print(
        f"Unique widths  : "
        f"{unique}"
    )

    if len(unique) == 1:

        print(
            "Uniform Xmax binning: YES"
        )

    else:

        print(
            "Uniform Xmax binning: NO"
        )

    if len(unique) == 1:

        print(
            f"Bin width = "
            f"{unique[0]:.6f} g/cm^2"
        )

        print()

        print(
            "NOTE:"
        )

        print(
            "The Xmax values are currently treated "
            "as bin coordinates."
        )

        print(
            "If these values are LOWER BIN EDGES, "
            "the physical bin centers would be "
            "Xmax + bin_width/2."
        )


# ============================================================
# BUILD CONVOLVED MC
#
# ORDER:
#
# FULL MC
#   ↓
# GAUSSIAN CONVOLUTION
#   ↓
# XMAX CUT
#   ↓
# NORMALIZATION
#
# This is the important change.
# ============================================================

def build_convolved_templates(
    mc_data,
    energy_index,
    resolution_sigma,
    xmax_min,
):

    if resolution_sigma < 0:
        raise ValueError(
            "Resolution sigma cannot be negative."
        )

    reference = mc_data[
        0
    ][
        energy_index
    ]

    xmax_full = reference[
        "Xmax"
    ].to_numpy(
        dtype=float
    )

    full_templates = []

    for species_index, species in enumerate(
        SPECIES
    ):

        component = mc_data[
            species_index
        ][
            energy_index
        ]

        x = component[
            "Xmax"
        ].to_numpy(
            dtype=float
        )

        probability = component[
            "Frac"
        ].to_numpy(
            dtype=float
        )

        if not np.allclose(
            x,
            xmax_full,
        ):
            raise ValueError(
                f"Xmax axis mismatch for "
                f"{species}."
            )

        full_templates.append(
            probability.copy()
        )

    full_templates = np.asarray(
        full_templates,
        dtype=float,
    )

    if len(xmax_full) < 2:
        raise ValueError(
            "Not enough MC bins."
        )

    widths = np.diff(
        xmax_full
    )

    if not np.allclose(
        widths,
        np.median(widths),
    ):
        raise ValueError(
            "MC Xmax bins are not uniform."
        )

    bin_width = np.median(
        widths
    )

    # --------------------------------------------------------
    # FULL MC -> CONVOLUTION
    # --------------------------------------------------------

    if resolution_sigma > 0:

        sigma_bins = (
            resolution_sigma
            / bin_width
        )

        convolved = gaussian_filter1d(
            full_templates,
            sigma=sigma_bins,
            axis=1,
            mode="constant",
            cval=0.0,
        )

    else:

        convolved = (
            full_templates.copy()
        )

    # --------------------------------------------------------
    # CONVOLUTION -> XMAX CUT
    # --------------------------------------------------------

    cut_mask = (
        xmax_full > xmax_min
    )

    xmax_cut = (
        xmax_full[cut_mask]
    )

    cut_templates = (
        convolved[
            :,
            cut_mask,
        ]
    )

    # --------------------------------------------------------
    # CUT -> NORMALIZATION
    # --------------------------------------------------------

    row_sums = cut_templates.sum(
        axis=1,
        keepdims=True,
    )

    if np.any(
        row_sums <= 0
    ):
        raise ValueError(
            "At least one MC template has "
            "zero support after convolution "
            "and Xmax cut."
        )

    templates = (
        cut_templates
        / row_sums
    )

    print_header(
        "MC CONVOLUTION -> CUT -> NORMALIZATION"
    )

    print(
        f"Energy index       : "
        f"{energy_index}"
    )

    print(
        f"Resolution sigma   : "
        f"{resolution_sigma:.3f} g/cm^2"
    )

    print(
        f"Full MC Xmax range : "
        f"{xmax_full.min():.1f} - "
        f"{xmax_full.max():.1f}"
    )

    print(
        f"After Xmax cut     : "
        f"{xmax_cut.min():.1f} - "
        f"{xmax_cut.max():.1f}"
    )

    print()

    print(
        "Probability retained after Xmax cut:"
    )

    for species_index, species in enumerate(
        SPECIES
    ):

        print(
            f"  {species}: "
            f"{row_sums[species_index, 0]:.12f}"
        )

    print()

    print(
        "Final normalized template sums:"
    )

    for species_index, species in enumerate(
        SPECIES
    ):

        print(
            f"  {species}: "
            f"{templates[species_index].sum():.12f}"
        )

    return (
        xmax_cut,
        templates,
    )


# ============================================================
# ALIGN DATA WITH MC
# ============================================================

def align_data_with_mc(
    data,
    xmax_mc,
):

    selected_data = data[
        data["Xmax"] > XMAX_MIN
    ].copy()

    if selected_data.empty:
        raise ValueError(
            "No data remain after Xmax cut."
        )

    data_xmax = selected_data[
        "Xmax"
    ].to_numpy(
        dtype=float
    )

    common_xmax = np.intersect1d(
        data_xmax,
        xmax_mc,
    )

    if len(common_xmax) == 0:
        raise ValueError(
            "No common Xmax bins between "
            "data and MC."
        )

    table = selected_data[
        selected_data["Xmax"].isin(
            common_xmax
        )
    ].copy()

    table = (
        table
        .sort_values("Xmax")
        .reset_index(drop=True)
    )

    positive_data = selected_data[
        selected_data["Counts"] > 0
    ]

    unsupported = (
        ~positive_data["Xmax"].isin(
            xmax_mc
        )
    )

    if np.any(unsupported):

        bad = positive_data.loc[
            unsupported,
            "Xmax",
        ].to_list()

        raise ValueError(
            "Positive data counts exist in Xmax bins "
            "without MC coverage: "
            f"{bad}"
        )

    return table


# ============================================================
# TRIM MC TEMPLATES TO DATA RANGE
#
# The convolution has already been performed on
# the FULL MC.
#
# This function only makes the final likelihood
# arrays have exactly the same Xmax bins as the data.
# ============================================================

def trim_templates_to_data(
    table,
    xmax_mc,
    templates,
):

    data_xmax = table[
        "Xmax"
    ].to_numpy(
        dtype=float
    )

    template_mask = np.isin(
        xmax_mc,
        data_xmax,
    )

    if not np.all(
        np.isin(
            data_xmax,
            xmax_mc,
        )
    ):
        missing = data_xmax[
            ~np.isin(
                data_xmax,
                xmax_mc,
            )
        ]

        raise ValueError(
            "Some data Xmax bins are missing "
            "from the MC axis: "
            f"{missing}"
        )

    trimmed_templates = (
        templates[
            :,
            template_mask,
        ]
    )

    trimmed_xmax = xmax_mc[
        template_mask
    ]

    if not np.allclose(
        trimmed_xmax,
        data_xmax,
    ):
        raise ValueError(
            "Data and MC Xmax axes do not match "
            "after trimming."
        )

    return (
        data_xmax,
        trimmed_templates,
    )


# ============================================================
# PREPARE ARRAYS
# ============================================================

def prepare_arrays(
    table,
    templates,
):

    counts = table[
        "Counts"
    ].to_numpy(
        dtype=float
    )

    if np.any(
        ~np.isfinite(counts)
    ):
        raise ValueError(
            "Counts contain non-finite values."
        )

    if np.any(
        counts < 0
    ):
        raise ValueError(
            "Counts contain negative values."
        )

    if templates.shape[1] != len(
        counts
    ):
        raise ValueError(
            "Template/data dimensions do not match."
        )

    if np.any(
        ~np.isfinite(templates)
    ):
        raise ValueError(
            "Templates contain non-finite values."
        )

    if np.any(
        templates < 0
    ):
        raise ValueError(
            "Templates contain negative values."
        )

    row_sums = templates.sum(
        axis=1
    )

    if np.any(
        row_sums <= 0
    ):
        raise ValueError(
            "At least one MC template has "
            "no support."
        )

    templates = (
        templates
        / row_sums[:, None]
    )

    model_support = templates.sum(
        axis=0
    )

    unsupported = (
        (counts > 0)
        & (model_support <= 0)
    )

    if np.any(
        unsupported
    ):

        bad_indices = np.where(
            unsupported
        )[0]

        bad_xmax = table.iloc[
            bad_indices
        ]["Xmax"].to_list()

        raise ValueError(
            "Positive data counts occur in bins "
            "with zero MC support: "
            f"{bad_xmax}"
        )

    return (
        counts,
        templates,
    )


# ============================================================
# MODEL
# ============================================================

def model_probabilities(
    fractions,
    templates,
):

    fractions = np.asarray(
        fractions,
        dtype=float,
    )

    if np.any(
        ~np.isfinite(fractions)
    ):
        return np.full(
            templates.shape[1],
            np.nan,
        )

    probability = (
        fractions @ templates
    )

    total = probability.sum()

    if (
        not np.isfinite(total)
        or total <= 0
    ):
        return np.full(
            probability.shape,
            np.nan,
        )

    return (
        probability / total
    )


def expected_counts(
    fractions,
    counts,
    templates,
):

    probability = model_probabilities(
        fractions,
        templates,
    )

    return (
        counts.sum()
        * probability
    )


# ============================================================
# POISSON DEVIANCE
# ============================================================

def poisson_deviance(
    fractions,
    counts,
    templates,
):

    fractions = np.asarray(
        fractions,
        dtype=float,
    )

    if np.any(
        fractions < 0
    ) or np.any(
        fractions > 1
    ):
        return np.inf

    if not np.isfinite(
        fractions.sum()
    ):
        return np.inf

    if not np.isclose(
        fractions.sum(),
        1.0,
        atol=1e-7,
    ):
        return np.inf

    expected = expected_counts(
        fractions,
        counts,
        templates,
    )

    if np.any(
        ~np.isfinite(expected)
    ):
        return np.inf

    unsupported = (
        (counts > 0)
        & (expected <= 0)
    )

    if np.any(unsupported):
        return np.inf

    terms = (
        expected - counts
    )

    positive = counts > 0

    terms[positive] += (
        counts[positive]
        * np.log(
            counts[positive]
            / expected[positive]
        )
    )

    result = (
        2.0
        * terms.sum()
    )

    if not np.isfinite(
        result
    ):
        return np.inf

    return result


# ============================================================
# FIT
# ============================================================

def fit_fractions(
    counts,
    templates,
    start=None,
):

    if start is None:

        start = np.full(
            4,
            0.25,
        )

    start = np.asarray(
        start,
        dtype=float,
    )

    if not np.isclose(
        start.sum(),
        1.0,
    ):
        raise ValueError(
            "Starting fractions must sum to one."
        )

    result = optimize.minimize(
        poisson_deviance,
        x0=start,
        args=(
            counts,
            templates,
        ),
        method="SLSQP",
        bounds=[
            (0.0, 1.0)
        ] * 4,
        constraints={
            "type": "eq",
            "fun": lambda f:
                f.sum() - 1.0,
        },
        options={
            "ftol": 1e-9,
            "maxiter": 3000,
            "disp": False,
        },
    )

    if not result.success:
        raise RuntimeError(
            f"Optimization failed: "
            f"{result.message}"
        )

    return result


# ============================================================
# MOMENTS
# ============================================================

def mixture_moments(
    fractions,
    templates,
    xmax,
):

    probability = (
        np.asarray(
            fractions,
            dtype=float,
        )
        @ templates
    )

    probability = (
        probability
        / probability.sum()
    )

    mean = np.sum(
        xmax
        * probability
    )

    centered = (
        xmax - mean
    )

    mu2 = np.sum(
        centered**2
        * probability
    )

    mu3 = np.sum(
        centered**3
        * probability
    )

    mu4 = np.sum(
        centered**4
        * probability
    )

    sigma = np.sqrt(
        max(
            mu2,
            0.0,
        )
    )

    if sigma > 0:

        skewness = (
            mu3
            / sigma**3
        )

        kurtosis = (
            mu4
            / sigma**4
        )

        excess_kurtosis = (
            kurtosis
            - 3.0
        )

    else:

        skewness = np.nan
        kurtosis = np.nan
        excess_kurtosis = np.nan

    return (
        mean,
        sigma,
        skewness,
        kurtosis,
        excess_kurtosis,
    )


# ============================================================
# MULTI-START
# ============================================================

def multi_start_test(
    counts,
    templates,
):

    print_header(
        "MULTI-START STABILITY TEST"
    )

    starts = [
        [0.25, 0.25, 0.25, 0.25],
        [0.70, 0.10, 0.10, 0.10],
        [0.10, 0.70, 0.10, 0.10],
        [0.10, 0.10, 0.70, 0.10],
        [0.10, 0.10, 0.10, 0.70],
        [0.40, 0.03, 0.41, 0.16],
    ]

    solutions = []

    for start in starts:

        try:

            result = fit_fractions(
                counts,
                templates,
                start,
            )

            print(
                f"start={np.array(start)}"
            )

            print(
                f"  deviance = "
                f"{result.fun:.8f}"
            )

            print(
                "  fractions = "
                + " ".join(
                    f"{x:.6f}"
                    for x in result.x
                )
            )

            solutions.append(
                result.x
            )

        except Exception as exc:

            print(
                f"start={start}"
            )

            print(
                f"  ERROR: {exc}"
            )

    if len(solutions) > 1:

        solutions = np.asarray(
            solutions
        )

        variation = (
            solutions.max(axis=0)
            - solutions.min(axis=0)
        )

        print()

        print(
            "Maximum fraction variation:"
        )

        for species, value in zip(
            SPECIES,
            variation,
        ):

            print(
                f"  {species}: "
                f"{value:.3e}"
            )


# ============================================================
# TEMPLATE CORRELATION
# ============================================================

def template_correlation_test(
    templates,
):

    print_header(
        "MC TEMPLATE OVERLAP / CORRELATION TEST"
    )

    correlation = np.corrcoef(
        templates
    )

    print(
        pd.DataFrame(
            correlation,
            index=SPECIES,
            columns=SPECIES,
        ).to_string(
            float_format=lambda x:
                f"{x:.5f}"
        )
    )

    print()

    print(
        "Large positive correlations indicate "
        "similar template shapes and weaker "
        "composition separation."
    )


# ============================================================
# OBSERVED VS EXPECTED
# ============================================================

def observed_expected_test(
    table,
    counts,
    templates,
    fractions,
):

    print_header(
        "OBSERVED VS EXPECTED TEST"
    )

    expected = expected_counts(
        fractions,
        counts,
        templates,
    )

    print(
        f"{'Xmax':>8}"
        f"{'Observed':>12}"
        f"{'Expected':>12}"
        f"{'Pull':>12}"
    )

    print(
        "-" * 48
    )

    for x, observed, exp in zip(
        table["Xmax"],
        counts,
        expected,
    ):

        if exp > 0:

            pull = (
                (observed - exp)
                / np.sqrt(exp)
            )

        else:

            pull = np.nan

        # Only print interesting deviations
        if (
            abs(pull) >= 2.0
            or observed > 0
        ):

            print(
                f"{x:8.1f}"
                f"{observed:12.1f}"
                f"{exp:12.3f}"
                f"{pull:12.3f}"
            )


# ============================================================
# MC TEMPLATE MOMENTS
# ============================================================

def mc_template_diagnostics(
    xmax,
    templates,
):

    print_header(
        "CONVOLVED + CUT MC TEMPLATE DIAGNOSTICS"
    )

    for index, species in enumerate(
        SPECIES
    ):

        p = templates[
            index
        ]

        mean = np.sum(
            xmax * p
        )

        centered = (
            xmax - mean
        )

        mu2 = np.sum(
            centered**2
            * p
        )

        mu3 = np.sum(
            centered**3
            * p
        )

        mu4 = np.sum(
            centered**4
            * p
        )

        sigma = np.sqrt(
            max(
                mu2,
                0.0,
            )
        )

        if sigma > 0:

            skewness = (
                mu3
                / sigma**3
            )

            kurtosis = (
                mu4
                / sigma**4
            )

            excess_kurtosis = (
                kurtosis
                - 3.0
            )

        else:

            skewness = np.nan
            kurtosis = np.nan
            excess_kurtosis = np.nan

        print()
        print(species)
        print("-" * 40)

        print(
            f"Xmax range : "
            f"{xmax.min():.1f} - "
            f"{xmax.max():.1f}"
        )

        print(
            f"sum        : "
            f"{p.sum():.12f}"
        )

        print(
            f"mean       : "
            f"{mean:.3f}"
        )

        print(
            f"sigma      : "
            f"{sigma:.3f}"
        )

        print(
            f"skewness   : "
            f"{skewness:.6f}"
        )

        print(
            f"kurtosis   : "
            f"{kurtosis:.6f}"
        )

        print(
            f"excess kurtosis: "
            f"{excess_kurtosis:.6f}"
        )

        print(
            f"nonzero bins: "
            f"{np.count_nonzero(p)}"
        )


# ============================================================
# NORMALIZATION / CONVOLUTION ORDER TEST
# ============================================================

def normalization_order_test_full_mc(
    mc_data,
    energy_index,
):

    print_header(
        "FULL MC CONVOLUTION / NORMALIZATION ORDER TEST"
    )

    reference = mc_data[
        0
    ][
        energy_index
    ]

    xmax = reference[
        "Xmax"
    ].to_numpy(
        dtype=float
    )

    raw_templates = []

    for species_index in range(
        len(SPECIES)
    ):

        raw_templates.append(
            mc_data[
                species_index
            ][
                energy_index
            ]["Frac"].to_numpy(
                dtype=float
            )
        )

    raw_templates = np.asarray(
        raw_templates,
        dtype=float,
    )

    widths = np.diff(
        xmax
    )

    bin_width = np.median(
        widths
    )

    sigma_bins = (
        RESOLUTION_SIGMA
        / bin_width
    )

    # A:
    # normalize -> convolve -> cut -> normalize

    A = (
        raw_templates
        / raw_templates.sum(
            axis=1,
            keepdims=True,
        )
    )

    A = gaussian_filter1d(
        A,
        sigma=sigma_bins,
        axis=1,
        mode="constant",
        cval=0.0,
    )

    A = A[
        :,
        xmax > XMAX_MIN,
    ]

    A = (
        A
        / A.sum(
            axis=1,
            keepdims=True,
        )
    )

    # B:
    # convolve -> cut -> normalize

    B = gaussian_filter1d(
        raw_templates,
        sigma=sigma_bins,
        axis=1,
        mode="constant",
        cval=0.0,
    )

    B = B[
        :,
        xmax > XMAX_MIN,
    ]

    B = (
        B
        / B.sum(
            axis=1,
            keepdims=True,
        )
    )

    difference = np.max(
        np.abs(
            A - B
        )
    )

    print(
        "Maximum absolute difference:"
    )

    print(
        f"{difference:.6e}"
    )

    if difference < 1e-12:

        print(
            "The two normalization orders are "
            "numerically identical."
        )

    else:

        print(
            "The two normalization orders differ."
        )


# ============================================================
# RESOLUTION SCAN
# ============================================================

def resolution_scan(
    data,
    mc_data,
):

    print_header(
        "RESOLUTION SCAN"
    )

    results = []

    resolution_values = [
        10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39,
        40,
    ]

    for sigma in resolution_values:

        print()
        print(
            f"Testing sigma = "
            f"{sigma:.1f} g/cm^2"
        )

        try:

            xmax_mc, templates = (
                build_convolved_templates(
                    mc_data,
                    MC_ENERGY_INDEX,
                    sigma,
                    XMAX_MIN,
                )
            )

            table = align_data_with_mc(
                data,
                xmax_mc,
            )

            xmax, templates = (
                trim_templates_to_data(
                    table,
                    xmax_mc,
                    templates,
                )
            )

            counts, templates = (
                prepare_arrays(
                    table,
                    templates,
                )
            )

            fit = fit_fractions(
                counts,
                templates,
            )

            (
                mean,
                std,
                skewness,
                kurtosis,
                excess_kurtosis,
            ) = mixture_moments(
                fit.x,
                templates,
                xmax,
            )

            print(
                f"  deviance = "
                f"{fit.fun:.6f}"
            )

            print(
                f"  H  = "
                f"{fit.x[0]:.6f}"
            )

            print(
                f"  He = "
                f"{fit.x[1]:.6f}"
            )

            print(
                f"  N  = "
                f"{fit.x[2]:.6f}"
            )

            print(
                f"  Fe = "
                f"{fit.x[3]:.6f}"
            )

            results.append({
                "resolution": sigma,
                "deviance": fit.fun,
                "H": fit.x[0],
                "He": fit.x[1],
                "N": fit.x[2],
                "Fe": fit.x[3],
                "mean": mean,
                "sigma_xmax": std,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "excess_kurtosis": excess_kurtosis,
                "status": "OK",
            })

        except Exception as exc:

            print(
                "  STATUS: INVALID"
            )

            print(
                f"  Reason: {exc}"
            )

            results.append({
                "resolution": sigma,
                "deviance": np.nan,
                "H": np.nan,
                "He": np.nan,
                "N": np.nan,
                "Fe": np.nan,
                "mean": np.nan,
                "sigma_xmax": np.nan,
                "skewness": np.nan,
                "kurtosis": np.nan,
                "excess_kurtosis": np.nan,
                "status": "ERROR",
            })

    result_df = pd.DataFrame(
        results
    )

    print_header(
        "RESOLUTION SUMMARY"
    )

    print(
        result_df.to_string(
            index=False
        )
    )

    valid = result_df[
        np.isfinite(
            result_df["deviance"]
        )
    ]

    if not valid.empty:

        best = valid.loc[
            valid["deviance"].idxmin()
        ]

        print()

        print(
            "Lowest-deviance scanned resolution:"
        )

        print(
            f"  sigma = "
            f"{best['resolution']:.1f} g/cm^2"
        )

        print(
            f"  deviance = "
            f"{best['deviance']:.6f}"
        )

        print()

        print(
            "IMPORTANT:"
        )

        print(
            "This is only a sensitivity diagnostic."
        )

        print(
            "It does NOT establish the physical detector "
            "resolution."
        )

    return result_df


# ============================================================
# BOOTSTRAP
# ============================================================

def parametric_bootstrap(
    counts,
    templates,
    fitted,
    n_toys=1000,
    seed=0,
):

    rng = np.random.default_rng(
        seed
    )

    n_events = int(
        counts.sum()
    )

    probability = model_probabilities(
        fitted,
        templates,
    )

    toy_fits = []

    for _ in range(
        n_toys
    ):

        toy_counts = rng.multinomial(
            n_events,
            probability,
        )

        try:

            result = fit_fractions(
                toy_counts,
                templates,
                fitted,
            )

            toy_fits.append(
                result.x
            )

        except Exception:
            continue

    toy_fits = np.asarray(
        toy_fits
    )

    if len(toy_fits) == 0:
        raise RuntimeError(
            "All bootstrap fits failed."
        )

    return toy_fits


def print_bootstrap(
    toy_fits,
    fitted,
):

    print_header(
        "BOOTSTRAP"
    )

    print(
        f"Successful toys: "
        f"{len(toy_fits)}"
    )

    for i, species in enumerate(
        SPECIES
    ):

        median = np.median(
            toy_fits[:, i]
        )

        lower = np.quantile(
            toy_fits[:, i],
            0.16,
        )

        upper = np.quantile(
            toy_fits[:, i],
            0.84,
        )

        print(
            f"{species}: "
            f"median={median:.6f}, "
            f"16%={lower:.6f}, "
            f"84%={upper:.6f}"
        )

        print(
            f"      fit = "
            f"{fitted[i]:.6f} "
            f"(-{fitted[i] - lower:.6f}, "
            f"+{upper - fitted[i]:.6f})"
        )


# ============================================================
# MAIN
# ============================================================

print_header(
    "POISSON XMAX COMPOSITION ANALYSIS"
)

print(
    f"Fit cut         : "
    f"Xmax > {XMAX_MIN:.0f}"
)

print(
    f"Resolution      : "
    f"{RESOLUTION_SIGMA:.1f} g/cm^2"
)

print(
    f"Energy index    : "
    f"{MC_ENERGY_INDEX}"
)

print()

print(
    "IMPORTANT MODEL ORDER:"
)

print(
    "FULL MC"
)

print(
    "  -> Gaussian convolution"
)

print(
    "  -> Xmax cut"
)

print(
    "  -> normalization"
)

print(
    "  -> data-range trimming"
)

print(
    "  -> Poisson deviance fit"
)


# ============================================================
# LOAD DATA
# ============================================================

data = read_data(
    DATA_FILE
)

diagnose_data_range(
    data
)


# ============================================================
# LOAD COMPLETE MC
# ============================================================

mc_data = load_full_mc()

diagnose_mc_axes(
    mc_data
)

check_mc_axes(
    mc_data
)


# ============================================================
# BUILD MAIN CONVOLVED TEMPLATE
# ============================================================

xmax_mc, templates = (
    build_convolved_templates(
        mc_data,
        MC_ENERGY_INDEX,
        RESOLUTION_SIGMA,
        XMAX_MIN,
    )
)


# ============================================================
# ALIGN DATA AND MC
# ============================================================

table = align_data_with_mc(
    data,
    xmax_mc,
)

# ------------------------------------------------------------
# IMPORTANT:
# Convolution and Xmax cut have ALREADY happened.
#
# Now we only trim the template to the exact data bins
# used by the likelihood.
# ------------------------------------------------------------

xmax, templates = (
    trim_templates_to_data(
        table,
        xmax_mc,
        templates,
    )
)


# ============================================================
# BIN WIDTH DIAGNOSTICS
# ============================================================

diagnose_bin_width(
    xmax
)


# ============================================================
# PREPARE ARRAYS
# ============================================================

counts, templates = prepare_arrays(
    table,
    templates,
)


print_header(
    "COMMON DATA / MC RANGE"
)

print(
    f"Common Xmax range: "
    f"{xmax.min():.1f} - "
    f"{xmax.max():.1f}"
)

print(
    f"Number of bins: "
    f"{len(xmax)}"
)

print(
    f"Events in fit: "
    f"{int(counts.sum())}"
)


# ============================================================
# MAIN FIT
# ============================================================

fit = fit_fractions(
    counts,
    templates,
)


print_header(
    "MAIN FIT"
)

print(
    f"Success   : "
    f"{fit.success}"
)

print(
    f"Deviance  : "
    f"{fit.fun:.10f}"
)

for species, fraction in zip(
    SPECIES,
    fit.x,
):

    print(
        f"f_{species} = "
        f"{fraction:.8f}"
    )

print(
    f"sum = "
    f"{fit.x.sum():.12f}"
)


# ============================================================
# MIXTURE MOMENTS
# ============================================================

(
    fit_mean,
    fit_sigma,
    fit_skew,
    fit_kurt,
    fit_excess_kurt,
) = mixture_moments(
    fit.x,
    templates,
    xmax,
)

(
    auger_mean,
    auger_sigma,
    auger_skew,
    auger_kurt,
    auger_excess_kurt,
) = mixture_moments(
    AUGER_FRACTIONS,
    templates,
    xmax,
)


print_header(
    "MIXTURE MOMENTS"
)

print(
    "FIT"
)

print(
    f"  mean            = "
    f"{fit_mean:.3f}"
)

print(
    f"  sigma           = "
    f"{fit_sigma:.3f}"
)

print(
    f"  skewness        = "
    f"{fit_skew:.6f}"
)

print(
    f"  kurtosis        = "
    f"{fit_kurt:.6f}"
)

print(
    f"  excess kurtosis = "
    f"{fit_excess_kurt:.6f}"
)

print()

print(
    "AUGER REFERENCE"
)

print(
    f"  mean            = "
    f"{auger_mean:.3f}"
)

print(
    f"  sigma           = "
    f"{auger_sigma:.3f}"
)

print(
    f"  skewness        = "
    f"{auger_skew:.6f}"
)

print(
    f"  kurtosis        = "
    f"{auger_kurt:.6f}"
)

print(
    f"  excess kurtosis = "
    f"{auger_excess_kurt:.6f}"
)

print()

print(
    "Definitions:"
)

print(
    "  skewness = mu3 / sigma^3"
)

print(
    "  kurtosis = mu4 / sigma^4"
)

print(
    "  excess kurtosis = kurtosis - 3"
)

print(
    "For a Gaussian distribution, "
    "excess kurtosis = 0."
)


# ============================================================
# AUGER REFERENCE TEST
# ============================================================

fit_deviance = poisson_deviance(
    fit.x,
    counts,
    templates,
)

auger_deviance = poisson_deviance(
    AUGER_FRACTIONS,
    counts,
    templates,
)


print_header(
    "AUGER REFERENCE TEST"
)

print(
    f"Fit deviance   : "
    f"{fit_deviance:.10f}"
)

print(
    f"Auger deviance : "
    f"{auger_deviance:.10f}"
)

print(
    f"Delta D        : "
    f"{auger_deviance - fit_deviance:.10f}"
)

for species, f_fit, f_auger in zip(
    SPECIES,
    fit.x,
    AUGER_FRACTIONS,
):

    print(
        f"{species}: "
        f"fit={f_fit:.6f}, "
        f"Auger={f_auger:.6f}, "
        f"difference="
        f"{f_fit - f_auger:+.6f}"
    )


# ============================================================
# MC TEMPLATE DIAGNOSTICS
# ============================================================

mc_template_diagnostics(
    xmax,
    templates,
)


# ============================================================
# MULTI-START
# ============================================================

multi_start_test(
    counts,
    templates,
)


# ============================================================
# TEMPLATE CORRELATION
# ============================================================

template_correlation_test(
    templates,
)


# ============================================================
# FRACTION BOUNDARY TEST
# ============================================================

print_header(
    "FRACTION BOUNDARY TEST"
)

for species, fraction in zip(
    SPECIES,
    fit.x,
):

    print(
        f"{species}: "
        f"{fraction:.8f}"
    )

print()

print(
    "A value close to 0 or 1 indicates "
    "that the optimum is close to a "
    "physical boundary."
)


# ============================================================
# OBSERVED VS EXPECTED
# ============================================================

observed_expected_test(
    table,
    counts,
    templates,
    fit.x,
)


# ============================================================
# NORMALIZATION ORDER TEST
# ============================================================

normalization_order_test_full_mc(
    mc_data,
    MC_ENERGY_INDEX,
)


# ============================================================
# RESOLUTION SCAN
# ============================================================

resolution_scan(
    data,
    mc_data,
)


# ============================================================
# BOOTSTRAP
# ============================================================

toy_fits = parametric_bootstrap(
    counts,
    templates,
    fit.x,
    n_toys=BOOTSTRAP_TOYS,
    seed=BOOTSTRAP_SEED,
)

print_bootstrap(
    toy_fits,
    fit.x,
)


# ============================================================
# FINAL
# ============================================================

print_header(
    "FINAL"
)

for species, fraction in zip(
    SPECIES,
    fit.x,
):

    print(
        f"f_{species}  = "
        f"{fraction:.6f}"
    )

print()

print(
    f"Mean            = "
    f"{fit_mean:.6f}"
)

print(
    f"Sigma           = "
    f"{fit_sigma:.6f}"
)

print(
    f"Skewness        = "
    f"{fit_skew:.6f}"
)

print(
    f"Kurtosis        = "
    f"{fit_kurt:.6f}"
)

print(
    f"Excess kurtosis = "
    f"{fit_excess_kurt:.6f}"
)

print()

print(
    f"Events used by fit: "
    f"{int(counts.sum())}"
)

print(
    f"Xmax used: "
    f"{xmax.min():.1f} - "
    f"{xmax.max():.1f}"
)

print()

raw_events = int(
    data["Counts"].sum()
)

fit_events = int(
    counts.sum()
)

print(
    "MOST IMPORTANT DIAGNOSTIC:"
)

print(
    f"Raw events             : "
    f"{raw_events}"
)

print(
    f"Events used in fit     : "
    f"{fit_events}"
)

print(
    f"Events not used        : "
    f"{raw_events - fit_events}"
)

if raw_events != fit_events:

    print()

    print(
        "NOTE:"
    )

    print(
        "The difference is caused by the explicit "
        "Xmax cut and/or the common data/MC range."
    )

    print(
        "The MC convolution itself is performed "
        "BEFORE the cut."
    )

print()

print(
    "MODEL ORDER USED:"
)

print(
    "  full MC"
)

print(
    "  -> convolution"
)

print(
    "  -> Xmax > 504 cut"
)

print(
    "  -> normalization"
)

print(
    "  -> trim to data Xmax range"
)

print(
    "  -> Poisson deviance fit"
)

print()

print(
    "=" * 80
)

print(
    "ANALYSIS COMPLETE"
)

print(
    "=" * 80
)