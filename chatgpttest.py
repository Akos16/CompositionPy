
"""
Raw-count Poisson composition fit for Xmax distributions.

Features
--------
- Experimental data are used as RAW COUNTS.
- MC templates are normalized to probabilities.
- Poisson likelihood / deviance fit.
- Fractions constrained to:
      f_H + f_He + f_N + f_Fe = 1
- Explicit exclusion of the problematic Xmax = 504 bin.
- Gaussian smearing of MC templates.
- Fits for lgE = 18, 18.5 and 19.
- Mean, variance, skewness and excess kurtosis.
- Parametric Poisson bootstrap uncertainties.
- Resolution scan for sigma = 15, 20 and 25 g/cm^2.
- Multi-start optimization check.
- Poisson deviance residuals.
- CSV output of nominal results and resolution scan.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

from monteCarloDataHandler import monteCarloDataHandler


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("./xMaxData")
MC_DIR = Path("./MonteCarloSimulations")
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# Energies to be fitted
LGES = ["18", "18.5", "19"]

# ---------------------------------------------------------------------------
# Experimental-data energy mapping
#
# IMPORTANT:
# The experimental data contain only:
#
#   XmaxDist_Ebin0.txt -> lgE = 18
#   XmaxDist_Ebin1.txt -> lgE = 18.5
#   XmaxDist_Ebin2.txt -> lgE = 19
#
# Do NOT confuse this with the MC energy index.
# ---------------------------------------------------------------------------

DATA_ENERGY_INDEX = {
    "18": 0,
    "18.5": 1,
    "19": 2,
}


# ---------------------------------------------------------------------------
# Monte-Carlo energy mapping
#
# In monteCarloDataHandler:
#
#   0 -> lgE = 17.5
#   1 -> lgE = 18
#   2 -> lgE = 18.5
#   3 -> lgE = 19
# ---------------------------------------------------------------------------

MC_ENERGY_INDEX = {
    "18": 1,
    "18.5": 2,
    "19": 3,
}


COMPONENT_NAMES = [
    "H",
    "He",
    "N",
    "Fe"
]


# ============================================================================
# XMAX FIT RANGE
# ============================================================================

# The Xmax = 504 g/cm^2 bin is explicitly excluded.
#
# This avoids the known support mismatch where experimental data contain
# events but the unsmeared MC templates have no support.
#
# IMPORTANT:
# We do NOT introduce an artificial 1e-10 probability floor.
#
# If you later want to include 504 after implementing a physically justified
# detector response, simply use:
#
#     EXCLUDED_XMAX = set()
#

EXCLUDED_XMAX = {
    504
}


# ============================================================================
# DETECTOR RESOLUTION
# ============================================================================

# Nominal detector resolution
RESOLUTION_SIGMA = 20.0  # g/cm^2


# Resolution sensitivity scan
RESOLUTION_SIGMAS = [
    15.0,
    20.0,
    25.0
]


# IMPORTANT:
#
# If the MC templates are already detector-level reconstructed distributions,
# then do NOT smear them again.
#
# In that case use:
#
#     RESOLUTION_SIGMA = 0.0
#     RESOLUTION_SIGMAS = [0.0]
#
# Otherwise the Gaussian smearing above is applied.


# ============================================================================
# BOOTSTRAP
# ============================================================================

N_BOOTSTRAP = 1000

BOOTSTRAP_SEED = 12345


# ============================================================================
# MINIMIZER STARTING POINTS
# ============================================================================

STARTS = [

    np.array([
        0.25,
        0.25,
        0.25,
        0.25
    ]),

    np.array([
        0.10,
        0.25,
        0.25,
        0.40
    ]),

    np.array([
        0.40,
        0.10,
        0.40,
        0.10
    ]),

    np.array([
        0.05,
        0.05,
        0.80,
        0.10
    ]),
]


# ============================================================================
# READ EXPERIMENTAL DATA
# ============================================================================

def read_data_file(energy_bin):
    """
    Read experimental Xmax distribution.

    Experimental files:

        Ebin0 -> lgE = 18
        Ebin1 -> lgE = 18.5
        Ebin2 -> lgE = 19

    Counts are kept as RAW COUNTS.
    """

    filename = (
        DATA_DIR /
        f"XmaxDist_Ebin{energy_bin}.txt"
    )

    if not filename.exists():

        raise FileNotFoundError(
            f"\nExperimental data file does not exist:\n"
            f"  {filename}\n\n"
            f"Expected files are:\n"
            f"  {DATA_DIR / 'XmaxDist_Ebin0.txt'}\n"
            f"  {DATA_DIR / 'XmaxDist_Ebin1.txt'}\n"
            f"  {DATA_DIR / 'XmaxDist_Ebin2.txt'}\n"
        )

    df = pd.read_csv(
        filename,
        sep="\t",
        decimal=","
    )

    required_columns = {
        "Xmax",
        "Counts"
    }

    missing = (
        required_columns -
        set(df.columns)
    )

    if missing:

        raise ValueError(
            f"{filename}: missing columns "
            f"{sorted(missing)}"
        )

    x = df[
        "Xmax"
    ].to_numpy(
        dtype=float
    )

    counts = df[
        "Counts"
    ].to_numpy(
        dtype=float
    )

    if np.any(counts < 0):

        raise ValueError(
            f"{filename}: negative event counts found."
        )

    return x, counts


# ============================================================================
# LOAD MC
# ============================================================================

def load_mc_templates():

    mc = (
        monteCarloDataHandler()
        .getMonteCarloData()
    )

    return mc


# ============================================================================
# SELECT FIT RANGE
# ============================================================================

def select_fit_range(
    x,
    counts
):

    x = np.asarray(
        x,
        dtype=float
    )

    counts = np.asarray(
        counts,
        dtype=float
    )

    mask = ~np.isin(
        x.astype(int),
        list(EXCLUDED_XMAX)
    )

    return (
        x[mask],
        counts[mask]
    )


# ============================================================================
# ALIGN MC TEMPLATES
# ============================================================================

def align_templates(
    x_data,
    mc,
    energy_index,
    sigma
):
    """
    Construct a 4 x N matrix:

        H
        He
        N
        Fe

    Each row is a normalized probability distribution.

    The MC templates are matched to the experimental Xmax bins by Xmax
    value rather than by array position.
    """

    x_data = np.asarray(
        x_data,
        dtype=float
    )

    mc_x = np.asarray(
        mc[4],
        dtype=float
    )

    template_matrix = []


    # ------------------------------------------------------------------------
    # H, He, N, Fe
    # ------------------------------------------------------------------------

    for component in range(4):

        frame = mc[
            component
        ][
            energy_index
        ]

        frame_x = frame[
            "Xmax"
        ].to_numpy(
            dtype=float
        )

        frame_p = frame[
            "Frac"
        ].to_numpy(
            dtype=float
        )

        # Match Xmax values explicitly.
        lookup = dict(
            zip(
                frame_x,
                frame_p
            )
        )

        p = np.array(
            [
                lookup.get(
                    x,
                    0.0
                )
                for x in x_data
            ],
            dtype=float
        )


        # --------------------------------------------------------------------
        # Gaussian detector resolution
        # --------------------------------------------------------------------

        if sigma > 0:

            dx = np.median(
                np.diff(mc_x)
            )

            if (
                not np.isfinite(dx)
                or dx <= 0
            ):

                raise ValueError(
                    "Could not determine "
                    "MC Xmax bin width."
                )

            sigma_bins = (
                sigma /
                dx
            )

            p = gaussian_filter1d(
                p,
                sigma_bins,
                mode="constant",
                cval=0.0
            )


        # --------------------------------------------------------------------
        # Normalize template
        # --------------------------------------------------------------------

        total = p.sum()

        if total <= 0:

            raise ValueError(
                f"MC component "
                f"{COMPONENT_NAMES[component]} "
                f"has zero probability."
            )

        p /= total

        template_matrix.append(
            p
        )


    return np.asarray(
        template_matrix,
        dtype=float
    )


# ============================================================================
# MOMENTS
# ============================================================================

def moments_from_counts(
    x,
    counts
):
    """
    Calculate histogram moments:

        mean
        variance
        skewness
        excess kurtosis

    The kurtosis returned here is EXCESS kurtosis.

    Gaussian distribution -> excess kurtosis = 0.
    """

    x = np.asarray(
        x,
        dtype=float
    )

    counts = np.asarray(
        counts,
        dtype=float
    )

    total = counts.sum()

    if total <= 0:

        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan
        )

    p = (
        counts /
        total
    )

    mean = np.sum(
        x * p
    )

    centered = (
        x -
        mean
    )

    mu2 = np.sum(
        centered**2 *
        p
    )

    if mu2 <= 0:

        return (
            mean,
            0.0,
            np.nan,
            np.nan
        )

    mu3 = np.sum(
        centered**3 *
        p
    )

    mu4 = np.sum(
        centered**4 *
        p
    )

    skewness = (
        mu3 /
        mu2**1.5
    )

    excess_kurtosis = (
        mu4 /
        mu2**2
        - 3.0
    )

    return (
        mean,
        mu2,
        skewness,
        excess_kurtosis
    )


# ============================================================================
# EXPECTED COUNTS
# ============================================================================

def poisson_expected(
    counts,
    templates,
    fractions
):
    """
    Expected counts:

        mu_i =
            N_data *
            sum_k f_k p_ki
    """

    fractions = np.asarray(
        fractions,
        dtype=float
    )

    probability = (
        fractions @
        templates
    )

    expected = (
        counts.sum() *
        probability
    )

    return expected


# ============================================================================
# POISSON DEVIANCE
# ============================================================================

def poisson_deviance(
    fractions,
    counts,
    templates
):
    """
    Poisson deviance:

        D = 2 sum_i [
            mu_i - n_i
            + n_i ln(n_i / mu_i)
        ]

    for n_i > 0.

    No artificial probability floor is used.
    """

    fractions = np.asarray(
        fractions,
        dtype=float
    )


    # ------------------------------------------------------------------------
    # Physical fraction constraints
    # ------------------------------------------------------------------------

    if np.any(
        fractions < 0
    ):

        return np.inf

    if not np.isclose(
        fractions.sum(),
        1.0,
        atol=1e-8
    ):

        return np.inf


    # ------------------------------------------------------------------------
    # Expected counts
    # ------------------------------------------------------------------------

    expected = poisson_expected(
        counts,
        templates,
        fractions
    )


    # ------------------------------------------------------------------------
    # Zero-support check
    # ------------------------------------------------------------------------

    unsupported = (
        (counts > 0) &
        (expected <= 0)
    )

    if np.any(
        unsupported
    ):

        return np.inf


    # ------------------------------------------------------------------------
    # Deviance
    # ------------------------------------------------------------------------

    positive = (
        counts > 0
    )

    deviance = 2.0 * (
        expected.sum()
        - counts.sum()
        +
        np.sum(
            counts[positive]
            *
            np.log(
                counts[positive] /
                expected[positive]
            )
        )
    )

    return float(
        deviance
    )


# ============================================================================
# POISSON DEVIANCE RESIDUALS
# ============================================================================

def poisson_deviance_residuals(
    counts,
    expected
):
    """
    Signed Poisson deviance residuals.

    For n > 0:

        r =
            sign(n-mu) *
            sqrt(
                2 * [
                    n ln(n/mu)
                    - (n-mu)
                ]
            )

    For n = 0:

        r = -sqrt(2 mu)
    """

    counts = np.asarray(
        counts,
        dtype=float
    )

    expected = np.asarray(
        expected,
        dtype=float
    )

    residuals = np.zeros_like(
        counts,
        dtype=float
    )


    # ------------------------------------------------------------------------
    # n > 0
    # ------------------------------------------------------------------------

    positive = (
        counts > 0
    )

    residuals[
        positive
    ] = (
        np.sign(
            counts[positive]
            -
            expected[positive]
        )
        *
        np.sqrt(
            2.0 *
            (
                counts[positive]
                *
                np.log(
                    counts[positive] /
                    expected[positive]
                )
                -
                (
                    counts[positive]
                    -
                    expected[positive]
                )
            )
        )
    )


    # ------------------------------------------------------------------------
    # n = 0
    # ------------------------------------------------------------------------

    zero = (
        counts == 0
    )

    residuals[
        zero
    ] = -np.sqrt(
        2.0 *
        expected[zero]
    )


    return residuals


# ============================================================================
# POISSON FIT
# ============================================================================

def fit_poisson(
    counts,
    templates
):
    """
    Constrained multi-start SLSQP fit.
    """

    bounds = [
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
    ]

    constraint = {
        "type": "eq",
        "fun": lambda f:
            np.sum(f) - 1.0
    }

    results = []


    for start in STARTS:

        result = minimize(
            poisson_deviance,
            start,
            args=(
                counts,
                templates
            ),
            method="SLSQP",
            bounds=bounds,
            constraints=[
                constraint
            ],
            options={
                "ftol": 1e-12,
                "maxiter": 2000
            }
        )

        if (
            np.isfinite(
                result.fun
            )
        ):

            results.append(
                result
            )


    if not results:

        raise RuntimeError(
            "No finite Poisson solution found.\n"
            "Check MC support, Xmax exclusion "
            "and resolution."
        )


    best = min(
        results,
        key=lambda r: r.fun
    )

    return (
        best,
        results
    )


# ============================================================================
# BOOTSTRAP
# ============================================================================

def bootstrap_fit(
    x,
    counts,
    templates,
    n_bootstrap=N_BOOTSTRAP,
    seed=BOOTSTRAP_SEED
):
    """
    Parametric Poisson bootstrap.

    Each toy data set is generated according to:

        n_i^toy ~ Poisson(n_i)

    and the composition fit is repeated.
    """

    rng = np.random.default_rng(
        seed
    )

    fitted_fractions = []

    fitted_moments = []


    for _ in range(
        n_bootstrap
    ):

        toy_counts = rng.poisson(
            np.clip(
                counts,
                0,
                None
            )
        )


        try:

            result, _ = fit_poisson(
                toy_counts,
                templates
            )

        except RuntimeError:

            continue


        fractions = (
            result.x
        )


        expected = poisson_expected(
            toy_counts,
            templates,
            fractions
        )


        moments = moments_from_counts(
            x,
            expected
        )


        fitted_fractions.append(
            fractions
        )

        fitted_moments.append(
            moments
        )


    fitted_fractions = np.asarray(
        fitted_fractions
    )

    fitted_moments = np.asarray(
        fitted_moments
    )


    if len(
        fitted_fractions
    ) < 2:

        raise RuntimeError(
            "Not enough successful "
            "bootstrap fits."
        )


    fraction_mean = (
        fitted_fractions.mean(
            axis=0
        )
    )

    fraction_error = (
        fitted_fractions.std(
            axis=0,
            ddof=1
        )
    )


    moment_mean = (
        fitted_moments.mean(
            axis=0
        )
    )

    moment_error = (
        fitted_moments.std(
            axis=0,
            ddof=1
        )
    )


    return (
        fraction_mean,
        fraction_error,
        moment_mean,
        moment_error,
        fitted_fractions
    )


# ============================================================================
# ANALYSE ONE ENERGY
# ============================================================================

def analyse_energy(
    lgE,
    mc,
    sigma,
    do_bootstrap=False,
    seed=0
):

    # ------------------------------------------------------------------------
    # Experimental data
    # ------------------------------------------------------------------------

    data_index = (
        DATA_ENERGY_INDEX[lgE]
    )

    mc_index = (
        MC_ENERGY_INDEX[lgE]
    )


    x_raw, counts_raw = read_data_file(
        data_index
    )


    # ------------------------------------------------------------------------
    # Remove problematic Xmax bin
    # ------------------------------------------------------------------------

    x, counts = select_fit_range(
        x_raw,
        counts_raw
    )


    # ------------------------------------------------------------------------
    # MC templates
    # ------------------------------------------------------------------------

    templates = align_templates(
        x,
        mc,
        mc_index,
        sigma
    )


    # ------------------------------------------------------------------------
    # Explicit support check
    # ------------------------------------------------------------------------

    model_support = (
        templates.sum(
            axis=0
        ) > 0
    )

    unsupported = (
        (counts > 0)
        &
        ~model_support
    )


    if np.any(
        unsupported
    ):

        bad_x = (
            x[unsupported]
            .tolist()
        )

        raise RuntimeError(
            f"lgE={lgE}, "
            f"sigma={sigma}: "
            f"non-zero data remain in "
            f"zero-support bins: "
            f"{bad_x}"
        )


    # ------------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------------

    result, all_results = fit_poisson(
        counts,
        templates
    )

    fractions = (
        result.x
    )


    expected = poisson_expected(
        counts,
        templates,
        fractions
    )


    # ------------------------------------------------------------------------
    # Deviance residuals
    # ------------------------------------------------------------------------

    residuals = poisson_deviance_residuals(
        counts,
        expected
    )


    # ------------------------------------------------------------------------
    # Moments
    # ------------------------------------------------------------------------

    data_moments = np.asarray(
        moments_from_counts(
            x,
            counts
        )
    )

    model_moments = np.asarray(
        moments_from_counts(
            x,
            expected
        )
    )


    # ------------------------------------------------------------------------
    # Result dictionary
    # ------------------------------------------------------------------------

    result_dict = {

        "lgE":
            lgE,

        "sigma":
            sigma,

        "x":
            x,

        "counts":
            counts,

        "templates":
            templates,

        "expected":
            expected,

        "fractions":
            fractions,

        "deviance":
            result.fun,

        "residuals":
            residuals,

        "success":
            result.success,

        "message":
            result.message,

        "data_moments":
            data_moments,

        "model_moments":
            model_moments,

        "starts":
            all_results,
    }


    # ------------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------------

    if do_bootstrap:

        (
            _,
            fraction_errors,
            _,
            moment_errors,
            bootstrap_fractions
        ) = bootstrap_fit(
            x,
            counts,
            templates,
            n_bootstrap=N_BOOTSTRAP,
            seed=seed
        )


        result_dict[
            "fraction_errors"
        ] = fraction_errors

        result_dict[
            "model_moment_errors"
        ] = moment_errors

        result_dict[
            "bootstrap_fractions"
        ] = bootstrap_fractions

    else:

        result_dict[
            "fraction_errors"
        ] = np.full(
            4,
            np.nan
        )

        result_dict[
            "model_moment_errors"
        ] = np.full(
            4,
            np.nan
        )


    return result_dict


# ============================================================================
# PRINT RESULTS
# ============================================================================

def print_result(
    result
):

    print()
    print(
        "=" * 80
    )

    print(
        f"lgE = {result['lgE']}    "
        f"sigma = "
        f"{result['sigma']:.1f} "
        f"g/cm^2"
    )

    print(
        "=" * 80
    )


    # ------------------------------------------------------------------------
    # Fit information
    # ------------------------------------------------------------------------

    print(
        f"Poisson deviance = "
        f"{result['deviance']:.6f}"
    )

    print(
        f"Optimizer success = "
        f"{result['success']}"
    )

    print(
        f"Message = "
        f"{result['message']}"
    )


    # ------------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------------

    print()
    print(
        "Composition:"
    )


    for (
        name,
        fraction,
        error
    ) in zip(
        COMPONENT_NAMES,
        result["fractions"],
        result["fraction_errors"]
    ):

        print(
            f"  f_{name:>2s} = "
            f"{fraction:.6f} "
            f"+/- "
            f"{error:.6f}"
        )


    # ------------------------------------------------------------------------
    # Moments
    # ------------------------------------------------------------------------

    print()
    print(
        "Moments:"
    )


    labels = [
        "Mean",
        "Variance",
        "Skewness",
        "Excess kurtosis"
    ]


    for (
        label,
        data_value,
        model_value,
        error
    ) in zip(
        labels,
        result["data_moments"],
        result["model_moments"],
        result["model_moment_errors"]
    ):

        print(
            f"  {label:>18s}: "
            f"data = "
            f"{data_value:.6f}, "
            f"model = "
            f"{model_value:.6f}, "
            f"bootstrap error = "
            f"{error:.6f}"
        )


    # ------------------------------------------------------------------------
    # Deviance residual information
    # ------------------------------------------------------------------------

    residuals = (
        result["residuals"]
    )

    print()
    print(
        "Deviance residuals:"
    )

    print(
        f"  min = "
        f"{np.min(residuals):.4f}"
    )

    print(
        f"  max = "
        f"{np.max(residuals):.4f}"
    )

    print(
        f"  RMS = "
        f"{np.sqrt(np.mean(residuals**2)):.4f}"
    )


    # ------------------------------------------------------------------------
    # Multi-start
    # ------------------------------------------------------------------------

    print()
    print(
        "Multi-start solutions:"
    )


    for r in result[
        "starts"
    ]:

        print(
            "  fractions = ",
            np.round(
                r.x,
                6
            ),
            "  deviance = ",
            r.fun,
            " success = ",
            r.success
        )


# ============================================================================
# MAIN
# ============================================================================

def main():

    # ------------------------------------------------------------------------
    # Load MC
    # ------------------------------------------------------------------------

    mc = load_mc_templates()


    # =========================================================================
    # NOMINAL FITS
    # =========================================================================

    nominal_results = []


    for i, lgE in enumerate(
        LGES
    ):

        print()
        print(
            f"Running Poisson fit "
            f"for lgE = {lgE}"
        )


        result = analyse_energy(
            lgE,
            mc,
            RESOLUTION_SIGMA,
            do_bootstrap=True,
            seed=(
                BOOTSTRAP_SEED +
                i
            )
        )


        nominal_results.append(
            result
        )


        print_result(
            result
        )


    # =========================================================================
    # RESOLUTION SCAN
    # =========================================================================

    print()
    print(
        "#" * 80
    )

    print(
        "RESOLUTION SCAN"
    )

    print(
        "#" * 80
    )


    scan_rows = []


    for sigma in (
        RESOLUTION_SIGMAS
    ):

        for lgE in LGES:

            result = analyse_energy(
                lgE,
                mc,
                sigma,
                do_bootstrap=False
            )


            row = {

                "lgE":
                    lgE,

                "sigma":
                    sigma,

                "deviance":
                    result[
                        "deviance"
                    ],
            }


            # ---------------------------------------------------------------
            # Composition
            # ---------------------------------------------------------------

            for (
                name,
                fraction
            ) in zip(
                COMPONENT_NAMES,
                result[
                    "fractions"
                ]
            ):

                row[
                    f"f_{name}"
                ] = fraction


            # ---------------------------------------------------------------
            # Moments
            # ---------------------------------------------------------------

            data_m = (
                result[
                    "data_moments"
                ]
            )

            model_m = (
                result[
                    "model_moments"
                ]
            )


            row[
                "data_mean"
            ] = data_m[0]

            row[
                "model_mean"
            ] = model_m[0]

            row[
                "data_variance"
            ] = data_m[1]

            row[
                "model_variance"
            ] = model_m[1]

            row[
                "data_skewness"
            ] = data_m[2]

            row[
                "model_skewness"
            ] = model_m[2]

            row[
                "data_excess_kurtosis"
            ] = data_m[3]

            row[
                "model_excess_kurtosis"
            ] = model_m[3]


            scan_rows.append(
                row
            )


    scan_df = pd.DataFrame(
        scan_rows
    )


    print()

    print(
        scan_df.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.6f}"
        )
    )


    scan_df.to_csv(
    RESULTS_DIR / "composition_resolution_scan.csv",
    index=False
    )


    # =========================================================================
    # NOMINAL RESULTS CSV
    # =========================================================================

    result_rows = []


    for result in (
        nominal_results
    ):

        row = {

            "lgE":
                result["lgE"],

            "sigma":
                result["sigma"],

            "poisson_deviance":
                result["deviance"],

            "residual_rms":
                np.sqrt(
                    np.mean(
                        result[
                            "residuals"
                        ] ** 2
                    )
                ),
        }


        # ---------------------------------------------------------------------
        # Composition
        # ---------------------------------------------------------------------

        for (
            name,
            fraction,
            error
        ) in zip(
            COMPONENT_NAMES,
            result["fractions"],
            result["fraction_errors"]
        ):

            row[
                f"f_{name}"
            ] = fraction

            row[
                f"err_f_{name}"
            ] = error


        # ---------------------------------------------------------------------
        # Moments
        # ---------------------------------------------------------------------

        data_moments = (
            result[
                "data_moments"
            ]
        )

        model_moments = (
            result[
                "model_moments"
            ]
        )


        row[
            "data_mean"
        ] = data_moments[0]

        row[
            "model_mean"
        ] = model_moments[0]


        row[
            "data_variance"
        ] = data_moments[1]

        row[
            "model_variance"
        ] = model_moments[1]


        row[
            "data_skewness"
        ] = data_moments[2]

        row[
            "model_skewness"
        ] = model_moments[2]


        row[
            "data_excess_kurtosis"
        ] = data_moments[3]

        row[
            "model_excess_kurtosis"
        ] = model_moments[3]


        result_rows.append(
            row
        )


    result_df = pd.DataFrame(
        result_rows
    )


    result_df.to_csv(
    RESULTS_DIR / "composition_poisson_results.csv",
    index=False
    )


    # =========================================================================
    # PRINT NOMINAL TABLE
    # =========================================================================

    print()
    print(
        "#" * 80
    )

    print(
        "NOMINAL RESULTS"
    )

    print(
        "#" * 80
    )


    print(
        result_df.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.6f}"
        )
    )


    # =========================================================================
    # PLOT 1:
    # DATA VS POISSON MODEL
    # =========================================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
        sharey=True
    )


    for index, result in enumerate(
        nominal_results
    ):

        ax = axes[index]


        x = result[
            "x"
        ]

        counts = result[
            "counts"
        ]

        expected = result[
            "expected"
        ]


        # ---------------------------------------------------------------------
        # Data
        # ---------------------------------------------------------------------

        data_probability = (
            counts /
            counts.sum()
        )

        data_error = (
            np.sqrt(counts) /
            counts.sum()
        )


        ax.errorbar(
            x,
            data_probability,
            yerr=data_error,
            fmt="o",
            markersize=3,
            capsize=2,
            label="Data"
        )


        # ---------------------------------------------------------------------
        # Model
        # ---------------------------------------------------------------------

        model_probability = (
            expected /
            expected.sum()
        )


        ax.plot(
            x,
            model_probability,
            "-",
            linewidth=2,
            label="Poisson MC mixture"
        )


        # ---------------------------------------------------------------------
        # Composition text
        # ---------------------------------------------------------------------

        f = result[
            "fractions"
        ]

        e = result[
            "fraction_errors"
        ]


        composition_text = "\n".join(
            [
                f"$f_{{{name}}}$ = "
                f"{fraction:.3f} "
                f"$\\pm$ "
                f"{error:.3f}"
                for (
                    name,
                    fraction,
                    error
                )
                in zip(
                    COMPONENT_NAMES,
                    f,
                    e
                )
            ]
        )


        ax.text(
            0.03,
            0.96,
            composition_text,
            transform=ax.transAxes,
            va="top",
            fontsize=10
        )


        # ---------------------------------------------------------------------
        # Moments text
        # ---------------------------------------------------------------------

        data_m = result[
            "data_moments"
        ]

        model_m = result[
            "model_moments"
        ]


        moment_text = (

            f"mean: "
            f"{data_m[0]:.1f} / "
            f"{model_m[0]:.1f}\n"

            f"variance: "
            f"{data_m[1]:.1f} / "
            f"{model_m[1]:.1f}\n"

            f"skewness: "
            f"{data_m[2]:.3f} / "
            f"{model_m[2]:.3f}\n"

            f"excess kurtosis: "
            f"{data_m[3]:.3f} / "
            f"{model_m[3]:.3f}"
        )


        ax.text(
            0.03,
            0.55,
            moment_text,
            transform=ax.transAxes,
            fontsize=9
        )


        # ---------------------------------------------------------------------
        # Plot formatting
        # ---------------------------------------------------------------------

        ax.set_title(
            f"lgE = {result['lgE']}"
        )

        ax.set_xlabel(
            r"$X_{\max}$ [g/cm$^2$]"
        )

        ax.legend(
            loc="upper right"
        )


    axes[0].set_ylabel(
        "Probability per Xmax bin"
    )


    fig.suptitle(
        "Raw-count Poisson composition fit\n"
        f"sigma = "
        f"{RESOLUTION_SIGMA:.0f} "
        f"g/cm², "
        "Xmax = 504 excluded",
        fontsize=14
    )


    fig.tight_layout()


    # -------------------------------------------------------------------------
    # Save figure
    # -------------------------------------------------------------------------

    fig.savefig(
        RESULTS_DIR / "composition_poisson_fit.png",
        dpi=300,
        bbox_inches="tight"
    )


    # =========================================================================
    # PLOT 2:
    # POISSON DEVIANCE RESIDUALS
    # =========================================================================

    fig_res, axes_res = plt.subplots(
        1,
        3,
        figsize=(18, 5),
        sharey=True
    )


    for index, result in enumerate(
        nominal_results
    ):

        ax = axes_res[index]


        x = result[
            "x"
        ]

        residuals = result[
            "residuals"
        ]


        ax.axhline(
            0.0,
            linewidth=1
        )


        ax.axhline(
            2.0,
            linestyle="--",
            linewidth=1
        )


        ax.axhline(
            -2.0,
            linestyle="--",
            linewidth=1
        )


        ax.scatter(
            x,
            residuals,
            s=20
        )


        ax.set_title(
            f"lgE = {result['lgE']}"
        )


        ax.set_xlabel(
            r"$X_{\max}$ [g/cm$^2$]"
        )


        ax.grid(
            alpha=0.25
        )


    axes_res[0].set_ylabel(
        "Signed Poisson deviance residual"
    )


    fig_res.suptitle(
        "Poisson deviance residuals"
    )


    fig_res.tight_layout()


    fig_res.savefig(
        RESULTS_DIR / "composition_poisson_residuals.png",
        dpi=300,
        bbox_inches="tight"
    )


    # =========================================================================
    # FINAL
    # =========================================================================

    print()
print("=" * 80)
print("Analysis finished successfully.")
print("=" * 80)

print()
print("Output directory:")
print(f"  {RESULTS_DIR.resolve()}")

print()
print("Output files:")
print("  composition_poisson_results.csv")
print("  composition_resolution_scan.csv")
print("  composition_poisson_fit.png")
print("  composition_poisson_residuals.png")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()

