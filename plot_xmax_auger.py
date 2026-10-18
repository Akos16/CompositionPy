#!/usr/bin/env python3
"""Plot Auger Xmax counts and composition-weighted simulated templates at lgE=18.

Place this script alongside:
  XmaxDist_Ebin2.txt       measured Xmax, counts, uncertainty (12 g/cm2 bins)
  XmaxDistributions.txt   blocks labeled 'file: i, energy: j'; 1 g/cm2 MC bins

Run: python plot_xmax_auger.py
Options: python plot_xmax_auger.py --show --output myfigure.pdf
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PARTICLES = ('H', 'He', 'N', 'Fe')
FRACTIONS = np.array([0.360932, 0.054860, 0.419665, 0.164542])

#FRACTIONS = np.array([0.11857485448710794, 0.4877060313479121, 0.17410845354310223, 0.21961066062187773])
ENERGY_INDEX = 1  # simulation energy indices 0,1,2,3 -> lgE 17.5,18.0,18.5,19.0
COLORS = {'H': '#d62728', 'He': '#17becf', 'N': '#777777', 'Fe': '#3041ae'}


def load_simulations(filename, energy_index):
    """Read only the requested energy from the multi-block MC text file."""
    block_header = re.compile(r'file:\s*(\d+)\s*,\s*energy:\s*(\d+)')
    blocks = {}
    current = None
    with open(filename, encoding='utf-8') as stream:
        for line in stream:
            m = block_header.search(line)
            if m:
                p_index, e_index = map(int, m.groups())
                current = p_index if e_index == energy_index else None
                if current is not None:
                    if current in blocks:
                        raise ValueError(f'Duplicate template block: particle {current}, energy {energy_index}')
                    blocks[current] = []
                continue
            if current is None or line.lstrip().startswith('#') or not line.strip():
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f'Bad MC row: {line.rstrip()}')
            blocks[current].append([float(fields[0]), float(fields[1])])

    if set(blocks) != set(range(4)):
        raise ValueError(f'Expected MC blocks 0,1,2,3 at energy {energy_index}; got {sorted(blocks)}')
    result = {}
    for i, primary in enumerate(PARTICLES):
        arr = np.asarray(blocks[i])
        if not np.all(np.isfinite(arr)) or np.any(arr[:, 1] < 0):
            raise ValueError(f'Invalid MC counts for {primary}')
        if not np.all(np.diff(arr[:, 0]) > 0):
            raise ValueError(f'MC Xmax grid is not sorted for {primary}')
        result[primary] = (arr[:, 0], arr[:, 1])
    return result


def get_edges(centers):
    """Build bin edges from uniformly spaced data bin centres."""
    spacing = np.diff(centers)
    if not np.allclose(spacing, spacing[0]):
        raise ValueError('Experimental bins are not uniformly spaced')
    delta = spacing[0]
    return np.r_[centers - delta / 2, centers[-1] + delta / 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument('--data', type=Path, default=root / 'xMaxData' / 'XmaxDist_Ebin0.txt')
    parser.add_argument('--mc', type=Path, default=root / 'MonteCarloSimulations' /'XmaxDistributions.txt')
    parser.add_argument('--output', type=Path, default=root / 'Xmax_lgE18.png')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    data = np.loadtxt(args.data, skiprows=1)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError('Data must have three columns: Xmax, counts, uncertainty')
    x, observed, observed_err = data.T
    if (not np.all(np.isfinite(data)) or np.any(observed < 0)
            or np.any(observed_err < 0) or not np.all(np.diff(x) > 0)):
        raise ValueError('Invalid data values')

    edges = get_edges(x)
    total = observed.sum()
    fractions = FRACTIONS / FRACTIONS.sum()  # correct 1e-6 rounding discrepancy
    mc = load_simulations(args.mc, ENERGY_INDEX)

    components = {}
    for particle, fraction in zip(PARTICLES, fractions):
        sim_x, sim_counts = mc[particle]
        # Map fine MC histogram counts into the wider Auger bins.
        rebinned, _ = np.histogram(sim_x, bins=edges, weights=sim_counts)
        # Normalize using the full MC template; excludes negligible out-of-range tails
        # without artificially transferring those events into the measured range.
        n_mc = sim_counts.sum()
        if n_mc <= 0:
            raise ValueError(f'Empty MC template: {particle}')
        components[particle] = total * fraction * rebinned / n_mc

    model = np.sum(np.vstack([components[p] for p in PARTICLES]), axis=0)
    fig, ax = plt.subplots(figsize=(8.0, 5.8), constrained_layout=True)

    # Step curves trace the bin counts and avoid misleading interpolation.
    for p, f in zip(PARTICLES, fractions):
        ax.stairs(components[p], edges, color=COLORS[p], linestyle='--',
                  linewidth=1.7, label=f'{p} ({100*f:.1f}%)')
    ax.stairs(model, edges, color='olive', linewidth=2.0, label='Sum')
    ax.errorbar(x, observed, yerr=observed_err, fmt='o', color='black',
                markersize=4.3, capsize=2.5, elinewidth=1.0,
                label='Auger adat', zorder=10)

    ax.set_xlabel(r'$X_{\max}$ [g/cm$^2$]', fontsize=14)
    ax.set_ylabel('Események / 12 g/cm$^2$', fontsize=13)
    ax.text(0.03, 0.97, r'$\log_{10}(E/\mathrm{eV})=18$',
            transform=ax.transAxes, va='top', fontsize=12)
    ax.set_xlim(550, 1000)
    ax.set_ylim(bottom=0)
    ax.tick_params(direction='in', top=True, right=True)
    ax.legend(loc='upper right', fontsize=10, frameon=True)
    fig.savefig(args.output, dpi=250)
    if args.show:
        plt.show()
    plt.close(fig)

    print(f'Figure: {args.output.resolve()}')
    print(f'Auger event count = {total:g}')
    print(f'Model in Auger bin range = {model.sum():.5f}')
    for p, frac in zip(PARTICLES, fractions):
        print(f'{p:>2}: fraction={frac:.6f}, expected counts in plotted bins={components[p].sum():.3f}')


if __name__ == '__main__':
    main()
