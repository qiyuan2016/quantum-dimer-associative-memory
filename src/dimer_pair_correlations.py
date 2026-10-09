"""Monte Carlo singlet-projector pair correlations for the quantum dimer model.

Run python dimer_pair_correlations.py

Requires numpy, scipy, qutip, and matplotlib.
"""

import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

try:
    from .dimer_mattis_overlap import (
        build_second_order_hamiltonian,
        lowest_singlet,
        total_spin_squared,
    )
    from .dimer_utils import generate_pairings, second_order_coefficients
except ImportError:
    from dimer_mattis_overlap import (
        build_second_order_hamiltonian,
        lowest_singlet,
        total_spin_squared,
    )
    from dimer_utils import generate_pairings, second_order_coefficients

ROOT = Path(__file__).resolve().parent.parent


def pair_correlations(state, n):
    """Return D[i,j] = <P_ij> for i != j, with zero diagonal."""
    sigma_dot_sigma = sum(
        (qt.tensor(op, op) for op in (qt.sigmax(), qt.sigmay(), qt.sigmaz())),
        qt.tensor(qt.qzero(2), qt.qzero(2)),
    )
    projector = (qt.qeye([2, 2]) - sigma_dot_sigma) / 4

    correlations = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        for j in range(i + 1, n):
            reduced_state = state.ptrace([i, j])
            value = float(np.real(qt.expect(projector, reduced_state)))
            correlations[i, j] = value
            correlations[j, i] = value
    return correlations


def estimate_pair_correlations(n, p, n_runs, sigma, rng, spin_squared, k):
    """Return sampled correlation matrices and number of skipped trials."""
    results = []
    skipped = 0

    for run in range(n_runs):
        memories = generate_pairings(n, p, rng)
        coefficients = second_order_coefficients(memories, sigma=sigma, rng=rng)
        hamiltonian = build_second_order_hamiltonian(n, coefficients)
        ground_state = lowest_singlet(hamiltonian, spin_squared, k)

        if ground_state is None:
            skipped += 1
        else:
            results.append(pair_correlations(ground_state, n))

        # if (run + 1) % 100 == 0:
        #     print(f"N={n}, P={p}: {run + 1}/{n_runs} trials", flush=True)

    return results, skipped


def effective_participation(d, tol=1e-6):
    """Compute the effective participation number for one correlation matrix."""
    n = len(d)
    d = np.asarray(d)

    participation = 0.0

    for i in range(n):
        weights = np.maximum(0, 2 * d[i] - 1) ** 2
        weights[i] = 0.0

        if np.sum(weights) > tol:
            weights /= np.sum(weights)
            participation += 1 / np.sum(weights**2)

    return participation / n


def plot_effective_participation(
    pair_data,
    sizes=None,
    save_path=None,
    show=False,
):
    """Plot disorder-averaged effective participation versus memory count."""
    if sizes is None:
        sizes = sorted({n for n, p in pair_data})

    colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))
    markers = ["o", "s", "^", "D", "v", "P"]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    fig.subplots_adjust(
        left=0.22,
        right=0.96,
        bottom=0.22,
        top=0.96,
    )

    for k, n in enumerate(sizes):
        ps = sorted(p for nn, p in pair_data if nn == n)

        x_values = []
        averages = []

        for p in ps:
            matrices = pair_data[(n, p)]

            if not matrices:
                continue

            values = [effective_participation(d) for d in matrices]

            x_values.append(p)
            averages.append(np.mean(values))

        if not x_values:
            continue

        ax.plot(
            x_values,
            averages,
            marker=markers[k % len(markers)],
            color=colors[k],
            lw=1,
            ms=5,
            label=rf"$N={n}$",
        )

    ax.set_xlabel(r"$K$", fontsize=22)
    ax.set_ylabel(r"$N_{\mathrm{eff}}$", fontsize=22)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)

    ax.legend(
        frameon=False,
        fontsize=14,
        loc="upper left",
        bbox_to_anchor=(0.6, 0.48),
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=18,
        direction="in",
        length=5,
        width=1.2,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    if show:
        plt.show()

    plt.close(fig)


def average_concurrence_squared(d):
    """Compute the mean sum of squared concurrences per spin."""
    d = np.asarray(d)

    cij = np.maximum(0, 2 * d - 1)
    np.fill_diagonal(cij, 0)

    return np.sum(cij**2) / len(d)


def plot_average_concurrence_squared(
    pair_data,
    sizes=None,
    save_path=None,
    show=False,
):
    """Plot disorder-averaged squared concurrence versus memory count."""
    if sizes is None:
        sizes = sorted({n for n, p in pair_data})

    colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))
    markers = ["o", "s", "^", "D", "v", "P"]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    fig.subplots_adjust(
        left=0.22,
        right=0.96,
        bottom=0.22,
        top=0.96,
    )

    for i, n in enumerate(sizes):
        ps = sorted(p for nn, p in pair_data if nn == n)

        x_values = []
        averages = []

        for p in ps:
            matrices = pair_data[(n, p)]

            if not matrices:
                continue

            values = [average_concurrence_squared(d) for d in matrices]

            x_values.append(p)
            averages.append(np.mean(values))

        if not x_values:
            continue

        ax.plot(
            x_values,
            averages,
            marker=markers[i % len(markers)],
            color=colors[i],
            lw=1,
            ms=5,
            label=rf"$N={n}$",
        )

    ax.set_xlabel(r"$K$", fontsize=22)
    ax.set_ylabel(r"$\bar{C}^2$", fontsize=22)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)

    ax.legend(
        frameon=False,
        fontsize=14,
        loc="upper left",
        bbox_to_anchor=(0.6, 0.98),
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=18,
        direction="in",
        length=5,
        width=1.2,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    if show:
        plt.show()

    plt.close(fig)


def main():
    sizes = [10]
    ps = {
        10: list(range(1, 31)),
        12: list(range(1, 31)),
        14: list(range(1, 31)),
    }
    n_runs = 1000
    sigma = 0.1
    seed = 12345

    rng = np.random.default_rng(seed)
    output = ROOT / "data" / "dimer_pair_correlations.pkl"
    output.parent.mkdir(parents=True, exist_ok=True)

    pair_data = defaultdict(list)
    for n in sizes:
        spin_squared = total_spin_squared(n)
        k = n + 2
        for p in ps[n]:
            print(f"N={n}, P={p}", flush=True)
            matrices, skipped = estimate_pair_correlations(
                n, p, n_runs, sigma, rng, spin_squared, k
            )
            pair_data[(n, p)].extend(matrices)

            with output.open("wb") as file:
                pickle.dump(dict(pair_data), file, protocol=pickle.HIGHEST_PROTOCOL)

            print(
                f"Saved {len(matrices)} matrices; skipped {skipped} without a singlet",
                flush=True,
            )

    # with open(ROOT / "data" / "dimer_pair_correlations.pkl", "rb") as f:
    #     pair_data = pickle.load(f)

    plot_effective_participation(
        pair_data,
        sizes=sizes,
        save_path=ROOT / "figures" / "dimer_effective_participation.pdf",
    )

    plot_average_concurrence_squared(
        pair_data,
        sizes=sizes,
        save_path=ROOT / "figures" / "dimer_concurrence_squared.pdf",
    )


if __name__ == "__main__":
    main()
