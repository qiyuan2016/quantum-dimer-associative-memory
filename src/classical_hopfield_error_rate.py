"""Monte Carlo local stability error rates for the classical Hopfield model.

Requires NumPy and Matplotlib. Run: python classical_hopfield_error_rate.py
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .classical_hopfield_mattis import build_hopfield_couplings, generate_patterns
except ImportError:
    from classical_hopfield_mattis import build_hopfield_couplings, generate_patterns


ROOT = Path(__file__).resolve().parent.parent


def is_unstable(pattern, couplings, tolerance=0.0):
    """Return True if a single-spin flip does not increase the energy.

    Couplings are the Hamiltonian coefficients J_ij (negative Hebbian weights)
    with zero diagonal, so E(s) = sum_{i<j} J_ij s_i s_j.
    """
    delta_energies = -2.0 * pattern * (couplings @ pattern)
    return bool(np.any(delta_energies <= tolerance))


def estimate_error_rate(n, p, n_runs, rng, tolerance=0.0):
    """Fraction of disorder samples with an unstable randomly chosen memory."""
    n_errors = 0

    for _ in range(n_runs):
        patterns = generate_patterns(n, p, rng)
        couplings = build_hopfield_couplings(patterns)
        target = patterns[rng.integers(p)]
        n_errors += is_unstable(target, couplings, tolerance)

    rate = n_errors / n_runs
    se = np.sqrt(rate * (1.0 - rate) / n_runs)
    return rate, se


def plot_classical_error_rates(
    error_rates,
    n_runs,
    sizes=None,
    save_path=None,
    show=False,
):
    """Plot classical Hopfield local instability probabilities."""
    if sizes is None:
        sizes = sorted({n for n, p in error_rates})

    colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))
    markers = ["o", "s", "^", "D", "v", "P"]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    fig.subplots_adjust(
        left=0.23,
        right=0.96,
        bottom=0.23,
        top=0.96,
    )

    for i, n in enumerate(sizes):
        ps = np.array(sorted(p for nn, p in error_rates if nn == n))

        rates = np.array([error_rates[(n, p)] for p in ps])

        errors = np.sqrt(rates * (1 - rates) / n_runs)

        ax.errorbar(
            ps * np.log(n) / n,
            rates,
            yerr=errors,
            marker=markers[i % len(markers)],
            color=colors[i],
            lw=1,
            ms=5,
            label=rf"$N={n}$",
        )

    ax.axhline(
        0.05,
        color="gray",
        ls="--",
        lw=1.5,
        alpha=0.8,
    )

    ax.set_xlabel(r"$K/(N/\log N)$", fontsize=22)
    ax.set_ylabel(r"$p_{\mathrm{unstable}}$", fontsize=22)

    ax.legend(frameon=False, fontsize=18)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=18,
        direction="in",
        length=5,
        width=1.2,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    if show:
        plt.show()

    plt.close(fig)


def main():
    sizes = [50, 60, 70]
    ps = {
        50: list(range(1, 2, 1)) + list(range(2, 20, 1)),
        60: list(range(1, 2, 1)) + list(range(2, 23, 1)),
        70: list(range(1, 2, 1)) + list(range(2, 26, 1)),
        80: list(range(1, 2, 1)) + list(range(2, 28, 1)),
        90: list(range(1, 2, 1)) + list(range(2, 31, 1)),
        100: list(range(1, 2, 1)) + list(range(2, 33, 1)),
    }
    n_runs = 2000
    seed = 12345
    output = ROOT / "data" / "classical_hopfield_error_rates.pkl"
    output.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    error_rates = {}

    for n in sizes:
        for p in ps[n]:
            rate, se = estimate_error_rate(n, p, n_runs, rng)
            error_rates[(n, p)] = rate

            # Save progress after each (N, P).
            with output.open("wb") as file:
                pickle.dump(error_rates, file, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"N: {n}, P: {p}, error rate: {rate}", flush=True)

    # with (ROOT / "data" / "classical_hopfield_error_rates.pkl").open("rb") as f:
    #     error_rates = pickle.load(f)

    plot_classical_error_rates(
        error_rates,
        n_runs,
        sizes=sizes,
        save_path=ROOT / "figures" / "classical_hopfield_error_rates.pdf",
    )


if __name__ == "__main__":
    main()
