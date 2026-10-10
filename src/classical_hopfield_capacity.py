"""Estimate classical Hopfield local-stability capacities by Monte Carlo.

Requires NumPy, Matplotlib and SciPy. Run: python classical_hopfield_capacity.py
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

try:
    from .classical_hopfield_error_rate import estimate_error_rate
except ImportError:
    from classical_hopfield_error_rate import estimate_error_rate


ROOT = Path(__file__).resolve().parent.parent


def estimate_capacity(
    n,
    n_runs,
    rng,
    start_p=1,
    max_p=20,
    error_threshold=0.05,
):
    """Return the last P before the first estimated error-rate threshold crossing.

    Returns None if the threshold is not reached by max_p.
    """
    for p in range(start_p, max_p + 1):
        rate, _ = estimate_error_rate(n, p, n_runs, rng)
        print(f"N: {n}, P: {p}, error rate: {rate:.6f}", flush=True)

        if rate >= error_threshold:
            return p - 1

    return None


def plot_classical_capacity(
    capacities,
    sizes=None,
    save_path=None,
    show=False,
):
    """Plot classical Hopfield capacity and fit N/log(N) scaling."""
    if sizes is None:
        sizes = np.array(
            sorted(n for n, values in capacities.items() if len(values) > 0)
        )

    means = np.array([np.mean(capacities[n]) for n in sizes])
    stds = np.array([np.std(capacities[n]) for n in sizes])

    def scaling(n, a):
        return a * n / np.log(n)

    params, cov = curve_fit(scaling, sizes, means)
    a = params[0]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    fig.subplots_adjust(
        left=0.23,
        right=0.96,
        bottom=0.23,
        top=0.96,
    )

    ax.errorbar(
        sizes,
        means,
        yerr=stds,
        fmt="o",
        markersize=5,
        color="black",
        ecolor="black",
        elinewidth=1,
        capsize=5,
        label="Simulation",
    )

    fit_x = np.linspace(min(sizes), max(sizes), 500)

    ax.plot(
        fit_x,
        scaling(fit_x, a),
        lw=1.5,
        color="firebrick",
        label=r"Fit $\propto N/\log N$",
    )

    ax.set_xlabel(r"$N$", fontsize=22)
    ax.set_ylabel(r"$K_{\mathrm{max}}^c$", fontsize=22)

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

    return a


def main():
    sizes = [50, 60, 70, 80, 90, 100]
    # n_repeats = 10
    # n_runs = 2000
    # start_p = 1
    # max_p = 20
    # error_threshold = 0.05
    # seed = 12345

    # output = ROOT / "data" / "classical_hopfield_capacities.pkl"
    # output.parent.mkdir(parents=True, exist_ok=True)

    # rng = np.random.default_rng(seed)
    # capacities = {n: [] for n in sizes}

    # for n in sizes:
    #     for repeat in range(n_repeats):
    #         capacity = estimate_capacity(
    #             n,
    #             n_runs,
    #             rng,
    #             start_p=start_p,
    #             max_p=max_p,
    #             error_threshold=error_threshold,
    #         )
    #         capacities[n].append(capacity)

    #         # Save progress after each completed capacity estimate.
    #         with output.open("wb") as file:
    #             pickle.dump(capacities, file, protocol=pickle.HIGHEST_PROTOCOL)

    #         print(
    #             f"N={n}, repeat={repeat + 1}/{n_repeats}, capacity={capacity}",
    #             flush=True,
    #         )

    with open(ROOT / "data" / "classical_hopfield_capacities.pkl", "rb") as f:
        capacities = pickle.load(f)

    a = plot_classical_capacity(
        capacities,
        sizes=sizes,
        save_path=ROOT / "figures" / "classical_hopfield_capacities.pdf",
    )


if __name__ == "__main__":
    main()
