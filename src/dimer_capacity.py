"""Estimate capacity using the instability error rate.

Requires NumPy, Matplotlib and SciPy. Run: python dimer_capacity.py
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

try:
    from .dimer_error_rate import estimate_error_rate
except ImportError:
    from dimer_error_rate import estimate_error_rate

ROOT = Path(__file__).resolve().parent.parent


def estimate_capacity(
    n, n_runs, rng, start_p=1, max_p=1200, error_threshold=0.05, sigma=0.0
):
    for p in range(start_p, max_p + 1):
        rate, se = estimate_error_rate(n, p, n_runs, rng, sigma=sigma)
        print(f"N={n}, P={p}, error rate={rate:.4f} ± {se:.4f}", flush=True)
        if rate >= error_threshold:
            return p - 1
    return None


def plot_capacity(
    capacities,
    sizes=None,
    save_path=None,
    show=False,
):
    """Plot the capacity scaling and fit K_max = a * N^3."""
    if sizes is None:
        sizes = sorted(capacities)

    sizes = np.asarray(sizes)

    means = np.array([np.mean(capacities[n]) for n in sizes])
    stds = np.array([np.std(capacities[n]) for n in sizes])

    # Cubic scaling model
    def power_law(n, a):
        return a * n**3

    # Weighted least-squares fit
    params, cov = curve_fit(
        power_law,
        sizes,
        means,
        sigma=np.where(stds > 0, stds, 1.0),
    )

    a = params[0]

    # Plot
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
        elinewidth=2,
        capsize=5,
        label="Simulation",
    )

    fit_x = np.linspace(min(sizes), max(sizes), 500)

    ax.plot(
        fit_x,
        power_law(fit_x, a),
        lw=1.5,
        color="firebrick",
        label="Cubic fit",
    )

    ax.set_xlabel(r"$N$", fontsize=22)
    ax.set_ylabel(r"$K_\mathrm{max}^q$", fontsize=22)

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
    sizes = (16,)
    n_repeats = 10
    n_runs = 2000
    rng = np.random.default_rng(12345)
    output = ROOT / "data" / "dimer_capacities.pkl"
    output.parent.mkdir(parents=True, exist_ok=True)
    capacities = {n: [] for n in sizes}
    for n in sizes:
        for repeat in range(n_repeats):
            capacity = estimate_capacity(n, n_runs, rng)
            capacities[n].append(capacity)
            with output.open("wb") as f:
                pickle.dump(capacities, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(
                f"N={n}, repeat={repeat + 1}/{n_repeats}, capacity={capacity}",
                flush=True,
            )

    # with open(ROOT / "data" / "dimer_capacities.pkl", "rb") as f:
    #     capacities = pickle.load(f)

    a = plot_capacity(
        capacities,
        sizes=[16, 20, 24, 28, 32, 36, 40],
        save_path=ROOT / "figures" / "dimer_capacities.pdf",
    )


if __name__ == "__main__":
    main()
