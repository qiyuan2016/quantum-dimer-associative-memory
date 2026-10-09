"""Monte Carlo local instability error rates for quantum dimer memories.

Requires NumPy and Matplotlib. Run: python dimer_error_rate.py
"""

import pickle
from collections import defaultdict
from itertools import combinations
from math import prod

import matplotlib.pyplot as plt
import numpy as np


def bond(a, b):
    """Canonical representation of an unordered singlet bond."""
    return tuple(sorted((int(a), int(b))))


def bond_pair(first, second):
    """Canonical representation of two unordered bonds."""
    return tuple(sorted((first, second)))


def number_of_matchings(n):
    """Number of perfect matchings on n labeled spins."""
    if n < 0 or n % 2:
        raise ValueError("n must be a nonnegative even integer")
    return prod(range(1, n, 2))


def generate_pairings(n, count, rng):
    """Draw count distinct uniformly random perfect matchings."""
    if n < 2 or n % 2:
        raise ValueError("n must be even and at least 2")
    if not 1 <= count <= number_of_matchings(n):
        raise ValueError("count exceeds the number of available matchings")

    seen = set()
    memories = []
    while len(memories) < count:
        permutation = rng.permutation(n)
        pairing = tuple(
            sorted(bond(permutation[k], permutation[k + 1]) for k in range(0, n, 2))
        )
        if pairing not in seen:
            seen.add(pairing)
            memories.append(pairing)
    return memories


def partner_map(pairing):
    """Map each spin to its singlet partner in a dimer covering."""
    partners = {}
    for a, b in pairing:
        partners[a] = b
        partners[b] = a
    return partners


def second_order_coefficients(memories, sigma=0.0, rng=None):
    """Compute second-order Hamiltonian coefficients.

    Each memory has weight 1 when sigma=0.
    Otherwise, positive Gaussian weights are sampled.
    """
    coefficients = defaultdict(float)

    if sigma > 0 and rng is None:
        rng = np.random.default_rng()

    for pairing in memories:
        weight = 1.0

        if sigma > 0:
            weight = rng.normal(1.0, sigma)
            while weight < 1e-6:
                weight = rng.normal(1.0, sigma)

        for first, second in combinations(pairing, 2):
            coefficients[bond_pair(first, second)] += weight

    return coefficients


def energy_difference(i, j, pairing, partners, coefficients):
    """analytic second-order energy difference expression."""
    pi, pj = partners[i], partners[j]
    original_i, original_j = bond(i, pi), bond(j, pj)
    rearranged_i, rearranged_j = bond(i, pj), bond(j, pi)

    def coefficient(a, b):
        return coefficients.get(bond_pair(a, b), 0.0)

    delta = 16.0 * (
        coefficient(original_i, original_j) - coefficient(rearranged_i, rearranged_j)
    )

    for other in pairing:
        if other in (original_i, original_j):
            continue
        delta += 24.0 * (
            coefficient(original_i, other)
            + coefficient(original_j, other)
            - coefficient(rearranged_i, other)
            - coefficient(rearranged_j, other)
        )

    affected = {i, j, pi, pj}
    for m, n in pairing:
        if m in affected:
            continue
        for a, b, sign in (
            (i, pi, +1),
            (j, pj, +1),
            (i, pj, -1),
            (j, pi, -1),
        ):
            delta += (
                8.0
                * sign
                * (
                    coefficient(bond(a, m), bond(b, n))
                    + coefficient(bond(a, n), bond(b, m))
                )
            )
    return delta


def is_unstable(pairing, coefficients, tolerance=1e-8):
    """Whether any two spins from different singlets have delta E <= tolerance."""
    partners = partner_map(pairing)
    for i, j in combinations(range(len(partners)), 2):
        if partners[i] == j:
            continue
        if energy_difference(i, j, pairing, partners, coefficients) <= tolerance:
            return True
    return False


def estimate_error_rate(n, p, n_runs, rng, sigma=0.0, tolerance=1e-8):
    """Estimate the instability probability of a uniformly selected memory."""
    failures = 0

    for _ in range(n_runs):
        memories = generate_pairings(n, p, rng)
        coefficients = second_order_coefficients(memories, sigma, rng)
        target = memories[int(rng.integers(p))]
        failures += is_unstable(target, coefficients, tolerance)

    rate = failures / n_runs
    standard_error = np.sqrt(rate * (1.0 - rate) / n_runs)

    return rate, standard_error


def plot_error_rates(error_rates, n_runs, sizes=None, save_path=None, show=True):
    """Plot instability probabilities against P/N^3 with binomial error bars.

    Only available (N, P) results are plotted, allowing partial runs.
    """
    if sizes is None:
        sizes = sorted({n for n, _ in error_rates})

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    fig.subplots_adjust(left=0.23, right=0.96, bottom=0.23, top=0.96)

    colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))
    markers = ("o", "s", "^", "D", "v", "P")

    for index, n in enumerate(sizes):
        points = sorted(
            (p, rate) for (size, p), rate in error_rates.items() if size == n
        )
        if not points:
            continue

        p_values = np.array([p for p, _ in points])
        rates = np.array([rate for _, rate in points])
        errors = np.sqrt(rates * (1.0 - rates) / n_runs)

        ax.errorbar(
            p_values / n**3,
            rates,
            yerr=errors,
            marker=markers[index % len(markers)],
            color=colors[index],
            lw=1,
            ms=5,
            label=rf"$N={n}$",
        )

    ax.axhline(0.05, color="gray", ls="--", lw=1.5, alpha=0.8)
    ax.set_xlabel(r"$K/N^3$", fontsize=22)
    ax.set_ylabel(r"$p_{\mathrm{unstable}}$", fontsize=22)
    ax.legend(frameon=False, fontsize=18)
    ax.tick_params(
        axis="both", which="major", labelsize=18, direction="in", length=5, width=1.2
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)

    if save_path is not None:
        fig.savefig(save_path)
    if show:
        plt.show()
    return fig, ax


def main():
    sizes = (16, 20, 24, 28)
    ps = {
        16: [1, *range(2, 280, 6)],
        20: [1, *range(2, 550, 10)],
        24: [1, *range(2, 940, 16)],
        28: [1, *range(2, 1480, 24)],
    }
    n_runs = 2000
    sigma = 0.0
    seed = 12345
    output = "../data/dimer_error_rates.pkl"

    rng = np.random.default_rng(seed)
    error_rates = {}

    for n in sizes:
        for p in ps[n]:
            rate, se = estimate_error_rate(n, p, n_runs, rng, sigma)
            error_rates[(n, p)] = rate

            # Save progress after each (N, P).
            with open(output, "wb") as file:
                pickle.dump(error_rates, file, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"N: {n}, P: {p}, error rate: {rate}", flush=True)

    plot_error_rates(
        error_rates, n_runs, sizes=sizes, save_path="../figures/dimer_error_rates.pdf"
    )


if __name__ == "__main__":
    main()
