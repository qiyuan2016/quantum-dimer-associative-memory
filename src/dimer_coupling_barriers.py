"""Monte Carlo study of the hybridization metric,
i.e., H2 couplings relative to energy barriers between a memory and its nearby dimer coverings.

Requires NumPy and Matplotlib. Run: python dimer_coupling_barriers.py
"""

import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .dimer_utils import bond, generate_pairings, second_order_coefficients
except ImportError:
    from dimer_utils import bond, generate_pairings, second_order_coefficients


ROOT = Path(__file__).resolve().parent.parent


def generate_transition_graph(pairing):
    """Map each site to (partner, orientation) for an oriented dimer covering."""
    graph = {}
    for a, b in pairing:
        graph[a] = (b, 1)
        graph[b] = (a, -1)
    return graph


def swap_pairs(pairing, q1, q2):
    """Swap site labels q1 and q2, returning a canonical dimer covering."""
    swapped = []
    for a, b in pairing:
        a = q2 if a == q1 else q1 if a == q2 else a
        b = q2 if b == q1 else q1 if b == q2 else b
        swapped.append(bond(a, b))
    return tuple(sorted(swapped))


def compute_overlap(graph_1, graph_2):
    """Signed overlap of two valence-bond states via transition loops."""
    n = len(graph_1)
    visited = set()
    sign = (-1) ** (n // 2)
    n_loops = 0

    for start in range(n):
        if start in visited:
            continue
        n_loops += 1
        site = start
        while site not in visited:
            visited.add(site)
            site, direction = graph_1[site]
            sign *= direction
            visited.add(site)
            site, direction = graph_2[site]
            sign *= direction

    return sign * 2.0 ** (n_loops - n // 2)


def swap_transition_graph(graph, q1, q2):
    """Swap site labels q1 and q2, returning a new transition graph."""
    new_graph = graph.copy()
    if graph[q1][0] == q2:
        new_graph[q1] = (q2, -graph[q1][1])
        new_graph[q2] = (q1, -graph[q2][1])
        return new_graph

    partner_1, direction_1 = graph[q1]
    partner_2, direction_2 = graph[q2]
    new_graph[q1] = (partner_2, direction_2)
    new_graph[q2] = (partner_1, direction_1)
    new_graph[partner_1] = (q2, graph[partner_1][1])
    new_graph[partner_2] = (q1, graph[partner_2][1])
    return new_graph


def compute_h2_coupling(coefficients, graph_1, graph_2, existing_pairs):
    """Compute the H2 matrix element."""
    overlap_12 = compute_overlap(graph_1, graph_2)
    coupling = 0.0

    for (first, second), coefficient in coefficients.items():
        a, b = first
        c, d = second
        first_exists = first in existing_pairs
        second_exists = second in existing_pairs

        if first_exists and second_exists:
            coupling -= 9 * coefficient * overlap_12
        elif first_exists:
            swapped = swap_transition_graph(graph_2, c, d)
            coupling += coefficient * (
                6 * compute_overlap(graph_1, swapped) - 3 * overlap_12
            )
        elif second_exists:
            swapped = swap_transition_graph(graph_2, a, b)
            coupling += coefficient * (
                6 * compute_overlap(graph_1, swapped) - 3 * overlap_12
            )
        else:
            swapped_1 = swap_transition_graph(graph_1, a, b)
            swapped_2 = swap_transition_graph(graph_2, c, d)
            coupling += coefficient * (
                -4 * compute_overlap(swapped_1, swapped_2)
                + 2 * compute_overlap(swapped_1, graph_2)
                + 2 * compute_overlap(swapped_2, graph_1)
                - overlap_12
            )

    return coupling


def coupling_barrier_ratios(pairing, coefficients, tolerance=1e-6):
    """Compute ratios for all distinct coverings obtained by two-site swaps."""
    n = 2 * len(pairing)
    graph = generate_transition_graph(pairing)
    energy = compute_h2_coupling(coefficients, graph, graph, set(pairing))
    visited = set()
    ratios = []

    for j in range(n - 1):
        for k in range(j + 1, n):
            if graph[j][0] == k:
                continue  # Swapping two sites in one singlet is not a new covering.
            new_pairing = swap_pairs(pairing, j, k)
            if new_pairing in visited:
                continue
            visited.add(new_pairing)

            new_graph = generate_transition_graph(new_pairing)
            overlap = compute_overlap(new_graph, graph)
            norm = 1.0 - overlap**2
            if norm <= 0:
                continue

            new_energy = compute_h2_coupling(
                coefficients, new_graph, new_graph, set(new_pairing)
            )
            coupling = compute_h2_coupling(
                coefficients, new_graph, graph, set(new_pairing) | set(pairing)
            )
            orthogonal_energy = (
                new_energy + overlap**2 * energy - 2 * overlap * coupling
            ) / norm
            barrier = (orthogonal_energy - energy) * np.sqrt(norm)

            if abs(barrier) > tolerance:
                ratios.append(abs((coupling - overlap * energy) / barrier))

    return ratios


def estimate_coupling_barriers(n, p, n_runs, rng, sigma=0.0, tolerance=1e-6):
    """Return one list of coupling/barrier ratios per independent realization."""
    results = []
    for _ in range(n_runs):
        memories = generate_pairings(n, p, rng)
        coefficients = second_order_coefficients(memories, sigma=sigma, rng=rng)
        target = memories[int(rng.integers(p))]
        results.append(coupling_barrier_ratios(target, coefficients, tolerance))
    return results


def plot_coupling_barriers(
    coupling_data,
    save_path=None,
    show=False,
):
    """Plot coupling-to-energy-barrier ratios for all available (N, P)."""

    sizes = sorted({n for n, p in coupling_data})

    colors = [
        "#0072B2",
        "#E69F00",
        "#009E73",
        "#CC79A7",
    ]
    markers = ["o", "s", "^", "D", "v", "P"]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    fig.subplots_adjust(
        left=0.23,
        right=0.96,
        bottom=0.23,
        top=0.96,
    )

    for i, n in enumerate(sizes):
        x_values = []
        means = []
        stds = []

        ps = sorted(p for nn, p in coupling_data if nn == n)

        for p in ps:
            maxima = np.array(
                [max(ratios) for ratios in coupling_data[(n, p)] if len(ratios) > 0]
            )

            if len(maxima) == 0:
                continue

            x_values.append(p)
            means.append(np.mean(maxima))
            stds.append(np.std(maxima))

        if not x_values:
            continue

        x_values = np.asarray(x_values)
        means = np.asarray(means)
        stds = np.asarray(stds)

        ax.errorbar(
            x_values,
            means,
            yerr=[np.minimum(stds, means), stds],
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            lw=1,
            ms=5,
            capsize=0,
            label=rf"$N={n}$",
        )

    ax.set_xlabel(r"$K$", fontsize=22)
    ax.set_ylabel(r"$W$", fontsize=22)

    ax.legend(
        frameon=False,
        fontsize=16,
        loc="upper right",
    )

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
    sizes = [16, 20, 24]
    ps = {
        16: [1, *range(5, 105, 5)],
        20: [1, *range(10, 210, 10)],
        24: [1, *range(20, 360, 20)],
    }
    n_runs = 1000
    sigma = 0.0
    tolerance = 1e-6
    seed = 12345

    rng = np.random.default_rng(seed)
    output = ROOT / "data" / "dimer_coupling_barriers.pkl"
    output.parent.mkdir(parents=True, exist_ok=True)
    coupling_data = defaultdict(list)

    for n in sizes:
        for p in ps[n]:
            print(f"N={n}, P={p}", flush=True)
            coupling_data[(n, p)] = estimate_coupling_barriers(
                n, p, n_runs, rng, sigma=sigma, tolerance=tolerance
            )
            # Save after each completed (N, P) point.
            with output.open("wb") as file:
                pickle.dump(dict(coupling_data), file, protocol=pickle.HIGHEST_PROTOCOL)

    # with open(ROOT / "data" / "dimer_coupling_barriers.pkl", "rb") as f:
    #     coupling_data = pickle.load(f)

    plot_coupling_barriers(
        coupling_data,
        save_path=ROOT / "figures" / "dimer_coupling_barriers.pdf",
    )


if __name__ == "__main__":
    main()
