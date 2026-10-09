"""Ground state Mattis overlaps for the standard classical Hopfield model.

Requires NumPy and Matplotlib. Run: python classical_hopfield_mattis.py
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def generate_patterns(n, k, rng):
    """Generate k distinct patterns, identifying a pattern with its anti-pattern."""
    if k > 2 ** (n - 1):
        raise ValueError(
            "Cannot generate more than 2**(n-1) sign-inequivalent patterns"
        )

    patterns = []
    seen = set()
    while len(patterns) < k:
        pattern = rng.choice([-1, 1], size=n).astype(np.int8)
        key = tuple(pattern)
        if key in seen:
            continue
        patterns.append(pattern)
        seen.add(key)
        seen.add(tuple(-pattern))
    return np.stack(patterns)


def build_hopfield_couplings(patterns):
    """J[i,j] = -sum_mu xi[mu,i]*xi[mu,j]/N, with zero diagonal."""
    n = patterns.shape[1]
    couplings = -(patterns.T.astype(float) @ patterns.astype(float)) / n
    np.fill_diagonal(couplings, 0.0)
    return couplings


def find_lowest_energy_pattern(couplings, batch_size=8192):
    """Exhaustively minimize sum_{i<j} J[i,j]*s[i]*s[j].

    Enumerates half the configurations by fixing the first spin to +1,
    exploiting global spin-flip symmetry. Uses batches to limit memory.
    """
    n = couplings.shape[0]
    if couplings.shape != (n, n):
        raise ValueError("Couplings must be a square matrix")
    if n < 1 or n > 62:
        raise ValueError("Enumeration requires 1 <= n <= 62")

    best_energy = np.inf
    best_pattern = None
    bit_positions = np.arange(n - 1, dtype=np.uint64)
    total = 1 << (n - 1)
    for start in range(0, total, batch_size):
        indices = np.arange(start, min(start + batch_size, total), dtype=np.uint64)
        bits = ((indices[:, None] >> bit_positions) & 1).astype(np.int8)
        states = np.concatenate(
            (np.ones((len(indices), 1), dtype=np.int8), 2 * bits - 1), axis=1
        )
        energies = 0.5 * np.einsum(
            "bi,ij,bj->b", states, couplings, states, optimize=True
        )
        winner = int(np.argmin(energies))
        if energies[winner] < best_energy:
            best_energy = float(energies[winner])
            best_pattern = states[winner].copy()
    return best_pattern


def mattis_overlaps(patterns, ground_state):
    """Return all signed overlaps m_mu = (1/N) sum_i xi[mu,i] * s[i]."""
    return (patterns @ ground_state / patterns.shape[1]).tolist()


def plot_classical_mattis_overlaps(
    mattis_data,
    sizes=None,
    save_path=None,
    show=False,
):
    """Plot the disorder-averaged maximum absolute Mattis overlap."""
    if sizes is None:
        sizes = sorted({n for n, p in mattis_data})

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
        ps = sorted(p for nn, p in mattis_data if nn == n)

        x_values = []
        averages = []

        for p in ps:
            realizations = mattis_data[(n, p)]

            if not realizations:
                continue

            maxima = [np.max(np.abs(overlaps)) for overlaps in realizations]

            x_values.append(p)
            averages.append(np.mean(maxima))

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
    ax.set_ylabel(r"$m_{\mathrm{max}}^c$", fontsize=22)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)

    ax.legend(
        frameon=False,
        fontsize=14,
        loc="upper left",
        bbox_to_anchor=(0.58, 0.98),
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


ROOT = Path(__file__).resolve().parent.parent


def main():
    sizes = [10, 12]
    ps = {
        10: list(range(1, 31, 1)),
        12: list(range(1, 31, 1)),
        14: list(range(1, 31, 1)),
    }
    n_runs = 2000
    seed = 12345
    output = ROOT / "data" / "classical_hopfield_mattis.pkl"
    output.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    mattis_data = {}

    for n in sizes:
        for p in ps[n]:
            overlaps_list = []

            for _ in range(n_runs):
                patterns = generate_patterns(n, p, rng)
                couplings = build_hopfield_couplings(patterns)
                ground_state = find_lowest_energy_pattern(couplings)
                overlaps = mattis_overlaps(patterns, ground_state)

                overlaps_list.append(overlaps)

            mattis_data[(n, p)] = overlaps_list

            # Save progress after each (N, P).
            with open(output, "wb") as file:
                pickle.dump(
                    mattis_data,
                    file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            print(f"N: {n}, P: {p}, completed", flush=True)

    # with open(ROOT / "data" / "classical_hopfield_mattis.pkl", "rb") as f:
    #     mattis_data = pickle.load(f)

    plot_classical_mattis_overlaps(
        mattis_data,
        sizes=sizes,
        save_path=ROOT / "figures" / "classical_hopfield_mattis.pdf",
    )


if __name__ == "__main__":
    main()
