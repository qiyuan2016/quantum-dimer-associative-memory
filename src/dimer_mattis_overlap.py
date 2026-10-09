"""Monte Carlo Mattis overlaps for the second-order quantum dimer Hamiltonian.

Run python dimer_mattis_overlap.py

Requires numpy, scipy, qutip, and matplotlib.
"""

import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.sparse.linalg import eigsh

try:
    from .dimer_utils import generate_pairings, second_order_coefficients
except ImportError:
    from dimer_utils import generate_pairings, second_order_coefficients

ROOT = Path(__file__).resolve().parent.parent


def build_second_order_hamiltonian(n, coefficients):
    """H2 = -sum_(ab,cd) J_(ab,cd) (sigma_a.sigma_b)(sigma_c.sigma_d)."""
    sigma_dot_sigma = sum(
        (qt.tensor(op, op) for op in (qt.sigmax(), qt.sigmay(), qt.sigmaz())),
        qt.tensor(qt.qzero(2), qt.qzero(2)),
    )
    bonds = {bond for bond_pair in coefficients for bond in bond_pair}
    expanded = {
        bond: qt.expand_operator(sigma_dot_sigma, [2] * n, list(bond)) for bond in bonds
    }
    hamiltonian = None
    for (first, second), weight in coefficients.items():
        term = -weight * (expanded[first] * expanded[second])
        hamiltonian = term if hamiltonian is None else hamiltonian + term
    if hamiltonian is None:
        raise ValueError("No second-order terms were generated")
    return hamiltonian


def total_spin_squared(n):
    """Return (sum_j sigma_j)^2."""
    identities = [qt.qeye(2)] * n
    total = None
    for pauli in (qt.sigmax(), qt.sigmay(), qt.sigmaz()):
        components = []
        for site in range(n):
            operators = identities.copy()
            operators[site] = pauli
            components.append(qt.tensor(operators))
        spin_component = sum(components[1:], components[0])
        squared = spin_component * spin_component
        total = squared if total is None else total + squared
    return total


def lowest_singlet(hamiltonian, spin_squared, k, tolerance=1e-10):
    """Return the first low-energy eigenvector passing the singlet test.

    Does not guarantee finding a singlet if the lowest k eigenvectors contain none.
    """
    matrix = hamiltonian.to("csr").data.as_scipy()
    dimension = matrix.shape[0]
    if not 1 <= k < dimension:
        raise ValueError(f"k must be between 1 and {dimension - 1}")
    ncv = min(dimension, 3 * k)
    eigenvalues, eigenvectors = eigsh(
        matrix,
        k=k,
        which="SA",
        tol=1e-14,
        maxiter=100000,
        ncv=ncv,
    )
    dims = hamiltonian.dims[0]
    for index in np.argsort(eigenvalues):
        state = qt.Qobj(eigenvectors[:, index], dims=[dims, [1] * len(dims)])
        if abs(qt.expect(spin_squared, state)) < tolerance:
            return state
    return None


def mattis_overlaps(state, memories):
    """Average singlet-projector expectation for every stored memory."""
    sigma_dot_sigma = sum(
        (qt.tensor(op, op) for op in (qt.sigmax(), qt.sigmay(), qt.sigmaz())),
        qt.tensor(qt.qzero(2), qt.qzero(2)),
    )
    projector = (qt.qeye([2, 2]) - sigma_dot_sigma) / 4
    expectations = {}
    overlaps = []
    for pairing in memories:
        values = []
        for bond in pairing:
            if bond not in expectations:
                expectations[bond] = float(
                    np.real(qt.expect(projector, state.ptrace(list(bond))))
                )
            values.append(expectations[bond])
        overlaps.append(float(np.mean(values)))
    return overlaps


def estimate_mattis_overlaps(n, p, n_runs, sigma, rng, spin_squared, k):
    """Return a list of overlap vectors; skip trials with no found singlet."""
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
            results.append(mattis_overlaps(ground_state, memories))
        # if (run + 1) % 100 == 0:
        #     print(f"N={n}, P={p}: {run + 1}/{n_runs} trials", flush=True)
    return results, skipped


def plot_mattis_overlaps(
    mattis_data,
    sizes=None,
    save_path=None,
    show=False,
):
    """Plot the mean maximum Mattis overlap for each (N, P)."""

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
        means = []

        for p in ps:
            realizations = mattis_data[(n, p)]

            if not realizations:
                continue

            maxima = [
                np.max(overlaps) for overlaps in realizations if len(overlaps) > 0
            ]

            if not maxima:
                continue

            x_values.append(p)
            means.append(np.mean(maxima))

        if not x_values:
            continue

        ax.plot(
            x_values,
            means,
            marker=markers[i % len(markers)],
            color=colors[i],
            lw=1,
            ms=5,
            label=rf"$N={n}$",
        )

    ax.set_xlabel(r"$K$", fontsize=22)
    ax.set_ylabel(r"$m_{\mathrm{max}}^q$", fontsize=22)

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
    sizes = [10]
    ps = {
        10: list(range(1, 31)),
        12: list(range(1, 31)),
        14: list(range(1, 31)),
    }
    n_runs = 2000
    sigma = 0.1
    rng = np.random.default_rng(12345)
    output = ROOT / "data" / "dimer_mattis_overlaps.pkl"
    output.parent.mkdir(parents=True, exist_ok=True)

    mattis_data = defaultdict(list)
    for n in sizes:
        spin_squared = total_spin_squared(n)
        k = (n + 2) + (n - 1) ** 2
        for p in ps[n]:
            print(f"N={n}, P={p}", flush=True)
            overlaps, skipped = estimate_mattis_overlaps(
                n, p, n_runs, sigma, rng, spin_squared, k
            )
            mattis_data[(n, p)].extend(overlaps)
            with output.open("wb") as file:
                pickle.dump(dict(mattis_data), file, protocol=pickle.HIGHEST_PROTOCOL)
            print(
                f"Saved {len(overlaps)} samples; skipped {skipped} without a singlet found",
                flush=True,
            )

    # with open(ROOT / "data" / "dimer_mattis_overlaps.pkl", "rb") as f:
    #     mattis_data = pickle.load(f)

    plot_mattis_overlaps(
        mattis_data,
        sizes=sizes,
        save_path=ROOT / "figures" / "dimer_mattis_overlaps.pdf",
    )


if __name__ == "__main__":
    main()
