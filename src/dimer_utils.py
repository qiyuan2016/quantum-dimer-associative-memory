"""Helper functions for quantum dimer Monte Carlo."""

from collections import defaultdict
from itertools import combinations
from math import prod

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
