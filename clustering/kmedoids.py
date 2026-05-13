"""
K-Medoids (PAM) clustering using a precomputed distance matrix.

A medoid is always a real data point (country), unlike K-Means which uses
abstract centroids. This makes results directly interpretable.
"""

import random


def _distance(matrix: list, i: int, j: int) -> float:
    """Convert similarity to distance."""
    return 1.0 - matrix[i][j]


def _total_cost(assignments: list[int], medoids: list[int], matrix: list) -> float:
    return sum(_distance(matrix, i, medoids[assignments[i]]) for i in range(len(assignments)))


def _assign(n: int, medoids: list[int], matrix: list) -> list[int]:
    """Assign each point to the nearest medoid."""
    assignments = []
    for i in range(n):
        best = min(range(len(medoids)), key=lambda m: _distance(matrix, i, medoids[m]))
        assignments.append(best)
    return assignments


def kmedoids(matrix: list, k: int, max_iter: int = 100, seed: int = 42) -> dict:
    """
    PAM (Partitioning Around Medoids) algorithm.

    Parameters
    ----------
    matrix   : N×N similarity matrix (values in [0,1])
    k        : number of clusters
    max_iter : iteration limit
    seed     : random seed for reproducibility

    Returns
    -------
    {
      "medoids"     : [int, ...]          index of each medoid in the country list
      "assignments" : [int, ...]          cluster index (0..k-1) for each country
      "cost"        : float               total distance (sum of distances to medoids)
      "iterations"  : int                 number of iterations until convergence
      "inertia_history": [float, ...]     cost after each iteration
    }
    """
    n = len(matrix)
    if k >= n:
        raise ValueError(f"k ({k}) must be less than number of countries ({n})")

    rng = random.Random(seed)
    medoids = rng.sample(range(n), k)
    assignments = _assign(n, medoids, matrix)
    cost = _total_cost(assignments, medoids, matrix)
    inertia_history = [cost]

    for iteration in range(1, max_iter + 1):
        improved = False

        for m_idx in range(k):
            for candidate in range(n):
                if candidate in medoids:
                    continue

                new_medoids = medoids[:]
                new_medoids[m_idx] = candidate
                new_assignments = _assign(n, new_medoids, matrix)
                new_cost = _total_cost(new_assignments, new_medoids, matrix)

                if new_cost < cost - 1e-9:
                    medoids = new_medoids
                    assignments = new_assignments
                    cost = new_cost
                    improved = True
                    break

            if improved:
                break

        inertia_history.append(cost)
        if not improved:
            break

    clusters = [[] for _ in range(k)]
    for i, m_idx in enumerate(assignments):
        clusters[m_idx].append(i)

    return {
        "medoids": medoids,
        "assignments": assignments,
        "clusters": clusters,
        "cost": cost,
        "iterations": iteration,
        "inertia_history": inertia_history,
    }
