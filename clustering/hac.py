"""
Hierarchical Agglomerative Clustering (HAC) using a precomputed similarity matrix.

Supports single, complete, and average linkage.
Produces a full dendrogram that can be cut at any number of clusters.
"""


def _dist(matrix: list, i: int, j: int) -> float:
    return 1.0 - matrix[i][j]


def _linkage_distance(
    matrix: list,
    cluster_a: list[int],
    cluster_b: list[int],
    linkage: str,
) -> float:
    distances = [_dist(matrix, a, b) for a in cluster_a for b in cluster_b]
    if linkage == "single":
        return min(distances)
    if linkage == "complete":
        return max(distances)
    return sum(distances) / len(distances)  # average


def hac(matrix: list, linkage: str = "average") -> dict:
    """
    Hierarchical Agglomerative Clustering.

    Parameters
    ----------
    matrix  : N×N similarity matrix (values in [0,1])
    linkage : "single" | "complete" | "average"

    Returns
    -------
    {
      "linkage"  : str
      "merges"   : list of merge steps, each:
                   {
                     "step"     : int,
                     "cluster_a": int   (original country index or merge id),
                     "cluster_b": int,
                     "distance" : float,
                     "size"     : int   (countries in new cluster)
                   }
      "dendro"   : D3-ready nested structure for dendrogram rendering
    }

    Cut with hac_cut(result, n_clusters) to get cluster assignments.
    """
    if linkage not in ("single", "complete", "average"):
        raise ValueError("linkage must be 'single', 'complete', or 'average'")

    n = len(matrix)

    # Each cluster is identified by an id; leaf ids = 0..n-1,
    # merged cluster ids start at n.
    cluster_members: dict[int, list[int]] = {i: [i] for i in range(n)}
    active = set(range(n))
    next_id = n
    merges = []

    # For D3 dendrogram: store node info by cluster id
    dendro_nodes: dict[int, dict] = {i: {"id": i, "leaf": True, "index": i} for i in range(n)}

    for step in range(n - 1):
        active_list = sorted(active)
        best_dist = float("inf")
        best_pair = None

        for i in range(len(active_list)):
            for j in range(i + 1, len(active_list)):
                a, b = active_list[i], active_list[j]
                d = _linkage_distance(matrix, cluster_members[a], cluster_members[b], linkage)
                if d < best_dist:
                    best_dist = d
                    best_pair = (a, b)

        a, b = best_pair
        new_id = next_id
        next_id += 1
        new_members = cluster_members[a] + cluster_members[b]
        cluster_members[new_id] = new_members
        active.discard(a)
        active.discard(b)
        active.add(new_id)

        merges.append({
            "step": step,
            "cluster_a": a,
            "cluster_b": b,
            "new_id": new_id,
            "distance": round(best_dist, 6),
            "size": len(new_members),
        })

        dendro_nodes[new_id] = {
            "id": new_id,
            "leaf": False,
            "distance": round(best_dist, 6),
            "size": len(new_members),
            "children": [dendro_nodes[a], dendro_nodes[b]],
        }

    root_id = next_id - 1
    return {
        "linkage": linkage,
        "n": n,
        "merges": merges,
        "dendro": dendro_nodes[root_id],
    }


def hac_cut(hac_result: dict, n_clusters: int) -> dict:
    """
    Cut the HAC dendrogram to produce exactly n_clusters clusters.

    Returns
    -------
    {
      "n_clusters"  : int,
      "assignments" : [int, ...]   cluster index (0..n_clusters-1) per country
      "clusters"    : [[int,...],] list of country-index lists per cluster
      "cut_distance": float        distance at which the cut was made
    }
    """
    n = hac_result["n"]
    merges = hac_result["merges"]

    if n_clusters < 1 or n_clusters > n:
        raise ValueError(f"n_clusters must be between 1 and {n}")

    # Replay merges and stop when we have n_clusters remaining groups
    # Start: n singleton clusters; each merge reduces count by 1
    # Stop after (n - n_clusters) merges.
    stop_after = n - n_clusters
    cluster_members: dict[int, list[int]] = {i: [i] for i in range(n)}
    active = set(range(n))
    cut_distance = 0.0

    for step_idx, merge in enumerate(merges):
        if step_idx >= stop_after:
            break
        a, b, new_id = merge["cluster_a"], merge["cluster_b"], merge["new_id"]
        cluster_members[new_id] = cluster_members[a] + cluster_members[b]
        active.discard(a)
        active.discard(b)
        active.add(new_id)
        cut_distance = merge["distance"]

    assignments = [0] * n
    clusters = []
    for cluster_idx, cid in enumerate(sorted(active)):
        members = cluster_members[cid]
        clusters.append(members)
        for country_idx in members:
            assignments[country_idx] = cluster_idx

    return {
        "n_clusters": n_clusters,
        "assignments": assignments,
        "clusters": clusters,
        "cut_distance": cut_distance,
    }
