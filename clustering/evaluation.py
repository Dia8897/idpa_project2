"""
Cluster evaluation metrics using a precomputed distance matrix.

Silhouette Score  : measures how well each point fits its cluster vs. others. Range [-1, 1].
Davies-Bouldin Index : measures average cluster scatter vs. separation. Lower is better.
"""


def _dist(matrix: list, i: int, j: int) -> float:
    return 1.0 - matrix[i][j]


def silhouette_score(matrix: list, assignments: list[int]) -> dict:
    """
    Compute per-point and overall silhouette scores.

    s(i) = (b(i) - a(i)) / max(a(i), b(i))
      a(i) = avg distance to points in same cluster
      b(i) = avg distance to points in nearest other cluster
    """
    n = len(assignments)
    k = max(assignments) + 1

    clusters = [[] for _ in range(k)]
    for i, c in enumerate(assignments):
        clusters[c].append(i)

    scores = []
    for i in range(n):
        ci = assignments[i]
        same = [j for j in clusters[ci] if j != i]

        if len(same) == 0:
            scores.append(0.0)
            continue

        a = sum(_dist(matrix, i, j) for j in same) / len(same)

        b = float("inf")
        for c in range(k):
            if c == ci:
                continue
            if not clusters[c]:
                continue
            avg = sum(_dist(matrix, i, j) for j in clusters[c]) / len(clusters[c])
            if avg < b:
                b = avg

        if b == float("inf"):
            scores.append(0.0)
        else:
            denom = max(a, b)
            scores.append((b - a) / denom if denom > 0 else 0.0)

    overall = sum(scores) / len(scores) if scores else 0.0
    per_cluster = []
    for c in range(k):
        members = clusters[c]
        avg = sum(scores[i] for i in members) / len(members) if members else 0.0
        per_cluster.append(round(avg, 4))

    return {
        "overall": round(overall, 4),
        "per_point": [round(s, 4) for s in scores],
        "per_cluster": per_cluster,
    }


def davies_bouldin_index(matrix: list, assignments: list[int]) -> float:
    """
    Davies-Bouldin Index. Lower values indicate better clustering.

    DB = (1/k) * sum_i max_{j!=i} [ (s_i + s_j) / d(c_i, c_j) ]
      s_i  = avg distance from cluster i's points to their centroid (medoid)
      d(ci, cj) = distance between cluster centroids (medoids)
    """
    n = len(assignments)
    k = max(assignments) + 1

    clusters = [[] for _ in range(k)]
    for i, c in enumerate(assignments):
        clusters[c].append(i)

    # Find medoid of each cluster (point with min avg distance to others)
    medoids = []
    scatter = []
    for c in range(k):
        members = clusters[c]
        if len(members) == 1:
            medoids.append(members[0])
            scatter.append(0.0)
            continue
        best = min(members, key=lambda p: sum(_dist(matrix, p, q) for q in members if q != p))
        medoids.append(best)
        avg_dist = sum(_dist(matrix, best, q) for q in members if q != best) / (len(members) - 1)
        scatter.append(avg_dist)

    db_sum = 0.0
    for i in range(k):
        max_ratio = 0.0
        for j in range(k):
            if i == j:
                continue
            inter = _dist(matrix, medoids[i], medoids[j])
            if inter == 0:
                continue
            ratio = (scatter[i] + scatter[j]) / inter
            if ratio > max_ratio:
                max_ratio = ratio
        db_sum += max_ratio

    return round(db_sum / k, 4)


def evaluate(matrix: list, assignments: list[int]) -> dict:
    """Run both metrics and return combined results."""
    sil = silhouette_score(matrix, assignments)
    db = davies_bouldin_index(matrix, assignments)
    return {
        "silhouette": sil,
        "davies_bouldin": db,
    }
