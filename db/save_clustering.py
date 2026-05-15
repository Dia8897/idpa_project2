"""
Utility to save a clustering run to MongoDB.
Called directly by the Flask API (api.py).

A clustering run document looks like:
{
    "timestamp":   <datetime>,
    "algorithm":   "kmedoids" | "hac",
    "parameters":  { "k": 5, "seed": 42, "linkage": "ward", "feature_mode": "all", ... },
    "assignments": { "Lebanon": 0, "Switzerland": 2, ... },
    "num_clusters": 5,
    "metrics": {
        "silhouette":      0.45,
        "davies_bouldin":  1.2,
        "purity":          0.61,
        "precision":       0.50,
        "recall":          0.55,
        "f_measure":       0.52
    }
}
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from db_config import get_db


def save_run(algorithm: str, parameters: dict, assignments: dict,
             metrics: dict | None = None) -> str:
    """
    Insert one clustering run into the clustering_runs collection.
    Returns the inserted document id as a string.
    """
    db = get_db()

    num_clusters = len(set(assignments.values())) if assignments else 0

    doc = {
        "timestamp":    datetime.now(timezone.utc),
        "algorithm":    algorithm,
        "parameters":   parameters,
        "assignments":  assignments,
        "num_clusters": num_clusters,
        "metrics":      metrics or {},
    }

    result = db.clustering_runs.insert_one(doc)
    return str(result.inserted_id)


def get_recent_runs(limit: int = 10) -> list[dict]:
    """Return the most recent clustering runs (without the full assignments to keep it light)."""
    db = get_db()
    cursor = (
        db.clustering_runs
        .find({}, {"assignments": 0})
        .sort("timestamp", -1)
        .limit(limit)
    )
    runs = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        runs.append(doc)
    return runs
