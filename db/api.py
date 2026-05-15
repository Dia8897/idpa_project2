"""
Minimal Flask API that lets the browser save results to MongoDB.

Run from the idpa_project2/ folder:
    py db/api.py

Endpoints:
    POST /api/clustering        — save a clustering run
    GET  /api/clustering        — list recent runs (no assignments)
    POST /api/matrix            — save a browser-computed feature matrix
    GET  /api/matrix            — list recent matrix computations
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, jsonify, request
from flask_cors import CORS
from db_config import get_db
from save_clustering import get_recent_runs, save_run

app = Flask(__name__)
CORS(app)  # allow requests from the frontend (localhost:8000)


@app.post("/api/clustering")
def save_clustering():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "No JSON body"}), 400

    algorithm  = body.get("algorithm")
    parameters = body.get("parameters", {})
    assignments = body.get("assignments", {})
    metrics    = body.get("metrics", {})

    if not algorithm or not assignments:
        return jsonify({"error": "algorithm and assignments are required"}), 400

    inserted_id = save_run(algorithm, parameters, assignments, metrics)
    return jsonify({"saved": True, "id": inserted_id}), 201


@app.get("/api/clustering")
def list_clustering():
    limit = int(request.args.get("limit", 10))
    runs  = get_recent_runs(limit)
    return jsonify(runs)


@app.get("/api/matrix/full")
def get_full_matrix():
    """
    Return the full 194×194 similarity matrix.
    First call reads the JSON file and seeds MongoDB — every subsequent call
    is served directly from MongoDB (no file I/O).
    """
    import json
    from pathlib import Path

    db = get_db()
    doc = db.matrix_computations.find_one({"type": "full"}, {"_id": 0})
    if doc:
        if "timestamp" in doc:
            doc["timestamp"] = doc["timestamp"].isoformat()
        return jsonify({"countries": doc["countries"], "matrix": doc["matrix"]})

    # Not in MongoDB yet — seed it from the JSON file
    matrix_file = Path(__file__).parent.parent / "data" / "similarity_matrix.json"
    if not matrix_file.exists():
        return jsonify({"error": "similarity_matrix.json not found"}), 404

    with open(matrix_file, encoding="utf-8") as f:
        data = json.load(f)

    seed_doc = {
        "timestamp":     __import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        "type":          "full",
        "source":        "json_file",
        "num_countries": len(data["countries"]),
        "num_pairs":     len(data["countries"]) * (len(data["countries"]) - 1) // 2,
        "attributes":    [],
        "countries":     data["countries"],
        "matrix":        data["matrix"],
    }
    db.matrix_computations.insert_one(seed_doc)
    print("Full matrix seeded into MongoDB (matrix_computations).")
    return jsonify({"countries": data["countries"], "matrix": data["matrix"]})


@app.get("/api/matrix/lookup")
def lookup_matrix():
    """
    Check if a feature matrix with exactly these attributes already exists.
    Query: /api/matrix/lookup?attrs=gdp_nominal,total,water,...
    Returns the full matrix doc if found, 404 otherwise.
    """
    attrs_param = request.args.get("attrs", "")
    if not attrs_param:
        return jsonify({"error": "attrs query param required"}), 400

    requested = sorted(attrs_param.split(","))

    db = get_db()
    doc = db.matrix_computations.find_one(
        {"attributes": requested},
        {"_id": 0}   # exclude _id so JSON serialization is simple
    )
    if doc:
        # convert datetime to string so Flask can serialize it
        if "timestamp" in doc:
            doc["timestamp"] = doc["timestamp"].isoformat()
        return jsonify(doc)
    return jsonify(None), 404


@app.post("/api/matrix")
def save_matrix():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "No JSON body"}), 400

    matrix_type = body.get("type", "feature")   # "feature" or "full"
    countries   = body.get("countries", [])
    matrix      = body.get("matrix", [])
    attributes  = body.get("attributes", [])     # only for feature matrices

    if not countries or not matrix:
        return jsonify({"error": "countries and matrix are required"}), 400

    db = get_db()
    doc = {
        "timestamp":   __import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        "type":        matrix_type,
        "source":      "browser",
        "num_countries": len(countries),
        "num_pairs":   len(countries) * (len(countries) - 1) // 2,
        "attributes":  sorted(attributes),
        "countries":   countries,
        "matrix":      matrix,
    }
    result = db.matrix_computations.insert_one(doc)
    return jsonify({"saved": True, "id": str(result.inserted_id)}), 201


@app.get("/api/matrix")
def list_matrices():
    db    = get_db()
    limit = int(request.args.get("limit", 10))
    docs  = list(
        db.matrix_computations
        .find({}, {"matrix": 0, "countries": 0})   # exclude large fields
        .sort("timestamp", -1)
        .limit(limit)
    )
    for d in docs:
        d["_id"] = str(d["_id"])
    return jsonify(docs)


@app.get("/api/trees")
def get_all_trees():
    """Return all country trees in one request."""
    db   = get_db()
    docs = list(db.countries.find({}, {"_id": 0, "name": 1, "tokenized_tree": 1, "display_tree": 1}))
    return jsonify(docs)


@app.get("/api/features")
def get_features():
    """
    Return the list of top-level attribute names and how many countries have each one.
    Uses a MongoDB aggregation — no tree loading needed on the client side.
    Response: [{"label": "capital", "count": 191}, ...]
    """
    db = get_db()
    pipeline = [
        {"$unwind": "$tokenized_tree.tree.children"},
        {"$group": {"_id": "$tokenized_tree.tree.children.label", "count": {"$sum": 1}}},
        {"$sort": {"count": -1, "_id": 1}},
        {"$project": {"_id": 0, "label": "$_id", "count": 1}},
    ]
    features = list(db.countries.aggregate(pipeline))
    return jsonify(features)


@app.get("/api/mds")
def get_mds():
    """Return cached MDS coordinates for the full matrix."""
    db  = get_db()
    doc = db.mds_cache.find_one({"type": "full"}, {"_id": 0})
    if doc:
        return jsonify({"x": doc["x"], "y": doc["y"]})
    return jsonify(None), 404


@app.post("/api/mds")
def save_mds():
    """Cache MDS coordinates for the full matrix."""
    body = request.get_json(silent=True)
    if not body or "x" not in body or "y" not in body:
        return jsonify({"error": "x and y required"}), 400
    db = get_db()
    db.mds_cache.replace_one(
        {"type": "full"},
        {"type": "full", "x": body["x"], "y": body["y"]},
        upsert=True,
    )
    return jsonify({"saved": True}), 201


if __name__ == "__main__":
    app.run(port=5050, debug=True)
