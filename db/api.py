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


if __name__ == "__main__":
    app.run(port=5050, debug=True)
