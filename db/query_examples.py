"""
Example queries against the MongoDB database.
Run from the idpa_project2/ folder:
    py db/query_examples.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from db_config import get_db

db = get_db()


# ── 1. Fetch one country document (name + tree structure) ─────────────────────
print("=" * 60)
print("1. Lebanon document (top-level keys)")
lebanon = db.countries.find_one({"name": "Lebanon"})
print("   Keys:", list(lebanon.keys()))
print("   Tokenized tree root label:", lebanon["tokenized_tree"]["tree"]["label"])
print("   Number of top-level attributes:",
      len(lebanon["tokenized_tree"]["tree"]["children"]))


# ── 2. List all top-level attributes of a country ────────────────────────────
print("\n2. Lebanon's tokenized tree attributes (sorted):")
attrs = sorted(
    child["label"]
    for child in lebanon["tokenized_tree"]["tree"]["children"]
)
for a in attrs:
    print("   -", a)


# ── 3. Get similarity between two specific countries ─────────────────────────
print("\n3. Similarity: Lebanon ↔ Switzerland")
pair = db.similarity.find_one({"country_a": "Lebanon", "country_b": "Switzerland"})
print(f"   Score: {pair['score']:.4f}")


# ── 4. Top 5 most similar countries to Lebanon ───────────────────────────────
print("\n4. Top 5 most similar countries to Lebanon:")
top5 = (
    db.similarity
    .find({"country_a": "Lebanon", "country_b": {"$ne": "Lebanon"}})
    .sort("score", -1)
    .limit(5)
)
for doc in top5:
    print(f"   {doc['country_b']:<30} {doc['score']:.4f}")


# ── 5. Countries that have a specific attribute ───────────────────────────────
print("\n5. How many countries have a 'gdp_nominal' attribute in their tokenized tree:")
count = db.countries.count_documents({
    "tokenized_tree.tree.children.label": "gdp_nominal"
})
print(f"   {count} countries")


# ── 6. Total documents in each collection ────────────────────────────────────
print("\n6. Collection sizes:")
print(f"   countries  : {db.countries.count_documents({})} documents")
print(f"   similarity : {db.similarity.count_documents({})} documents")
