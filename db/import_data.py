"""
Reads all existing JSON files and the similarity matrix,
then inserts everything into MongoDB.

Run from the idpa_project2/ folder:
    py db/import_data.py
"""

import json
import sys
from pathlib import Path

ROOT        = Path(__file__).parent.parent
DISPLAY_DIR = ROOT / "data" / "trees"
TOKEN_DIR   = ROOT / "data" / "trees_tokens"
MATRIX_FILE = ROOT / "data" / "similarity_matrix.json"

sys.path.insert(0, str(Path(__file__).parent))
from db_config import get_db


def import_countries(db):
    """
    One document per country. Embeds both trees directly inside the document.

    Document shape:
    {
        "name": "Lebanon",
        "display_tree":   { ...full tree from data/trees/Lebanon.json ... },
        "tokenized_tree": { ...full tree from data/trees_tokens/Lebanon.json ... }
    }
    """
    names = sorted(p.stem for p in TOKEN_DIR.glob("*.json"))
    if not names:
        sys.exit(f"No tokenized tree files found in {TOKEN_DIR}")

    print(f"Found {len(names)} countries. Importing country documents...")

    inserted = updated = skipped = 0
    for name in names:
        token_path   = TOKEN_DIR   / f"{name}.json"
        display_path = DISPLAY_DIR / f"{name}.json"

        if not token_path.exists():
            print(f"  Missing tokenized tree for {name}, skipping.")
            skipped += 1
            continue

        with open(token_path, encoding="utf-8") as f:
            tokenized = json.load(f)

        display = None
        if display_path.exists():
            with open(display_path, encoding="utf-8") as f:
                display = json.load(f)

        doc = {
            "name":           name,
            "display_tree":   display,
            "tokenized_tree": tokenized,
        }

        result = db.countries.update_one(
            {"name": name},
            {"$set": doc},
            upsert=True,
        )
        if result.upserted_id:
            inserted += 1
        else:
            updated += 1

    print(f"  Inserted: {inserted}  Updated: {updated}  Skipped: {skipped}")


def import_similarity(db):
    """
    One document per ordered pair (a, b).  Both (a,b) and (b,a) are stored
    so lookups by either country are simple.

    Document shape:
    {
        "country_a": "Lebanon",
        "country_b": "Switzerland",
        "score": 0.312847
    }
    """
    if not MATRIX_FILE.exists():
        print(f"Similarity matrix not found at {MATRIX_FILE}, skipping.")
        return

    print("Importing similarity matrix...")

    with open(MATRIX_FILE, encoding="utf-8") as f:
        data = json.load(f)

    countries = data["countries"]
    matrix    = data["matrix"]

    ops = []
    for i, name_a in enumerate(countries):
        for j, name_b in enumerate(countries):
            ops.append({
                "country_a": name_a,
                "country_b": name_b,
                "score":     float(matrix[i][j]),
            })

    # Upsert in batches of 5000
    batch_size = 5000
    total = 0
    for start in range(0, len(ops), batch_size):
        batch = ops[start : start + batch_size]
        for doc in batch:
            db.similarity.update_one(
                {"country_a": doc["country_a"], "country_b": doc["country_b"]},
                {"$set": doc},
                upsert=True,
            )
        total += len(batch)
        print(f"  {total}/{len(ops)} pairs...", end="\r")

    print(f"\n  Done — {total} similarity documents inserted/updated.")


def main():
    db = get_db()
    import_countries(db)
    import_similarity(db)
    print("Import complete.")


if __name__ == "__main__":
    main()
