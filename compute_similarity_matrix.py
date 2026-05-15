"""
Precompute pairwise TED normalized similarities for all country tree pairs.
Saves a symmetric N×N matrix to data/similarity_matrix.json.

Usage:
    python compute_similarity_matrix.py
    python compute_similarity_matrix.py --workers 4
    python compute_similarity_matrix.py --tree-dir data/trees_tokens --output data/similarity_matrix.json
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ted_distance import load_tree, ted_distance as compute_ted

try:
    from db.db_config import get_db
    _MONGO_AVAILABLE = True
except Exception:
    _MONGO_AVAILABLE = False


def _worker(args: tuple) -> tuple[int, int, float]:
    """Compute normalized TED similarity for one country pair (worker process)."""
    i, j, path_i, path_j = args
    tree_i = load_tree(Path(path_i))
    tree_j = load_tree(Path(path_j))
    result = compute_ted(tree_i, tree_j)
    return i, j, result["normalized_similarity"]


def _save_json(path: Path, countries: list[str], matrix: list[list[float]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"countries": countries, "matrix": matrix}, f)


def _save_to_mongo(countries, matrix, tree_dir, workers, duration):
    if not _MONGO_AVAILABLE:
        return
    try:
        db = get_db()
        doc = {
            "timestamp":        datetime.now(timezone.utc),
            "type":             "full",
            "source":           "python",
            "duration_seconds": round(duration, 1),
            "num_countries":    len(countries),
            "num_pairs":        len(countries) * (len(countries) - 1) // 2,
            "parameters": {
                "tree_dir":      str(tree_dir),
                "workers":       workers,
                "sorted_trees":  True,
                "numeric_costs": True,
            },
            "countries": countries,
            "matrix":    matrix,
        }
        db.matrix_computations.insert_one(doc)
        print(f"Computation saved to MongoDB (matrix_computations).")
    except Exception as e:
        print(f"MongoDB save skipped: {e}")


def compute_matrix(tree_dir: Path, output_file: Path, workers: int) -> None:
    tree_paths = sorted(tree_dir.glob("*.json"))
    if not tree_paths:
        sys.exit(f"No .json files found in {tree_dir}")

    countries = [p.stem for p in tree_paths]
    n = len(countries)
    total_pairs = n * (n - 1) // 2
    print(f"Found {n} countries -> {total_pairs} unique pairs.")

    # Initialize matrix (diagonal = 1.0, rest = 0.0 until computed)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0

    # Resume from a partial save if one exists
    partial_file = output_file.parent / (output_file.stem + ".partial.json")
    done_pairs: set[tuple[int, int]] = set()

    if partial_file.exists():
        try:
            with open(partial_file, encoding="utf-8") as f:
                partial = json.load(f)
            if partial.get("countries") == countries:
                for i in range(n):
                    for j in range(n):
                        v = partial["matrix"][i][j]
                        if v != 0.0 or i == j:
                            matrix[i][j] = v
                for i in range(n):
                    for j in range(i + 1, n):
                        if partial["matrix"][i][j] != 0.0:
                            done_pairs.add((i, j))
                skipped = len(done_pairs)
                print(f"Resuming: {skipped} pairs restored, {total_pairs - skipped} remaining.")
            else:
                print("Partial file has different country list — ignoring it.")
        except Exception as e:
            print(f"Could not load partial file ({e}) — starting fresh.")

    work = [
        (i, j, str(tree_paths[i]), str(tree_paths[j]))
        for i in range(n)
        for j in range(i + 1, n)
        if (i, j) not in done_pairs
    ]

    if not work:
        print("All pairs already computed. Saving final file.")
        _save_json(output_file, countries, matrix)
        return

    done = len(done_pairs)
    start = time.time()
    SAVE_EVERY = 100  # save partial results every N completed pairs

    def _on_result(i: int, j: int, sim: float) -> None:
        nonlocal done
        matrix[i][j] = sim
        matrix[j][i] = sim
        done += 1

        if done % 10 == 0:
            elapsed = time.time() - start
            rate = (done - len(done_pairs)) / elapsed if elapsed > 0 else 0
            remaining_pairs = total_pairs - done
            eta = remaining_pairs / rate if rate > 0 else float("inf")
            eta_str = f"{eta:.0f}s" if eta < 3600 else f"{eta/3600:.1f}h"
            print(
                f"  [{done}/{total_pairs}] {100*done/total_pairs:.1f}%  "
                f"rate={rate:.1f} pairs/s  eta={eta_str}",
                flush=True,
            )

        if done % SAVE_EVERY == 0:
            _save_json(partial_file, countries, matrix)

    if workers > 1:
        print(f"Using {workers} parallel workers.")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_worker, args): args for args in work}
            for future in as_completed(futures):
                try:
                    i, j, sim = future.result()
                    _on_result(i, j, sim)
                except Exception as e:
                    args = futures[future]
                    print(f"  ERROR for pair ({args[0]}, {args[1]}): {e}")
    else:
        print("Using 1 worker (sequential).")
        for args in work:
            try:
                i, j, sim = _worker(args)
                _on_result(i, j, sim)
            except Exception as e:
                print(f"  ERROR for pair ({args[0]}, {args[1]}): {e}")

    duration = time.time() - start
    _save_json(output_file, countries, matrix)
    print(f"\nDone in {duration:.1f}s. Matrix saved to: {output_file}")

    if partial_file.exists():
        partial_file.unlink()

    _save_to_mongo(countries, matrix, tree_dir, workers, duration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute pairwise TED similarity matrix.")
    parser.add_argument(
        "--tree-dir",
        default="data/trees_tokens",
        type=Path,
        help="Directory with tokenized tree JSON files (default: data/trees_tokens)",
    )
    parser.add_argument(
        "--output",
        default="data/similarity_matrix.json",
        type=Path,
        help="Output file path (default: data/similarity_matrix.json)",
    )
    parser.add_argument(
        "--workers",
        default=1,
        type=int,
        help="Number of parallel worker processes (default: 1). Use 4-8 to speed up.",
    )
    args = parser.parse_args()

    compute_matrix(args.tree_dir, args.output, args.workers)
