import argparse
import json
import math
import re
from functools import lru_cache
from pathlib import Path


DEFAULT_TREE_DIR = Path("data/trees_tokens")
_ACTIVE_SOURCE_SUBTREES = frozenset()
_ACTIVE_TARGET_SUBTREES = frozenset()

# Attribute frequency weights (label → count across all country trees).
# Set via set_attr_weights() before computing the full similarity matrix.
# When empty (default), all attributes get weight 1.0 (original behaviour).
_ATTR_WEIGHTS: dict[str, int] = {}
_N_COUNTRIES: int = 1


def set_attr_weights(attr_freq: dict[str, int], n: int) -> None:
    """Pre-load per-attribute frequency counts used by the top-level TED DP."""
    global _ATTR_WEIGHTS, _N_COUNTRIES
    _N_COUNTRIES = max(n, 1)
    _ATTR_WEIGHTS = dict(attr_freq)


def _attr_weight(label: str) -> float:
    """Coverage-based weight in [0.01, 1] for a depth-1 (attribute) node."""
    if not _ATTR_WEIGHTS:
        return 1.0
    coverage = _ATTR_WEIGHTS.get(label, 0) / _N_COUNTRIES
    return max(0.01, coverage)


def round3(value: float) -> float:
    return round(float(value), 3)


def load_tree(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        tree = json.load(f)["tree"]
    return sort_tree(tree)


def sort_tree(node: dict) -> dict:
    """
    Canonicalize structure order while preserving value-leaf order.

    Rule:
    - sort only non-leaf children by label (structural keys/attributes)
    - keep leaf children (values/tokens) in their original source order
    """
    original_children = node.get("children", [])
    structural_sorted = sorted(
        (c for c in original_children if not is_leaf(c)),
        key=lambda c: node_label(c),
    )
    structural_iter = iter(structural_sorted)

    merged = [child if is_leaf(child) else next(structural_iter) for child in original_children]
    return {**node, "children": [sort_tree(c) for c in merged]}


def node_label(node: dict) -> str:
    return str(node.get("label", ""))
    # {"label": "capital", "children": [...]}
# This function extracts "capital".


def is_leaf(node: dict) -> bool:
    return len(node.get("children", [])) == 0


def clone_serializable(node: dict) -> dict:
    """
    Only keep the fields that matter to TED itself.
    Metadata such as raw_values is ignored for the distance computation.
    """
    return {
        "label": node_label(node),
        "children": [clone_serializable(child) for child in node.get("children", [])],
    }
    # This removes anything extra from the node.


def serialize_node(node: dict) -> str:
    return json.dumps(clone_serializable(node), ensure_ascii=False, sort_keys=True)
# This turns a subtree into a JSON string.


def collect_subtree_serials(node: dict) -> set[str]:
    serials = {serialize_node(node)}
    for child in node.get("children", []):
        serials.update(collect_subtree_serials(child))
    return serials


def contained_in_source_tree(node: dict) -> bool:
    return serialize_node(node) in _ACTIVE_SOURCE_SUBTREES


def contained_in_target_tree(node: dict) -> bool:
    return serialize_node(node) in _ACTIVE_TARGET_SUBTREES


def subtree_size(node: dict) -> int:
    return 1 + sum(subtree_size(child) for child in node.get("children", []))
# This counts how many nodes are inside a subtree.

def cost_del_tree(node: dict) -> int:
    """
    Subtree deletion cost:
    - 1 if the exact subtree already exists in the destination tree
    - subtree size otherwise
    """
    return 1 if contained_in_target_tree(node) else subtree_size(node)


def cost_ins_tree(node: dict) -> int:
    """
    Subtree insertion cost:
    - 1 if the exact subtree already exists in the source tree
    - subtree size otherwise
    """
    return 1 if contained_in_source_tree(node) else subtree_size(node)


def _try_numeric(label: str) -> float | None:
    """Parse a label as a number, stripping commas and optional trailing '%'."""
    text = str(label).strip().replace(",", "")
    m = re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?%?", text)
    if not m:
        return None
    try:
        value = float(m.group(0).rstrip("%"))
        return value if math.isfinite(value) else None
    except ValueError:
        return None


def _numeric_leaf_cost(label_a: str, label_b: str) -> float | None:
    """Return proportional distance in [0,1] if both labels are numeric, else None."""
    va = _try_numeric(label_a)
    vb = _try_numeric(label_b)
    if va is None or vb is None:
        return None
    scale = max(abs(va), abs(vb), 1.0)
    return max(0.0, min(1.0, abs(va - vb) / scale))


def cost_upd_root(a: dict, b: dict) -> float:
    """
    Root update cost.

    Project rule:
    - identical labels => 0
    - different leaf labels => numeric proportional cost if both numeric, else 1
    - different internal labels => replace whole subtree rather than rename it
    """
    if node_label(a) == node_label(b):
        return 0.0
    if is_leaf(a) and is_leaf(b):
        numeric_cost = _numeric_leaf_cost(node_label(a), node_label(b))
        if numeric_cost is not None:
            return numeric_cost
        return 1.0
    return float(cost_del_tree(a) + cost_ins_tree(b))


@lru_cache(maxsize=None)
def ted(a_serial: str, b_serial: str) -> float:
    """
    Ordered tree edit distance based on the subtree-DP recurrence in the provided
    algorithm sketch.

    Dist[i][j] stores the best cost for transforming the first i children of A
    into the first j children of B, after accounting for the root update cost.
    """
    a = json.loads(a_serial)
    b = json.loads(b_serial)

    if node_label(a) != node_label(b) and not (is_leaf(a) and is_leaf(b)):
        return cost_del_tree(a) + cost_ins_tree(b)

    a_children = a.get("children", [])
    b_children = b.get("children", [])
    m = len(a_children)
    n = len(b_children)

    dist = [[0] * (n + 1) for _ in range(m + 1)]
    dist[0][0] = cost_upd_root(a, b)

    for i in range(1, m + 1):
        dist[i][0] = dist[i - 1][0] + cost_del_tree(a_children[i - 1])

    for j in range(1, n + 1):
        dist[0][j] = dist[0][j - 1] + cost_ins_tree(b_children[j - 1])

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dist[i][j] = min(
                dist[i - 1][j - 1] + ted(serialize_node(a_children[i - 1]), serialize_node(b_children[j - 1])),
                dist[i - 1][j] + cost_del_tree(a_children[i - 1]),
                dist[i][j - 1] + cost_ins_tree(b_children[j - 1]),
            )

    return dist[m][n]


def normalized_similarity(distance: float, size1: int, size2: int) -> float:
    """
    Convert the distance into an intuitive similarity score in [0,1].
    """
    total = size1 + size2
    if total == 0:
        return 1.0
    return max(0.0, 1.0 - (distance / total))


def slide_similarity_formula1(distance: int) -> float:
    """
    Slide formula 1:
    Sim(A, B) = 1 / (1 + TED(A, B))
    """
    return 1.0 / (1.0 + distance)


def configure_ted_context(tree1: dict, tree2: dict):
    """
    Prepare subtree-membership state and clear the TED cache so callers that need
    both TED values and TED-based backtracking can share the same tested logic.
    """
    global _ACTIVE_SOURCE_SUBTREES, _ACTIVE_TARGET_SUBTREES

    _ACTIVE_SOURCE_SUBTREES = frozenset(collect_subtree_serials(tree1))
    _ACTIVE_TARGET_SUBTREES = frozenset(collect_subtree_serials(tree2))
    ted.cache_clear()


def ted_distance(tree1: dict, tree2: dict):
    configure_ted_context(tree1, tree2)

    a, b  = tree1, tree2
    size1 = subtree_size(a)
    size2 = subtree_size(b)

    # Top-level DP — mirrors makeTedContext() in cluster.html.
    # Depth-1 children (direct attributes of the root) are weighted by
    # _attr_weight(); deeper nodes use unit cost (weight = 1.0).
    if node_label(a) != node_label(b) and not (is_leaf(a) and is_leaf(b)):
        raw_dist = cost_del_tree(a) + cost_ins_tree(b)
    else:
        a_ch = a.get("children", [])
        b_ch = b.get("children", [])
        m, nc = len(a_ch), len(b_ch)

        dist = [[0.0] * (nc + 1) for _ in range(m + 1)]
        dist[0][0] = cost_upd_root(a, b)

        for i in range(1, m + 1):
            c = a_ch[i - 1]
            dist[i][0] = dist[i - 1][0] + cost_del_tree(c) * _attr_weight(node_label(c))
        for j in range(1, nc + 1):
            c = b_ch[j - 1]
            dist[0][j] = dist[0][j - 1] + cost_ins_tree(c) * _attr_weight(node_label(c))

        for i in range(1, m + 1):
            for j in range(1, nc + 1):
                ci, cj = a_ch[i - 1], b_ch[j - 1]
                dist[i][j] = min(
                    dist[i - 1][j - 1] + ted(serialize_node(ci), serialize_node(cj)),
                    dist[i - 1][j]     + cost_del_tree(ci) * _attr_weight(node_label(ci)),
                    dist[i][j - 1]     + cost_ins_tree(cj) * _attr_weight(node_label(cj)),
                )
        raw_dist = dist[m][nc]

    distance = round3(raw_dist)

    # Whole-tree normalized similarity: 1 - TED / (|C| + |D|).
    # Matches computeTedSim() in cluster.html — fair comparison regardless of
    # tree size differences between countries.
    similarity = round3(normalized_similarity(distance, size1, size2))
    similarity_formula1 = round3(slide_similarity_formula1(distance))
    common_score = round3((size1 + size2 - distance) / 2)

    return {
        "tree_dir":                  str(DEFAULT_TREE_DIR),
        "size1":                     size1,
        "size2":                     size2,
        "distance":                  distance,
        "common_score":              common_score,
        "slide_similarity_formula1": similarity_formula1,
        "normalized_similarity":     similarity,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute ordered TED on tokenized country trees.")
    parser.add_argument("source", nargs="?", default="Lebanon.json", help="Source tree file name or country name")
    parser.add_argument("target", nargs="?", default="Switzerland.json", help="Target tree file name or country name")
    parser.add_argument(
        "--tree-dir",
        default=DEFAULT_TREE_DIR,
        type=Path,
        help="Directory containing tokenized tree JSON files (default: data/trees_tokens)",
    )
    args = parser.parse_args()

    source_name = Path(args.source).name if str(args.source).endswith(".json") else f"{args.source}.json"
    target_name = Path(args.target).name if str(args.target).endswith(".json") else f"{args.target}.json"
    source_path = args.tree_dir / source_name
    target_path = args.tree_dir / target_name

    if not source_path.exists() or not target_path.exists():
        raise SystemExit(f"Missing tree file(s): {source_path} or {target_path}")

    tree1 = load_tree(source_path)
    tree2 = load_tree(target_path)
    result = ted_distance(tree1, tree2)

    print("Tree directory:", result["tree_dir"])
    print("Size1:", result["size1"], "| Size2:", result["size2"])
    print("Distance:", result["distance"])
    print("Common score:", result["common_score"])
    print("Slide similarity formula 1:", f"{result['slide_similarity_formula1']:.3f}")
    print("Normalized similarity:", f"{result['normalized_similarity']:.3f}")
