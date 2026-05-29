#!/usr/bin/env python3
"""
filter_isa_edges.py
-------------------
Filters isa_edges.csv to keep only clinically meaningful ancestors.

Two filters applied:
  1. Semantic type filter: keep only T047 (Disease or Syndrome) and
     T048 (Mental or Behavioral Dysfunction) parent nodes.
     Removes findings, morphologies, generic attributes.

  2. Blocklist filter: remove known overly broad UMLS terms
     regardless of semantic type.

Output:
  output/isa_edges_filtered.csv   (clean edges for Neo4j)
  output/isa_edges_removed.csv    (removed edges for audit)
  output/filter_report.txt        (summary)

Usage:
    python kg/symptom_diagnosis/filter_isa_edges.py \
        --isa_edges  kg/symptom_diagnosis/output/isa_edges.csv \
        --mrsty      kg/symptom_diagnosis/MRSTY.RRF \
        --out_dir    kg/symptom_diagnosis/output
"""

import os
import csv
import argparse
import logging
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Semantic types we KEEP as valid clinical family nodes
# T047 = Disease or Syndrome
# T048 = Mental or Behavioral Dysfunction
# T191 = Neoplastic Process (cancers)
# T046 = Pathologic Function
# ─────────────────────────────────────────────────────────────────────────────
KEEP_SEMTYPES = {"T047", "T048", "T191", "T046"}

# ─────────────────────────────────────────────────────────────────────────────
# Blocklist: overly broad terms that slip through semantic type filter
# Add any others you spot in the output
# ─────────────────────────────────────────────────────────────────────────────
BLOCKLIST_SUBSTRINGS = [
    "disease",                          # bare "Disease"
    "disorder",                         # bare "Disorder"
    "finding",                          # "Finding of..."
    "abnormal",                         # "Abnormal findings..."
    "general convenience",
    "disease attributes",
    "disease terms",
    "other specified",
    "other diseases",
    "unspecified",
    "not elsewhere",
    "nec",
    "nos",
    "morphology",
    "procedure",
    "observable",
    "qualifier",
    "attribute",
    "situation",
    "event",
    "regime",
    "administration",
    "specimen",
    "environment",
    "context",
    "navigational",
    "record artifact",
    "icd",
    "snomed",
    "physical illness",
    "cancer-related",
    "allergic condition",
    "hypersensitivity condition",
    "body as a whole",
    "general terms",
    "general conditions",
    "general disorders",
    "allied conditions",
    "diseases and injuries",
    "infectious and parasitic",
    "resistance mechanism",
]

# Names that are EXACTLY too broad (full match, case insensitive)
BLOCKLIST_EXACT = {
    "disease",
    "disorder",
    "finding",
    "condition",
    "syndrome",
    "illness",
    "problem",
    "complaint",
    "symptom",
    "sign",
}


def load_semtypes(mrsty_path: str) -> dict:
    """
    Returns cui -> set of semantic type codes.
    MRSTY columns: CUI|TUI|STN|STY|ATUI|CVF
    """
    log.info("Loading semantic types from MRSTY ...")
    cui_to_tuis = defaultdict(set)
    with open(mrsty_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            cui = parts[0]
            tui = parts[1]
            cui_to_tuis[cui].add(tui)
    log.info("  Loaded semantic types for %s CUIs", f"{len(cui_to_tuis):,}")
    return dict(cui_to_tuis)


def is_blocked_by_name(name: str) -> bool:
    """Returns True if the name is too broad to be a useful family node."""
    name_lower = name.lower().strip()

    # ALL_CAPS = ICD/MeSH chapter heading — always too broad
    stripped = name.strip()
    if stripped == stripped.upper() and len(stripped) > 3 and stripped.replace(" ","").replace(",","").replace("&","").replace("/","").replace("-","").isalpha():
        return True

    # Exact match
    if name_lower in BLOCKLIST_EXACT:
        return True

    # Substring match — only block if the name IS the generic term
    # e.g. block "Disease" but not "Cardiovascular Disease"
    # We block if the name has fewer than 3 words and contains a blocklist term
    words = name_lower.split()
    if len(words) <= 2:
        for substr in BLOCKLIST_SUBSTRINGS:
            if substr in name_lower:
                return True

    # Always block these patterns regardless of length
    always_block = [
        "general convenience",
        "disease attributes",
        "disease terms",
        "not elsewhere",
        "navigational",
        "record artifact",
        "icd-",
        "snomed",
    ]
    for pattern in always_block:
        if pattern in name_lower:
            return True

    return False


def filter_edges(
    isa_edges_path: str,
    cui_to_tuis: dict,
) -> tuple[list, list]:
    """
    Returns (kept_edges, removed_edges).
    Each edge is a dict with keys: child_cui, child_name, parent_cui, parent_name.
    """
    kept, removed = [], []

    with open(isa_edges_path) as f:
        for row in csv.DictReader(f):
            parent_cui  = row["parent_cui"]
            parent_name = row["parent_name"]

            # Filter 1: semantic type
            tuis = cui_to_tuis.get(parent_cui, set())
            passes_semtype = bool(tuis & KEEP_SEMTYPES)

            # Filter 2: name blocklist
            passes_name = not is_blocked_by_name(parent_name)

            if passes_semtype and passes_name:
                kept.append(row)
            else:
                reason = []
                if not passes_semtype:
                    reason.append(f"semtype={tuis or 'NONE'}")
                if not passes_name:
                    reason.append("blocklist_name")
                row["filter_reason"] = "; ".join(reason)
                removed.append(row)

    return kept, removed


def write_csv(path: str, rows: list, extra_fields: list = None):
    if not rows:
        log.info("  No rows to write for %s", path)
        return
    fields = list(rows[0].keys())
    if extra_fields:
        fields += [f for f in extra_fields if f not in fields]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    log.info("  Written %d rows -> %s", len(rows), path)


def print_report(kept: list, removed: list, out_dir: str):
    # Count unique parents
    kept_parents    = {r["parent_cui"] for r in kept}
    removed_parents = {r["parent_cui"] for r in removed}

    # Group kept parents by child for chain display
    child_to_parents = defaultdict(list)
    for r in kept:
        child_to_parents[r["child_name"]].append(r["parent_name"])

    lines = [
        "=" * 65,
        "IS_A FILTER REPORT",
        "=" * 65,
        f"  Total edges before filter : {len(kept) + len(removed)}",
        f"  Edges kept                : {len(kept)}",
        f"  Edges removed             : {len(removed)}",
        f"  Unique parents kept       : {len(kept_parents)}",
        f"  Unique parents removed    : {len(removed_parents)}",
        "",
        "  Cleaned hierarchy chains:",
    ]

    for child_name, parents in sorted(child_to_parents.items()):
        lines.append(f"    {child_name}")
        for p in sorted(set(parents)):
            lines.append(f"      -> {p}")

    report = "\n".join(lines)
    print(report)

    path = os.path.join(out_dir, "filter_report.txt")
    with open(path, "w") as f:
        f.write(report)
    log.info("  Report saved -> %s", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isa_edges", required=True,
                    help="kg/symptom_diagnosis/output/isa_edges.csv")
    ap.add_argument("--mrsty",     required=True,
                    help="kg/symptom_diagnosis/MRSTY.RRF")
    ap.add_argument("--out_dir",   default="kg/symptom_diagnosis/output")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cui_to_tuis = load_semtypes(args.mrsty)

    log.info("Filtering IS_A edges ...")
    kept, removed = filter_edges(args.isa_edges, cui_to_tuis)

    write_csv(
        os.path.join(args.out_dir, "isa_edges_filtered.csv"),
        kept,
    )
    write_csv(
        os.path.join(args.out_dir, "isa_edges_removed.csv"),
        removed,
        extra_fields=["filter_reason"],
    )

    print_report(kept, removed, args.out_dir)

    log.info("=" * 65)
    log.info("Next step: review filter_report.txt")
    log.info("If chains look clinically correct, run build_dx_hierarchy.py")
    log.info("with --isa_csv output/isa_edges_filtered.csv (no --dry_run)")
    log.info("=" * 65)


if __name__ == "__main__":
    main()