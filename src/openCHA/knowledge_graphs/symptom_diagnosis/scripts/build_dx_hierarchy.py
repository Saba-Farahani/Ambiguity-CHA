#!/usr/bin/env python3
"""
build_dx_hierarchy.py
---------------------
Extracts IS_A (parent-child) edges from UMLS MRREL for diagnosis CUIs
already present in DxSxKG, then writes them to:
  - output/isa_edges.csv          (for audit / offline use)
  - Neo4j  (DxNode)-[:IS_A]->(DxNode) edges

This replaces the post-hoc manual diagnosis-family mapping with a
graph-native clinical hierarchy that the Orchestrator can use for
two-level hypothesis reduction:
    Level 1: collapse hypothesis set to clinical family (parent node)
    Level 2: within-family disambiguation using HAS_SYMPTOM edges

Because IS_A edges come from UMLS itself (not from benchmark data),
this step requires NO benchmark split — it is fully data-free and
can be justified analytically, which also resolves the T_max
validation-tuning issue (see orchestrator_hierarchical.py).

Usage:
    python build_dx_hierarchy.py \
        --mrrel        /path/to/UMLS/MRREL.RRF \
        --dx_alignment output/dx_alignment.csv \
        --nodes_csv    output/nodes_strict.csv \
        --out_dir      output \
        --neo4j_uri    bolt://localhost:7687 \
        --neo4j_user   neo4j \
        --neo4j_pass   YOUR_PASSWORD \
        [--max_hops 2]         # how many ISA hops above leaf diagnoses to keep
        [--dry_run]            # parse + write CSV but skip Neo4j writes
"""

import os
import csv
import logging
import argparse
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# UMLS relation labels that express IS_A in MRREL
# PAR = "has parent"; CHD = "has child"; RB = "broader"; RN = "narrower"
# We extract PAR direction: child --IS_A--> parent
# ─────────────────────────────────────────────────────────────────────────────
ISA_RELTYPES = {"PAR", "RB"}           # CUI1 PAR CUI2  => CUI1 IS_A CUI2
ISA_RELAS    = {"isa", "inverse_isa"}  # fine-grained RELA filter (optional)


def load_dx_cuis(dx_alignment_csv: str) -> set:
    """Return the set of CUIs for aligned benchmark diagnoses."""
    cuis = set()
    with open(dx_alignment_csv) as f:
        for row in csv.DictReader(f):
            if row["matched"] == "True":
                cuis.add(row["cui"])
    log.info("  Loaded %d aligned diagnosis CUIs", len(cuis))
    return cuis


def load_all_kg_cuis(nodes_csv: str) -> set:
    """Return all CUIs present in the KG node list."""
    cuis = set()
    with open(nodes_csv) as f:
        for row in csv.DictReader(f):
            cuis.add(row["cui"])
    log.info("  Loaded %d KG CUIs from nodes_strict.csv", len(cuis))
    return cuis


def load_cui_names(nodes_csv: str) -> dict:
    """CUI -> preferred name for logging / output."""
    m = {}
    with open(nodes_csv) as f:
        for row in csv.DictReader(f):
            m[row["cui"]] = row["name"]
    return m


def extract_isa_edges_from_mrrel(
    mrrel_path: str,
    seed_cuis: set,
    all_kg_cuis: set,
    max_hops: int,
) -> list[tuple[str, str]]:
    """
    BFS upward from seed_cuis through IS_A edges in MRREL.

    Returns a list of (child_cui, parent_cui) tuples where both CUIs
    are present in all_kg_cuis (so we never introduce orphan nodes).

    max_hops controls how many ancestor levels above the leaf diagnoses
    we traverse.  2 hops is sufficient for SNOMED-style groupings
    (e.g., Strep sore throat → Streptococcal pharyngitis → Pharyngitis).
    """
    log.info("Scanning MRREL for IS_A edges (this may take a minute) ...")

    # Build in-memory child->parents map restricted to KG CUIs only
    # MRREL columns (pipe-delimited, 0-indexed):
    #   0=CUI1, 1=AUI1, 2=STYPE1, 3=REL, 4=CUI2, 5=AUI2,
    #   6=STYPE2, 7=RELA, 8=RUI, 9=SRUI, 10=SAB, 11=SL,
    #   12=RG, 13=DIR, 14=SUPPRESS, 15=CVF
    child_to_parents: dict[str, set] = defaultdict(set)
    total_lines = 0

    with open(mrrel_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            total_lines += 1
            parts = line.rstrip("\n").split("|")
            if len(parts) < 9:
                continue
            cui1 = parts[0]
            rel  = parts[3]
            cui2 = parts[4]

            if rel not in ISA_RELTYPES:
                continue
            # Both ends must be in the KG to avoid dangling references
            if cui1 not in all_kg_cuis or cui2 not in all_kg_cuis:
                continue
            if cui1 == cui2:
                continue
            child_to_parents[cui1].add(cui2)

    log.info("  Scanned %s MRREL lines; found %s child->parent pairs in KG",
             f"{total_lines:,}", f"{sum(len(v) for v in child_to_parents.values()):,}")

    # BFS upward from seed_cuis for max_hops levels
    edges: set[tuple[str, str]] = set()
    frontier = set(seed_cuis)

    for hop in range(max_hops):
        next_frontier = set()
        for child in frontier:
            for parent in child_to_parents.get(child, set()):
                if (child, parent) not in edges:
                    edges.add((child, parent))
                    next_frontier.add(parent)
        log.info("  Hop %d: +%d edges, %d new ancestors",
                 hop + 1, len(edges), len(next_frontier))
        if not next_frontier:
            break
        frontier = next_frontier

    return sorted(edges)


def write_isa_csv(edges: list, out_path: str, cui_names: dict):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["child_cui", "child_name", "parent_cui", "parent_name"])
        for child, parent in edges:
            w.writerow([
                child, cui_names.get(child, ""),
                parent, cui_names.get(parent, ""),
            ])
    log.info("  IS_A edges written -> %s  (%d rows)", out_path, len(edges))


def ingest_to_neo4j(
    edges: list,
    cui_names: dict,
    uri: str,
    user: str,
    password: str,
):
    """
    Writes IS_A edges into Neo4j.
    Assumes DxNode nodes already exist (created by your KG builder).
    Uses MERGE so re-running is safe.
    """
    try:
        from neo4j import GraphDatabase
    except ImportError:
        log.error("neo4j driver not installed. Run: pip install neo4j")
        raise

    driver = GraphDatabase.driver(uri, auth=(user, password))

    # First: ensure parent nodes exist as DxNode (some ancestors may be
    # in nodes_strict.csv but not yet loaded as DxNode — MERGE handles it)
    parent_cuis = {p for _, p in edges}
    with driver.session() as session:
        log.info("  Merging %d ancestor DxNode nodes ...", len(parent_cuis))
        for cui in parent_cuis:
            session.run(
                "MERGE (d:DxNode {cui: $cui}) "
                "ON CREATE SET d.name = $name, d.hierarchy_only = true",
                cui=cui,
                name=cui_names.get(cui, cui),
            )

        log.info("  Merging %d IS_A edges ...", len(edges))
        batch = [{"child": c, "parent": p} for c, p in edges]
        # Process in chunks of 500 to avoid tx size limits
        chunk_size = 500
        for i in range(0, len(batch), chunk_size):
            chunk = batch[i:i + chunk_size]
            session.run(
                """
                UNWIND $rows AS row
                MATCH (child:DxNode  {cui: row.child})
                MATCH (parent:DxNode {cui: row.parent})
                MERGE (child)-[:IS_A]->(parent)
                """,
                rows=chunk,
            )
            log.info("    ... %d / %d edges ingested", min(i + chunk_size, len(batch)), len(edges))

    driver.close()
    log.info("  Neo4j ingest complete.")


def print_coverage_report(
    seed_cuis: set,
    edges: list,
    cui_names: dict,
):
    child_cuis  = {c for c, _ in edges}
    covered     = seed_cuis & child_cuis
    uncovered   = seed_cuis - child_cuis
    parent_cuis = {p for _, p in edges}

    print("\n" + "=" * 65)
    print("IS_A HIERARCHY COVERAGE REPORT")
    print("=" * 65)
    print(f"  Seed diagnosis CUIs          : {len(seed_cuis)}")
    print(f"  Diagnoses WITH IS_A parent   : {len(covered)}  "
          f"({100*len(covered)/len(seed_cuis):.1f}%)")
    print(f"  Diagnoses WITHOUT IS_A parent: {len(uncovered)}")
    print(f"  Ancestor (parent) CUIs added : {len(parent_cuis)}")
    print(f"  Total IS_A edges             : {len(edges)}")

    if uncovered:
        print("\n  Diagnoses with no IS_A edge (will stay as leaf-only nodes):")
        for cui in sorted(uncovered):
            print(f"    {cui}  {cui_names.get(cui, '')}")

    # Show a sample of the hierarchy
    print("\n  Sample hierarchy chains (up to 5):")
    shown = 0
    child_to_parents = defaultdict(list)
    for c, p in edges:
        child_to_parents[c].append(p)

    for cui in sorted(seed_cuis):
        if cui not in child_to_parents:
            continue
        parents = child_to_parents[cui]
        chain   = f"{cui_names.get(cui, cui)} -> " + \
                  " | ".join(cui_names.get(p, p) for p in parents[:3])
        print(f"    {chain}")
        shown += 1
        if shown >= 5:
            break
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Add IS_A hierarchy edges to DxSxKG from UMLS MRREL"
    )
    ap.add_argument("--mrrel",        required=True,
                    help="Path to UMLS MRREL.RRF")
    ap.add_argument("--dx_alignment", required=True,
                    help="output/dx_alignment.csv from align_dataset_to_umls.py")
    ap.add_argument("--nodes_csv",    required=True,
                    help="output/nodes_strict.csv (all KG CUIs)")
    ap.add_argument("--out_dir",      default="output")
    ap.add_argument("--neo4j_uri",    default="bolt://localhost:7687")
    ap.add_argument("--neo4j_user",   default="neo4j")
    ap.add_argument("--neo4j_pass",   default="")
    ap.add_argument("--max_hops",     type=int, default=2,
                    help="Ancestor levels to traverse upward (default: 2). "
                         "2 gives family + superfamily grouping without "
                         "collapsing to trivial supersenses.")
    ap.add_argument("--dry_run",      action="store_true",
                    help="Write CSV only; skip Neo4j ingestion.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    log.info("Loading CUI sets ...")
    seed_cuis   = load_dx_cuis(args.dx_alignment)
    all_kg_cuis = load_all_kg_cuis(args.nodes_csv)
    cui_names   = load_cui_names(args.nodes_csv)

    edges = extract_isa_edges_from_mrrel(
        mrrel_path  = args.mrrel,
        seed_cuis   = seed_cuis,
        all_kg_cuis = all_kg_cuis,
        max_hops    = args.max_hops,
    )

    out_csv = os.path.join(args.out_dir, "isa_edges.csv")
    write_isa_csv(edges, out_csv, cui_names)
    print_coverage_report(seed_cuis, edges, cui_names)

    if not args.dry_run:
        if not args.neo4j_pass:
            log.error("--neo4j_pass is required for Neo4j ingestion. "
                      "Use --dry_run to skip.")
            return
        ingest_to_neo4j(
            edges    = edges,
            cui_names= cui_names,
            uri      = args.neo4j_uri,
            user     = args.neo4j_user,
            password = args.neo4j_pass,
        )
    else:
        log.info("Dry-run mode — Neo4j ingest skipped.")

    log.info("Done. Next step: run orchestrator_hierarchical.py")


if __name__ == "__main__":
    main()
