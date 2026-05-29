#!/usr/bin/env python3
"""
check_edge_coverage.py
----------------------
After running align_dataset_to_umls.py, checks how many of the
benchmark diagnoses have HAS_SYMPTOM edges in the KG.

This tells you whether UMLS alone is sufficient or whether you
need dataset-derived edges too.

Usage:
    python kg/symptom_diagnosis/check_edge_coverage.py \
        --dx_alignment  kg/symptom_diagnosis/output/dx_alignment.csv \
        --sx_alignment  kg/symptom_diagnosis/output/sx_alignment.csv \
        --edges_csv     kg/symptom_diagnosis/output/edges_strict.csv
"""

import csv
import argparse
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx_alignment", required=True)
    ap.add_argument("--sx_alignment", required=True)
    ap.add_argument("--edges_csv",    required=True)
    args = ap.parse_args()

    # Load dx alignment: dataset_term -> cui
    dx_map = {}
    with open(args.dx_alignment) as f:
        for row in csv.DictReader(f):
            if row["matched"] == "True":
                dx_map[row["dataset_term"]] = row["cui"]

    # Load sx alignment: dataset_term -> cui
    sx_map = {}
    with open(args.sx_alignment) as f:
        for row in csv.DictReader(f):
            if row["matched"] == "True":
                sx_map[row["dataset_term"]] = row["cui"]

    # Load edges: build dx_cui -> set of sx_cuis
    dx_to_sx = defaultdict(set)
    with open(args.edges_csv) as f:
        for row in csv.DictReader(f):
            dx_to_sx[row["src_cui"]].add(row["dst_cui"])

    print("=" * 65)
    print("EDGE COVERAGE FOR BENCHMARK DIAGNOSES")
    print("=" * 65)

    dx_cuis_with_edges    = 0
    dx_cuis_without_edges = 0
    sx_cuis_set = set(sx_map.values())

    for dx_term, dx_cui in sorted(dx_map.items()):
        edges = dx_to_sx.get(dx_cui, set())
        # How many connected sx_cuis are actually in the benchmark sx set
        benchmark_sx = edges & sx_cuis_set
        status = "✓" if edges else "✗"
        if edges:
            dx_cuis_with_edges += 1
        else:
            dx_cuis_without_edges += 1
        print(f"  {status} {dx_term:<50} "
              f"total_edges={len(edges):>4}  "
              f"benchmark_sx_overlap={len(benchmark_sx)}")

    print()
    print(f"  Diagnoses WITH edges    : {dx_cuis_with_edges}")
    print(f"  Diagnoses WITHOUT edges : {dx_cuis_without_edges}")
    print()

    if dx_cuis_without_edges > 0:
        print("  Diagnoses with no edges need dataset-derived edges.")
        print("  These will be added by build_dxsxkg.py using FULL_SYMPTOMS column.")
    else:
        print("  All diagnoses have UMLS edges.")
        print("  Dataset edges will further enrich the KG.")

    # Overall sx coverage through edges
    all_sx_via_edges = set()
    for dx_cui in dx_map.values():
        all_sx_via_edges |= dx_to_sx.get(dx_cui, set())
    reachable_benchmark_sx = all_sx_via_edges & sx_cuis_set

    print(f"\n  Benchmark symptoms reachable via edges: "
          f"{len(reachable_benchmark_sx)}/{len(sx_cuis_set)}")


if __name__ == "__main__":
    main()
