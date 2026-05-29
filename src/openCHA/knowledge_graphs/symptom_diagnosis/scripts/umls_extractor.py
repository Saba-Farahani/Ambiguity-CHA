#!/usr/bin/env python3
"""
umls_extractor.py
-----------------
Step 1: Build the base DxSxKG from UMLS only.

Implements all recommendations:
  - Strict vs expanded relation modes (--mode strict|expanded)
  - Multiple CUIs per normalized term (no silent collapsing)
  - Full provenance on every node and edge
  - Narrower semantic types by default
  - Comprehensive statistics logged to stats.json
  - Separation from dataset alignment (see align_dataset_to_umls.py)

Output files (in --out_dir):
  nodes.csv         All Diagnosis and Symptom nodes
  edges.csv         All HAS_SYMPTOM edges
  norm2cuis.json    normalized_string -> [list of CUIs]  (multi-valued)
  stats.json        Extraction statistics for paper / debugging

Usage:
    python umls_extractor.py \
        --umls_dir /path/to/umls/META \
        --out_dir  ./output \
        --mode     strict
"""

import os
import csv
import json
import argparse
import logging
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Semantic types  (point 4: start narrow)
# ─────────────────────────────────────────────────────────────
DISEASE_STYS_STRICT = {
    "T047",   # Disease or Syndrome
    "T048",   # Mental or Behavioral Dysfunction
}
DISEASE_STYS_EXPANDED = DISEASE_STYS_STRICT | {
    "T191",   # Neoplastic Process
}

SYMPTOM_STYS_STRICT = {
    "T184",   # Sign or Symptom
    "T033",   # Finding — needed for Edema, Rales, Weight Loss, Sweating etc.
}
SYMPTOM_STYS_EXPANDED = SYMPTOM_STYS_STRICT | {
    "T034",   # Laboratory or Test Result
}

# ─────────────────────────────────────────────────────────────
# Relation types  (point 1: strict vs expanded)
# ─────────────────────────────────────────────────────────────
STRICT_RELS = {
    # Direct symptom-manifestation relations confirmed in your MRREL.RRF
    "manifestation_of",           # disease->symptom: symptom manifests from disease
    "has_associated_finding",     # disease has this finding/symptom
    "may_be_finding_of_disease",  # symptom may be finding of disease (reversed)
    "has_defining_characteristic",# disease defined by this symptom
    # Original targets kept for other UMLS versions
    "has_sign_or_symptom",
    "has_finding",
    "has_manifestation",
}

EXPANDED_RELS = STRICT_RELS | {
    # Weaker but clinically relevant — confirmed present in your file
    "clinically_associated_with", # 445 occurrences in your MRREL
    "associated_with",            # 101 occurrences
    "has_associated_condition",   # 9 occurrences
    "co-occurs_with",             # 58 occurrences
    "due_to",                     # 22 occurrences
    "has_related_factor",         # 6 occurrences
}

# ─────────────────────────────────────────────────────────────
# Source vocabulary priority for preferred name selection
# ─────────────────────────────────────────────────────────────
PREFERRED_SOURCES = [
    "SNOMEDCT_US", "ICD10CM", "ICD10", "MSH", "NCI",
    "HPO", "OMIM", "MEDCIN", "MTH",
]
PREFERRED_TTYS = {"PT", "PN", "PF", "MH", "HT", "FN"}


# ─────────────────────────────────────────────────────────────
# Normalization  (shared with all other scripts)
# ─────────────────────────────────────────────────────────────
def normalize(s: str) -> str:
    """
    Lowercase + collapse whitespace + strip parenthetical qualifiers.
    e.g. "Dyspnea (finding)" -> "dyspnea"
    Keep in sync with align_dataset_to_umls.py and check_kg_coverage.py.
    """
    if not s:
        return ""
    s = s.strip()
    if s.endswith(")") and "(" in s:
        base = s[:s.rfind("(")].strip()
        if base:
            s = base
    return " ".join(s.lower().split())


def expand_slash_compound(term: str) -> list:
    """
    Split slash-joined compound symptom names.
    "Runny/Stuffy Nose" -> ["Runny Nose", "Stuffy Nose"]
    Returns empty list if pattern not recognized.
    """
    if "/" not in term:
        return []
    parts = term.split("/")
    if len(parts) != 2:
        return []
    left, right = parts[0].strip(), parts[1].strip()
    right_words = right.split()
    if len(right_words) >= 2:
        shared = " ".join(right_words[1:])
        return [f"{left} {shared}", right]
    return [left, right]


def get_source_rank(sab: str) -> int:
    try:
        return PREFERRED_SOURCES.index(sab)
    except ValueError:
        return 99


# ─────────────────────────────────────────────────────────────
# UMLS file readers
# ─────────────────────────────────────────────────────────────
def load_semantic_types(mrsty_path: str) -> dict:
    log.info("Loading MRSTY.RRF ...")
    cui2stys = defaultdict(set)
    with open(mrsty_path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("|")
            cui2stys[p[0]].add(p[1])
    log.info("  %s concepts with semantic types", f"{len(cui2stys):,}")
    return dict(cui2stys)


def load_concept_names(mrconso_path: str,
                       cui2stys: dict,
                       disease_stys: set,
                       symptom_stys: set):
    """
    Returns:
        disease_nodes  dict  cui -> {name, synonyms, icd10, sources}
        symptom_nodes  dict  cui -> {name, synonyms, sources}
        norm2cuis      dict  normalized_str -> [list of CUIs]  (multi-valued)
        overlap_cuis   set   CUIs that belong to BOTH disease and symptom groups
    """
    log.info("Loading MRCONSO.RRF (may take 1-2 min) ...")

    cui_entries  = defaultdict(list)   # cui -> [(src_rank, tty_pref, sab, tty, string)]
    cui_icd10    = {}
    cui_sources  = defaultdict(set)    # cui -> set of source vocabularies seen

    with open(mrconso_path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("|")
            if len(p) < 17:
                continue
            cui = p[0]; lat = p[1]; sab = p[11]
            tty = p[12]; code = p[13]; name = p[14]; sup = p[16]

            if lat != "ENG" or sup in {"Y", "O", "E"}:
                continue

            stys = cui2stys.get(cui, set())
            if not (stys & disease_stys or stys & symptom_stys):
                continue

            src_rank = get_source_rank(sab)
            tty_pref = 0 if tty in PREFERRED_TTYS else 1
            cui_entries[cui].append((src_rank, tty_pref, sab, tty, name))
            cui_sources[cui].add(sab)

            if sab in {"ICD10CM", "ICD10"} and cui not in cui_icd10:
                cui_icd10[cui] = code

    log.info("  %s relevant CUIs found", f"{len(cui_entries):,}")

    disease_nodes = {}
    symptom_nodes = {}
    # point 2: multi-valued lookup
    norm2cuis     = defaultdict(list)

    for cui, entries in cui_entries.items():
        stys       = cui2stys.get(cui, set())
        is_disease = bool(stys & disease_stys)
        is_symptom = bool(stys & symptom_stys)

        # Sort: best source first, then preferred TTY, then alphabetical
        entries.sort(key=lambda x: (x[0], x[1], x[4]))
        best_name = entries[0][4]
        best_sab  = entries[0][2]

        all_names = list({e[4] for e in entries})
        synonyms  = [n for n in all_names if n != best_name]

        # Compound expansion layer
        for n in list(all_names):
            synonyms.extend(expand_slash_compound(n))
        synonyms = list(set(synonyms))

        # point 5: provenance
        sources_str = "|".join(sorted(cui_sources[cui]))

        entry = {
            "name":         best_name,
            "synonyms":     synonyms,
            "icd10":        cui_icd10.get(cui, ""),
            "pref_source":  best_sab,
            "all_sources":  sources_str,
        }

        # point 2: populate multi-valued norm2cuis
        for text in [best_name] + synonyms:
            key = normalize(text)
            if key and cui not in norm2cuis[key]:
                norm2cuis[key].append(cui)

        if is_disease:
            disease_nodes[cui] = entry
        elif is_symptom:
            # only add as symptom if NOT already a disease
            symptom_nodes[cui] = entry

    # point 3: overlap detection
    overlap_cuis = set(disease_nodes) & set(symptom_nodes)

    log.info("  %s disease nodes", f"{len(disease_nodes):,}")
    log.info("  %s symptom nodes", f"{len(symptom_nodes):,}")
    log.info("  %s CUIs in both groups (overlap)", f"{len(overlap_cuis):,}")
    log.info("  %s normalized strings indexed", f"{len(norm2cuis):,}")

    return disease_nodes, symptom_nodes, dict(norm2cuis), overlap_cuis


def load_relations(mrrel_path: str,
                   disease_cuis: set,
                   symptom_cuis: set,
                   allowed_rels: set) -> list:
    """
    Extracts HAS_SYMPTOM edges from MRREL.RRF.

    Direction logic (confirmed from your MRREL diagnostic):
      - CUI1=disease, CUI2=symptom, RELA=manifestation_of
        -> disease HAS_SYMPTOM symptom  (forward, symptom manifests from disease)
      - CUI1=symptom, CUI2=disease, RELA=manifestation_of
        -> disease HAS_SYMPTOM symptom  (reversed)

    Empty RELA handling:
      - When RELA is empty, fall back to REL field
      - REL='RO' (related other) for disease->symptom = include in expanded mode
    """
    mode_label = "strict" if allowed_rels <= STRICT_RELS else "expanded"
    log.info("Loading MRREL.RRF mode=%s (scanning full file) ...", mode_label)

    edges = []
    seen  = set()
    lines_read   = 0
    lines_skipped_sup = 0

    with open(mrrel_path, encoding="utf-8") as f:
        for line in f:
            lines_read += 1
            if lines_read % 5_000_000 == 0:
                log.info("  ... %s lines read, %s edges so far",
                         f"{lines_read:,}", f"{len(edges):,}")

            p = line.rstrip("\n").split("|")
            if len(p) < 16:
                continue

            cui1 = p[0]
            cui2 = p[4]
            rel  = p[3].lower()   # generic relation e.g. ro, rn, par
            rela = p[7].lower()   # specific relation e.g. manifestation_of
            sab  = p[10]
            sup  = p[14]

            if sup in {"Y", "O", "E"}:
                lines_skipped_sup += 1
                continue

            # Use RELA if available, fall back to REL
            effective_rel = rela if rela else rel

            # ── Forward: disease -> symptom ──
            if cui1 in disease_cuis and cui2 in symptom_cuis:
                if effective_rel in allowed_rels:
                    key = (cui1, cui2)
                    if key not in seen:
                        seen.add(key)
                        edges.append({
                            "src_cui":       cui1,
                            "dst_cui":       cui2,
                            "type":          "HAS_SYMPTOM",
                            "source_rel":    rela or "",
                            "source_rel_attr": rel,
                            "source_vocab":  sab,
                            "direction":     "forward",
                        })

            # ── Reverse: symptom -> disease ──
            # e.g. "symptom is manifestation_of disease"
            # interpreted as disease HAS_SYMPTOM symptom
            # Uses the same allowed_rels so strict/expanded is respected
            elif cui1 in symptom_cuis and cui2 in disease_cuis:
                if effective_rel in allowed_rels:
                    key = (cui2, cui1)  # store as disease->symptom
                    if key not in seen:
                        seen.add(key)
                        edges.append({
                            "src_cui":       cui2,
                            "dst_cui":       cui1,
                            "type":          "HAS_SYMPTOM",
                            "source_rel":    rela or "",
                            "source_rel_attr": rel,
                            "source_vocab":  sab,
                            "direction":     "reversed",
                        })

    log.info("  %s total lines read", f"{lines_read:,}")
    log.info("  %s suppressed lines skipped", f"{lines_skipped_sup:,}")
    log.info("  %s HAS_SYMPTOM edges extracted", f"{len(edges):,}")
    return edges


# ─────────────────────────────────────────────────────────────
# Statistics  (point 3)
# ─────────────────────────────────────────────────────────────
def compute_stats(disease_nodes, symptom_nodes, edges,
                  norm2cuis, overlap_cuis, mode):
    """Build and return a statistics dict for logging and paper."""

    # Collision analysis: how many normalized strings map to >1 CUI
    collision_keys = {k: v for k, v in norm2cuis.items() if len(v) > 1}
    top_collisions = sorted(
        [(k, v) for k, v in collision_keys.items()],
        key=lambda x: -len(x[1])
    )[:10]

    # Edge source vocabulary breakdown
    edge_by_vocab = defaultdict(int)
    edge_by_rel   = defaultdict(int)
    for e in edges:
        edge_by_vocab[e["source_vocab"]] += 1
        edge_by_rel[e["source_rel"]] += 1

    stats = {
        "mode":                     mode,
        "disease_nodes":            len(disease_nodes),
        "symptom_nodes":            len(symptom_nodes),
        "total_nodes":              len(disease_nodes) + len(symptom_nodes),
        "has_symptom_edges":        len(edges),
        "overlap_cuis":             len(overlap_cuis),
        "overlap_cui_list":         sorted(overlap_cuis)[:20],
        "norm_strings_total":       len(norm2cuis),
        "norm_strings_multi_cui":   len(collision_keys),
        "collision_rate_pct":       round(
            100 * len(collision_keys) / len(norm2cuis), 2) if norm2cuis else 0,
        "top_collision_examples":   [
            {"key": k, "cuis": v} for k, v in top_collisions
        ],
        "edges_by_vocab":           dict(edge_by_vocab),
        "edges_by_rel":             dict(edge_by_rel),
    }
    return stats


def log_stats(stats):
    log.info("─" * 50)
    log.info("EXTRACTION STATISTICS")
    log.info("─" * 50)
    log.info("  Mode              : %s", stats["mode"])
    log.info("  Disease nodes     : %s", f"{stats['disease_nodes']:,}")
    log.info("  Symptom nodes     : %s", f"{stats['symptom_nodes']:,}")
    log.info("  Overlap CUIs      : %s", f"{stats['overlap_cuis']:,}")
    log.info("  HAS_SYMPTOM edges : %s", f"{stats['has_symptom_edges']:,}")
    log.info("  Norm strings      : %s", f"{stats['norm_strings_total']:,}")
    log.info("  Multi-CUI strings : %s (%.1f%%)",
             stats["norm_strings_multi_cui"], stats["collision_rate_pct"])
    log.info("  Top collisions    :")
    for ex in stats["top_collision_examples"][:5]:
        log.info("    '%s' -> %s", ex["key"], ex["cuis"])
    log.info("  Edges by relation :")
    for rel, cnt in sorted(stats["edges_by_rel"].items(), key=lambda x: -x[1]):
        log.info("    %-40s %s", rel, f"{cnt:,}")


# ─────────────────────────────────────────────────────────────
# Writers
# ─────────────────────────────────────────────────────────────
def write_nodes_csv(path: str, disease_nodes: dict, symptom_nodes: dict):
    log.info("Writing %s ...", path)
    fieldnames = ["cui", "label", "name", "synonyms",
                  "icd10", "pref_source", "all_sources"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for cui, d in disease_nodes.items():
            w.writerow({
                "cui":        cui,
                "label":      "Diagnosis",
                "name":       d["name"],
                "synonyms":   "|".join(d["synonyms"]),
                "icd10":      d["icd10"],
                "pref_source": d["pref_source"],
                "all_sources": d["all_sources"],
            })
        for cui, d in symptom_nodes.items():
            w.writerow({
                "cui":        cui,
                "label":      "Symptom",
                "name":       d["name"],
                "synonyms":   "|".join(d["synonyms"]),
                "icd10":      "",
                "pref_source": d["pref_source"],
                "all_sources": d["all_sources"],
            })
    log.info("  %s nodes written", f"{len(disease_nodes)+len(symptom_nodes):,}")


def write_edges_csv(path: str, edges: list):
    log.info("Writing %s ...", path)
    fieldnames = ["src_cui", "dst_cui", "type",
                  "source_rel", "source_rel_attr",
                  "source_vocab", "direction"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(edges)
    log.info("  %s edges written", f"{len(edges):,}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Build base DxSxKG from UMLS (Step 1 of pipeline)")
    ap.add_argument("--umls_dir", required=True,
                    help="Path to UMLS META folder (contains RRF files)")
    ap.add_argument("--out_dir",  default="./output",
                    help="Output directory")
    ap.add_argument("--mode",     choices=["strict", "expanded"],
                    default="strict",
                    help="strict: use only direct symptom relations; "
                         "expanded: also include weaker relations like associated_with")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Verify files exist
    for fname in ["MRCONSO.RRF", "MRREL.RRF", "MRSTY.RRF"]:
        p = os.path.join(args.umls_dir, fname)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing UMLS file: {p}")

    # Select semantic types and relations by mode
    if args.mode == "strict":
        disease_stys = DISEASE_STYS_STRICT
        symptom_stys = SYMPTOM_STYS_STRICT
        allowed_rels = STRICT_RELS
    else:
        disease_stys = DISEASE_STYS_EXPANDED
        symptom_stys = SYMPTOM_STYS_EXPANDED
        allowed_rels = EXPANDED_RELS

    log.info("=" * 50)
    log.info("DxSxKG Extractor  mode=%s", args.mode)
    log.info("Disease STYs : %s", sorted(disease_stys))
    log.info("Symptom STYs : %s", sorted(symptom_stys))
    log.info("Allowed rels : %s", sorted(allowed_rels))
    log.info("=" * 50)

    # Step 1: semantic types
    log.info("\n[1/4] Semantic types ...")
    cui2stys = load_semantic_types(os.path.join(args.umls_dir, "MRSTY.RRF"))

    # Step 2: concept names
    log.info("\n[2/4] Concept names ...")
    disease_nodes, symptom_nodes, norm2cuis, overlap_cuis = load_concept_names(
        os.path.join(args.umls_dir, "MRCONSO.RRF"),
        cui2stys, disease_stys, symptom_stys)

    # Step 3: relations
    log.info("\n[3/4] Relations ...")
    edges = load_relations(
        os.path.join(args.umls_dir, "MRREL.RRF"),
        set(disease_nodes.keys()),
        set(symptom_nodes.keys()),
        allowed_rels)

    # Step 4: write output
    log.info("\n[4/4] Writing output ...")
    write_nodes_csv(
        os.path.join(args.out_dir, f"nodes_{args.mode}.csv"),
        disease_nodes, symptom_nodes)
    write_edges_csv(
        os.path.join(args.out_dir, f"edges_{args.mode}.csv"),
        edges)

    # Save multi-valued norm2cuis  (point 2)
    norm2cuis_path = os.path.join(args.out_dir, f"norm2cuis_{args.mode}.json")
    with open(norm2cuis_path, "w") as f:
        json.dump(norm2cuis, f)
    log.info("norm2cuis saved -> %s", norm2cuis_path)

    # Compute and save stats  (point 3)
    stats = compute_stats(disease_nodes, symptom_nodes,
                          edges, norm2cuis, overlap_cuis, args.mode)
    log_stats(stats)
    stats_path = os.path.join(args.out_dir, f"stats_{args.mode}.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info("Stats saved -> %s", stats_path)

    log.info("\nDone. Next step:")
    log.info("  python check_kg_coverage.py --nodes_csv output/nodes_%s.csv ...", args.mode)


if __name__ == "__main__":
    main()
