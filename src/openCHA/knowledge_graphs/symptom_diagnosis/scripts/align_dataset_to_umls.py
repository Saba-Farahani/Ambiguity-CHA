#!/usr/bin/env python3
"""
align_dataset_to_umls.py
------------------------
Step 3: Map benchmark diagnoses and symptoms to UMLS CUIs.

Matching pipeline (in order):
  1. Exact normalized match
  2. Manual alias lookup  (configs/manual_synonyms.yaml)
  3. Qualifier stripping  (chronic, acute, nos, unspecified, ...)
  4. Partial match (first 3 words) — DISABLED by default.
     Enable only with --allow_partial_match for debugging.
     Do NOT use partial matches in final paper experiments.

For collision resolution when a term maps to multiple CUIs:
  - Prefer CUI whose label matches expected type
  - Prefer CUI from higher-ranked source vocab
  - All collisions saved to collision_review.csv for manual inspection

Output (in --out_dir):
  dx_alignment.csv        PATHOLOGY -> CUI mapping
  sx_alignment.csv        symptom   -> CUI mapping
  collision_review.csv    All multi-CUI terms for manual review
  alignment_report.txt    Human-readable coverage summary
  unmatched_terms.csv     Terms with no CUI found

Usage:
    python align_dataset_to_umls.py \
        --dataset_csv  data/test_patient_dx_symptoms.csv \
        --nodes_csv    output/nodes_strict.csv \
        --norm2cuis    output/norm2cuis_strict.json \
        --synonyms_cfg configs/manual_synonyms.yaml \
        --out_dir      output

    # Only for debugging — not for final experiments:
    python align_dataset_to_umls.py ... --allow_partial_match
"""

import os
import csv
import ast
import json
import yaml
import logging
import argparse
import pandas as pd
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Normalization — must match umls_extractor.py exactly
# ─────────────────────────────────────────────────────────────
def normalize(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    if s.endswith(")") and "(" in s:
        base = s[:s.rfind("(")].strip()
        if base:
            s = base
    return " ".join(s.lower().split())


# ─────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────
def load_dataset_terms(dataset_csv: str):
    df = pd.read_csv(dataset_csv)
    diagnoses = sorted({d.strip() for d in df["PATHOLOGY"].dropna()})
    symptoms  = set()
    for s in df["FULL_SYMPTOMS"].dropna():
        try:
            for item in ast.literal_eval(s):
                symptoms.add(item.strip())
        except Exception:
            continue
    return diagnoses, sorted(symptoms)


def load_norm2cuis(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_node_index(nodes_csv: str):
    cui2label  = {}
    cui2source = {}
    cui2name   = {}
    with open(nodes_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cui2label[row["cui"]]  = row["label"]
            cui2source[row["cui"]] = row.get("pref_source", "")
            cui2name[row["cui"]]   = row["name"]
    return cui2label, cui2source, cui2name


def load_manual_synonyms(yaml_path: str) -> dict:
    if not os.path.exists(yaml_path):
        log.warning("Manual synonyms file not found: %s", yaml_path)
        return {}
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    if not raw:
        return {}
    return {
        k.lower().strip(): [v.lower().strip() for v in vals]
        for k, vals in raw.items()
    }


PREFERRED_SOURCES_RANK = [
    "SNOMEDCT_US", "ICD10CM", "ICD10", "MSH", "NCI",
    "HPO", "OMIM", "MEDCIN", "MTH",
]

def source_rank(src: str) -> int:
    try:
        return PREFERRED_SOURCES_RANK.index(src)
    except ValueError:
        return 99


STRIP_SUFFIXES = [
    " syndrome", " disease", " disorder", " nos",
    " unspecified", " chronic", " acute", " recurrent",
    " idiopathic", " primary", " secondary",
]


# ─────────────────────────────────────────────────────────────
# Collision resolution
# ─────────────────────────────────────────────────────────────
def resolve_collision(cuis: list,
                      expected_label: str,
                      cui2label: dict,
                      cui2source: dict) -> tuple:
    """
    Returns (chosen_cui, resolution_note).

    Priority:
      1. CUIs whose label matches expected_label
      2. Among those, highest-ranked source vocab
      3. First alphabetically as tiebreak
    """
    label_matches = [c for c in cuis if cui2label.get(c) == expected_label]
    candidates    = label_matches if label_matches else cuis

    candidates = sorted(candidates,
                        key=lambda c: (source_rank(cui2source.get(c, "")), c))
    chosen = candidates[0]

    if len(cuis) == 1:
        note = "single_cui"
    elif label_matches:
        note = f"label_match ({len(cuis)} candidates)"
    else:
        note = f"label_mismatch_fallback ({len(cuis)} candidates, none matched '{expected_label}')"

    return chosen, note


# ─────────────────────────────────────────────────────────────
# Single-term matcher
# ─────────────────────────────────────────────────────────────
def match_term(term: str,
               expected_label: str,
               norm2cuis: dict,
               manual_synonyms: dict,
               cui2label: dict,
               cui2source: dict,
               allow_partial: bool = False) -> dict:
    """
    Returns a result dict:
        matched       bool
        cui           str
        method        str
        confidence    "high" | "medium" | "low"
        note          str
        all_cuis      list  (all candidates before collision resolution)
    """
    empty = {
        "matched": False, "cui": "", "method": "",
        "confidence": "", "note": "", "all_cuis": [],
    }

    # ── 1. Exact normalized match ──
    key = normalize(term)
    if key in norm2cuis:
        cuis = norm2cuis[key]
        cui, note = resolve_collision(cuis, expected_label, cui2label, cui2source)
        return {**empty, "matched": True, "cui": cui,
                "method": "exact", "confidence": "high",
                "note": note, "all_cuis": cuis}

    # ── 2. Manual alias lookup (forward) ──
    if key in manual_synonyms:
        for alias in manual_synonyms[key]:
            alias_key = normalize(alias)
            if alias_key in norm2cuis:
                cuis = norm2cuis[alias_key]
                cui, note = resolve_collision(
                    cuis, expected_label, cui2label, cui2source)
                return {**empty, "matched": True, "cui": cui,
                        "method": f"manual_alias:{alias}",
                        "confidence": "high",
                        "note": note, "all_cuis": cuis}

    # Manual alias lookup (reverse — term is an alias of a UMLS concept)
    for umls_key, aliases in manual_synonyms.items():
        if key in [normalize(a) for a in aliases]:
            if umls_key in norm2cuis:
                cuis = norm2cuis[umls_key]
                cui, note = resolve_collision(
                    cuis, expected_label, cui2label, cui2source)
                return {**empty, "matched": True, "cui": cui,
                        "method": f"manual_alias_reverse:{umls_key}",
                        "confidence": "high",
                        "note": note, "all_cuis": cuis}

    # ── 3. Qualifier stripping ──
    for suffix in STRIP_SUFFIXES:
        if key.endswith(suffix):
            stripped = key[:-len(suffix)].strip()
            if stripped and stripped in norm2cuis:
                cuis = norm2cuis[stripped]
                cui, note = resolve_collision(
                    cuis, expected_label, cui2label, cui2source)
                return {**empty, "matched": True, "cui": cui,
                        "method": f"qualifier_strip:{suffix}",
                        "confidence": "medium",
                        "note": note, "all_cuis": cuis}

    # ── 4. Partial match (disabled by default) ──
    if allow_partial:
        words = key.split()
        if len(words) >= 3:
            partial = " ".join(words[:3])
            if partial in norm2cuis:
                cuis = norm2cuis[partial]
                cui, note = resolve_collision(
                    cuis, expected_label, cui2label, cui2source)
                return {**empty, "matched": True, "cui": cui,
                        "method": "partial_3words",
                        "confidence": "low",
                        "note": f"LOW CONFIDENCE — review before using in experiments. {note}",
                        "all_cuis": cuis}

    return empty


# ─────────────────────────────────────────────────────────────
# Align all terms
# ─────────────────────────────────────────────────────────────
def align_terms(terms: list,
                expected_label: str,
                norm2cuis: dict,
                manual_synonyms: dict,
                cui2label: dict,
                cui2source: dict,
                cui2name: dict,
                allow_partial: bool) -> list:
    results = []
    for term in terms:
        r = match_term(term, expected_label, norm2cuis, manual_synonyms,
                       cui2label, cui2source, allow_partial)
        r["dataset_term"]    = term
        r["kg_name"]         = cui2name.get(r["cui"], "") if r["matched"] else ""
        r["expected_label"]  = expected_label
        results.append(r)
    return results


# ─────────────────────────────────────────────────────────────
# Writers
# ─────────────────────────────────────────────────────────────
def write_alignment_csv(path: str, results: list):
    fields = ["dataset_term", "matched", "cui", "kg_name",
              "method", "confidence", "note"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fields})


def write_collision_review(path: str, dx_results: list, sx_results: list):
    """
    Save every term that had multiple candidate CUIs before resolution.
    This file is for manual inspection — one of the easiest places
    for silent mistakes.
    """
    collision_rows = []
    for r in dx_results + sx_results:
        if len(r.get("all_cuis", [])) > 1:
            collision_rows.append(r)

    if not collision_rows:
        log.info("  No collisions found — skipping collision_review.csv")
        return

    fields = ["dataset_term", "expected_label", "all_cuis",
              "chosen_cui", "kg_name", "method", "confidence", "note"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in collision_rows:
            w.writerow({
                "dataset_term":   r["dataset_term"],
                "expected_label": r["expected_label"],
                "all_cuis":       "|".join(r["all_cuis"]),
                "chosen_cui":     r["cui"],
                "kg_name":        r["kg_name"],
                "method":         r["method"],
                "confidence":     r["confidence"],
                "note":           r["note"],
            })
    log.info("  %s collisions saved -> %s", len(collision_rows), path)
    log.info("  Review this file before running experiments.")


def write_unmatched_csv(path: str, dx_results: list, sx_results: list):
    unmatched = (
        [(r["dataset_term"], "Diagnosis") for r in dx_results if not r["matched"]] +
        [(r["dataset_term"], "Symptom")   for r in sx_results if not r["matched"]]
    )
    if not unmatched:
        log.info("  No unmatched terms.")
        return

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "type", "suggested_cui", "notes", "umls_search_url"])
        for term, ttype in unmatched:
            url = ("https://uts.nlm.nih.gov/uts/umls/searchResults?"
                   f"searchString={term.replace(' ', '+')}")
            w.writerow([term, ttype, "", "Fill in manually", url])
    log.info("  %s unmatched terms -> %s", len(unmatched), path)


def print_and_save_report(out_dir, dx_results, sx_results, allow_partial):

    def conf_counts(results):
        return {
            "high":   sum(1 for r in results if r["confidence"] == "high"),
            "medium": sum(1 for r in results if r["confidence"] == "medium"),
            "low":    sum(1 for r in results if r["confidence"] == "low"),
        }

    def method_counts(results):
        counts = defaultdict(int)
        for r in results:
            if r["matched"]:
                base = r["method"].split(":")[0]
                counts[base] += 1
        return dict(counts)

    dx_matched = sum(1 for r in dx_results if r["matched"])
    sx_matched = sum(1 for r in sx_results if r["matched"])
    dx_total   = len(dx_results)
    sx_total   = len(sx_results)
    dx_conf    = conf_counts(dx_results)
    sx_conf    = conf_counts(sx_results)

    lines = [
        "=" * 65,
        "DATASET -> UMLS ALIGNMENT REPORT",
        "=" * 65,
        f"  partial_match mode : {'ENABLED (debug only)' if allow_partial else 'DISABLED'}",
        "",
        f"  Diagnoses : {dx_matched}/{dx_total} ({100*dx_matched/dx_total:.1f}%)",
        f"    high confidence   : {dx_conf['high']}",
        f"    medium confidence : {dx_conf['medium']}",
        f"    low confidence    : {dx_conf['low']}  <- review before using",
        f"    unmatched         : {dx_total - dx_matched}",
        "",
        f"  Symptoms  : {sx_matched}/{sx_total} ({100*sx_matched/sx_total:.1f}%)",
        f"    high confidence   : {sx_conf['high']}",
        f"    medium confidence : {sx_conf['medium']}",
        f"    low confidence    : {sx_conf['low']}  <- review before using",
        f"    unmatched         : {sx_total - sx_matched}",
        "",
        f"  Match methods (diagnoses): {method_counts(dx_results)}",
        f"  Match methods (symptoms) : {method_counts(sx_results)}",
        "",
        "─" * 65,
        "DIAGNOSIS ALIGNMENT DETAILS",
        "─" * 65,
    ]

    for r in dx_results:
        status = "✓" if r["matched"] else "✗"
        lines.append(f"\n  {status} {r['dataset_term']}")
        if r["matched"]:
            conf_tag = f"[{r['confidence']}]"
            lines.append(f"      CUI={r['cui']}  {conf_tag}")
            lines.append(f"      KG name : {r['kg_name']}")
            lines.append(f"      method  : {r['method']}")
            if len(r.get("all_cuis", [])) > 1:
                lines.append(f"      COLLISION: {r['all_cuis']} -> chose {r['cui']}")
        else:
            lines.append("      NOT FOUND — add to manual_synonyms.yaml or unmatched_terms.csv")

    lines += ["", "─" * 65, "SYMPTOM ALIGNMENT DETAILS", "─" * 65]

    for r in sx_results:
        status = "✓" if r["matched"] else "✗"
        lines.append(f"\n  {status} {r['dataset_term']}")
        if r["matched"]:
            conf_tag = f"[{r['confidence']}]"
            lines.append(f"      CUI={r['cui']}  {conf_tag}")
            lines.append(f"      KG name : {r['kg_name']}")
            lines.append(f"      method  : {r['method']}")
            if len(r.get("all_cuis", [])) > 1:
                lines.append(f"      COLLISION: {r['all_cuis']} -> chose {r['cui']}")
        else:
            lines.append("      NOT FOUND — add to manual_synonyms.yaml or unmatched_terms.csv")

    report = "\n".join(lines)
    print(report)

    report_path = os.path.join(out_dir, "alignment_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    log.info("  Report saved -> %s", report_path)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Align benchmark terms to UMLS CUIs (Step 3)")
    ap.add_argument("--dataset_csv",       required=True)
    ap.add_argument("--nodes_csv",         required=True)
    ap.add_argument("--norm2cuis",         required=True)
    ap.add_argument("--synonyms_cfg",
                    default="configs/manual_synonyms.yaml")
    ap.add_argument("--out_dir",           default="./output")
    ap.add_argument("--allow_partial_match", action="store_true",
                    help="Enable low-confidence partial matching. "
                         "For debugging only — do NOT use in final experiments.")
    args = ap.parse_args()

    if args.allow_partial_match:
        log.warning("=" * 55)
        log.warning("Partial matching ENABLED. Debug mode only.")
        log.warning("Do not use these results in final experiments.")
        log.warning("=" * 55)

    os.makedirs(args.out_dir, exist_ok=True)

    log.info("Loading inputs ...")
    diagnoses, symptoms        = load_dataset_terms(args.dataset_csv)
    norm2cuis                  = load_norm2cuis(args.norm2cuis)
    cui2label, cui2source, cui2name = load_node_index(args.nodes_csv)
    manual_synonyms            = load_manual_synonyms(args.synonyms_cfg)

    log.info("  %s diagnoses, %s symptoms", len(diagnoses), len(symptoms))
    log.info("  %s norm2cuis entries", f"{len(norm2cuis):,}")
    log.info("  %s manual synonym groups", len(manual_synonyms))

    log.info("\nAligning diagnoses ...")
    dx_results = align_terms(
        diagnoses, "Diagnosis", norm2cuis, manual_synonyms,
        cui2label, cui2source, cui2name, args.allow_partial_match)

    log.info("Aligning symptoms ...")
    sx_results = align_terms(
        symptoms, "Symptom", norm2cuis, manual_synonyms,
        cui2label, cui2source, cui2name, args.allow_partial_match)

    log.info("\nWriting output ...")
    write_alignment_csv(
        os.path.join(args.out_dir, "dx_alignment.csv"), dx_results)
    write_alignment_csv(
        os.path.join(args.out_dir, "sx_alignment.csv"), sx_results)
    write_collision_review(
        os.path.join(args.out_dir, "collision_review.csv"),
        dx_results, sx_results)
    write_unmatched_csv(
        os.path.join(args.out_dir, "unmatched_terms.csv"),
        dx_results, sx_results)
    print_and_save_report(
        args.out_dir, dx_results, sx_results, args.allow_partial_match)

    # Final summary
    dx_pct = 100 * sum(1 for r in dx_results if r["matched"]) / len(diagnoses)
    sx_pct = 100 * sum(1 for r in sx_results if r["matched"]) / len(symptoms)
    low_conf = sum(1 for r in dx_results + sx_results
                   if r["confidence"] == "low")

    log.info("=" * 50)
    log.info("Diagnoses aligned : %.1f%%", dx_pct)
    log.info("Symptoms aligned  : %.1f%%", sx_pct)
    if low_conf:
        log.warning("%s low-confidence matches — review collision_review.csv",
                    low_conf)
    log.info("=" * 50)

    if dx_pct < 100 or sx_pct < 100:
        log.info("Action: fill unmatched_terms.csv or add to manual_synonyms.yaml")
    else:
        log.info("100%% coverage. Ready for build_dxsxkg.py")


if __name__ == "__main__":
    main()
