"""
run_food_ambiguity.py
=====================
Evaluation runner for the food safety ambiguity task.
Mirrors run_check10_hybrid.py from the symptom-diagnosis task.

Usage:
    cd agent/CHA
    export OPENAI_API_KEY=sk-...
    export NEO4J_PASS=.....
    python run_food_ambiguity.py

Output:
    results/food_ambiguity_task/food_results.csv
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI

from openCHA.tasks.ambiguity.food_ambiguity_task import FoodAmbiguityTask


# ── Paths — resolve relative to agent/CHA root ───────────────────
# Script lives at: examples/ambiguity/food_ambiguity_task/run_food_ambiguity.py
# CHA root is 3 levels up
CHA_ROOT   = Path(__file__).resolve().parents[3]
INPUT_CSV  = CHA_ROOT / "data" / "food_ambiguity_dataset.csv"
OUTPUT_DIR = CHA_ROOT / "results" / "food_ambiguity_task"
OUTPUT_CSV = OUTPUT_DIR / "food_results.csv"

MODEL          = "gpt-4o"
DELTA          = 0.05
REPHRASE       = True
NEO4J_PASSWORD = os.getenv("NEO4J_PASS", "12345678")


# ── Helpers ───────────────────────────────────────────────────────
def pretty_print(i: int, row: pd.Series, result: dict) -> None:
    print("\n" + "=" * 80)
    print(f"ROW {i+1}")
    print("=" * 80)
    print(f"QUERY:            {row.get('ambiguous_query','')}")
    print(f"PATIENT_CONTEXT:  {row.get('patient_context','')}")
    print(f"GROUND_TRUTH:     {result['ground_truth']}")
    print(f"LABEL:            {row.get('decision','')}")
    print()
    print(f"FOOD_PHRASE:      {result['food_phrase']}")
    print(f"H_INIT:           {result['n_hypotheses_init']} conditions")
    print(f"ENTROPY_INIT:     {result['entropy_init']:.4f} bits")
    print(f"CONDITIONS_INIT:  {[c['name'] for c in result['candidates_init'][:5]]}")
    print()
    print(f"TURNS:            {result['turns']}")
    print(f"STOP_REASON:      {result['stop_reason']}")
    print(f"AGENT_STATE:      {result['agent_state']}")
    print(f"DECISION:         {result['decision']}")
    if result.get("family_name"):
        print(f"FAMILY:           {result['family_name']}")
    print()
    print("QUESTION LOG:")
    for q in result["question_log"]:
        ans = "Yes" if q["answer"] else "No"
        print(f"  [{ans}] {q['question']}  (IG={q['ig']:.4f})")
    print()
    print(f"H_FINAL:          {result['n_hypotheses_final']} conditions")
    print(f"CONDITIONS_FINAL: {[c['name'] for c in result['candidates_final'][:5]]}")
    print()
    # Evaluation
    pred = result["decision"].strip().upper()
    label = str(row.get("decision","")).strip().lower()
    label_norm = "NOT OKAY" if "not" in label else "OKAY"
    correct = (pred == label_norm) or (
        pred in ("PARTIAL","ABSTAIN") and label_norm == "NOT OKAY"
    )
    print(f"CORRECT:          {'✓' if correct else '✗'}  "
          f"(pred={pred}, label={label_norm})")


# ── Main ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true",
                    help="Run on first 10 rows only (no OpenAI needed)")
    ap.add_argument("--n", type=int, default=None,
                    help="Run on first N rows")
    ap.add_argument("--no_openai", action="store_true",
                    help="Skip OpenAI calls (POMDP logic only)")
    args = ap.parse_args()

    # Test mode: no OpenAI, first 10 rows
    if args.test:
        args.no_openai = True
        args.n = args.n or 10

    client = None
    if not args.no_openai:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. "
                               "Use --no_openai to test without it.")
        client = OpenAI(api_key=api_key)

    df = pd.read_csv(INPUT_CSV)
    if args.n:
        df = df.head(args.n)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")
    if args.test:
        print("TEST MODE — first 10 rows, no OpenAI\n")

    task = FoodAmbiguityTask(password=NEO4J_PASSWORD)

    output_csv = OUTPUT_CSV
    if args.test or (args.n and args.n < 487):
        output_csv = OUTPUT_DIR / f"food_results_test{len(df)}.csv"

    rows_out = []
    for i, row in df.iterrows():
        food_query      = str(row.get("ambiguous_query", ""))
        patient_context = str(row.get("patient_context", ""))
        ground_truth    = patient_context
        label           = str(row.get("decision", "")).strip().lower()

        result = task.run_benchmark_case(
            food_query=food_query,
            patient_context=patient_context,
            ground_truth=ground_truth,
            delta=DELTA,
            openai_client=client,
            model=MODEL,
            rephrase=REPHRASE if client else False,
        )

        pretty_print(i, row, result)

        rows_out.append({
            "row_id":           i,
            "ambiguous_query":  food_query,
            "patient_context":  patient_context,
            "ground_truth":     result["ground_truth"],
            "label":            label,
            "food_phrase":      result["food_phrase"],
            "decision":         result["decision"],
            "agent_state":      result["agent_state"],
            "family_name":      result.get("family_name", ""),
            "max_belief":       result["max_belief"],
            "n_hypotheses_init":  result["n_hypotheses_init"],
            "n_hypotheses_final": result["n_hypotheses_final"],
            "entropy_init":       result["entropy_init"],
            "entropy_final":      result["entropy_final"],
            "candidates_init":  json.dumps(result["candidates_init"]),
            "candidates_final": json.dumps(result["candidates_final"]),
            "turns":            result["turns"],
            "stop_reason":      result["stop_reason"],
            "abstained":        result["abstained"],
            "question_log":     json.dumps(result["question_log"]),
            "clarified_query":  result["clarified_query"],
            "response":         result.get("response", ""),
        })

    out_df = pd.DataFrame(rows_out)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print("\n" + "=" * 80)
    print(f"Saved {len(out_df)} rows → {output_csv}")
    print("=" * 80)
    if args.test:
        print("\nTest passed! Run full evaluation with:")
        print("  export OPENAI_API_KEY=sk-...")
        print("  python run_food_ambiguity.py")
    else:
        print("\nRun evaluate_food_outputs.py to compute metrics.")


if __name__ == "__main__":
    main()
