"""
evaluate_food_outputs.py
========================
Computes all metrics for the food safety task.
Mirrors evaluate_check10_outputs.py and compute_all_metrics.py.

Produces numbers for:
  - Table 12 / Table 16 in the paper
  - Section 4.2 text
  - Abstention analysis
  - Three-state breakdown (ANSWER / PARTIAL / ABSTAIN)

Run after run_food_ambiguity.py:
    python evaluate_food_outputs.py
"""

import json
import math
from pathlib import Path

import pandas as pd


# ── Paths — resolve relative to agent/CHA root ───────────────────
CHA_ROOT    = Path(__file__).resolve().parents[3]
INPUT_CSV   = CHA_ROOT / "results" / "food_ambiguity_task" / "food_results.csv"
OUTPUT_CSV  = CHA_ROOT / "results" / "food_ambiguity_task" / "food_results_evaluated.csv"
OUTPUT_JSON = CHA_ROOT / "results" / "food_ambiguity_task" / "food_summary.json"


# ── Helpers ───────────────────────────────────────────────────────
def norm(x) -> str:
    return " ".join(str(x).strip().lower().split()) if x else ""


def label_to_binary(label: str) -> str:
    """Normalise dataset label to NOT OKAY / OKAY."""
    l = norm(label)
    if "not" in l or "unsafe" in l:
        return "NOT OKAY"
    return "OKAY"


def decision_correct(pred: str, label_binary: str) -> bool:
    """
    Correctness rules:
    - ANSWER NOT OKAY == label NOT OKAY  → correct
    - ANSWER OKAY     == label OKAY      → correct
    - PARTIAL / ABSTAIN when label NOT OKAY → counted as incorrect
      (conservative — mirrors symptom task where ABSTAIN = wrong)
    """
    p = norm(pred).upper()
    if p == "NOT OKAY" and label_binary == "NOT OKAY":
        return True
    if p == "OKAY" and label_binary == "OKAY":
        return True
    return False


def mcc(tp, tn, fp, fn) -> float:
    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return (tp*tn - fp*fn) / denom if denom else 0.0


# ── Load ──────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
n  = len(df)
print(f"Loaded {n} rows from {INPUT_CSV}")

# Normalise labels and predictions
df["label_binary"] = df["label"].apply(label_to_binary)
df["pred_binary"]  = df["decision"].apply(
    lambda x: norm(str(x)).upper() if norm(str(x)).upper() in ("NOT OKAY","OKAY")
              else norm(str(x)).upper()
)
df["correct"] = df.apply(
    lambda r: decision_correct(r["decision"], r["label_binary"]), axis=1
)


# ── Overall metrics ───────────────────────────────────────────────
accuracy     = df["correct"].mean()
abstain_mask = df["agent_state"] == "ABSTAIN"

# Confusion matrix (NOT OKAY = positive)
not_ok_label = df["label_binary"] == "NOT OKAY"
not_ok_pred  = df["pred_binary"]  == "NOT OKAY"

tp = int((not_ok_label &  not_ok_pred).sum())
tn = int((~not_ok_label & ~not_ok_pred).sum())
fp = int((~not_ok_label &  not_ok_pred).sum())
fn = int((not_ok_label  & ~not_ok_pred).sum())

precision_notok = tp / (tp + fp) if (tp + fp) else 0.0
recall_notok    = tp / (tp + fn) if (tp + fn) else 0.0
f1_notok        = (2 * precision_notok * recall_notok /
                   (precision_notok + recall_notok)
                   if (precision_notok + recall_notok) else 0.0)
mcc_val         = mcc(tp, tn, fp, fn)
bal_acc         = 0.5 * (
    recall_notok +
    (tn / (tn + fp) if (tn + fp) else 0.0)
)


# ── Three-state breakdown ─────────────────────────────────────────
answer_df  = df[df["agent_state"] == "ANSWER"]
partial_df = df[df["agent_state"] == "PARTIAL"]
abstain_df = df[df["agent_state"] == "ABSTAIN"]


def state_metrics(sub: pd.DataFrame, label: str) -> dict:
    if len(sub) == 0:
        return {"n": 0, "pct": 0.0, "accuracy": 0.0, "mean_max_belief": 0.0}
    acc = sub["correct"].mean()
    return {
        "n":              len(sub),
        "pct":            round(100 * len(sub) / n, 1),
        "accuracy":       round(100 * acc, 2),
        "mean_max_belief": round(sub["max_belief"].mean(), 3),
    }


# ── Ambiguity reduction ───────────────────────────────────────────
df["entropy_reduction"] = df["entropy_init"] - df["entropy_final"]


# ── Stopping condition distribution ──────────────────────────────
stop_counts = df["stop_reason"].value_counts(dropna=False).to_dict()


# ── Abstain analysis ──────────────────────────────────────────────
abstain_by_gt = abstain_df.groupby("ground_truth").size().to_dict() if len(abstain_df) else {}


# ── Per-condition performance ─────────────────────────────────────
cond_perf = []
for cond, sub in df.groupby("ground_truth"):
    c_acc = sub["correct"].mean()
    n_abs = (sub["agent_state"] == "ABSTAIN").sum()
    cond_perf.append({
        "condition": cond,
        "n": len(sub),
        "accuracy_pct": round(100 * c_acc, 1),
        "abstain_n": int(n_abs),
    })
cond_df = pd.DataFrame(cond_perf).sort_values("accuracy_pct", ascending=False)


# ── Summary ───────────────────────────────────────────────────────
summary = {
    "n_total":        n,
    "accuracy":       round(100 * accuracy, 2),
    "balanced_accuracy": round(100 * bal_acc, 2),
    "precision_not_ok": round(precision_notok, 4),
    "recall_not_ok":    round(recall_notok, 4),
    "f1_not_ok":        round(f1_notok, 4),
    "mcc":              round(mcc_val, 4),
    "tp": tp, "tn": tn, "fp": fp, "fn": fn,

    # Three-state
    "answer_state":  state_metrics(answer_df,  "ANSWER"),
    "partial_state": state_metrics(partial_df, "PARTIAL"),
    "abstain_state": state_metrics(abstain_df, "ABSTAIN"),

    # Ambiguity reduction
    "mean_h_init":          round(df["n_hypotheses_init"].mean(), 3),
    "mean_h_final":         round(df["n_hypotheses_final"].mean(), 3),
    "mean_entropy_init":    round(df["entropy_init"].mean(), 4),
    "mean_entropy_final":   round(df["entropy_final"].mean(), 4),
    "mean_entropy_reduction": round(df["entropy_reduction"].mean(), 4),
    "mean_turns":           round(df["turns"].mean(), 3),
    "median_turns":         float(df["turns"].median()),
    "abstention_pct":       round(100 * abstain_mask.mean(), 2),

    # Stopping
    "stop_reason_counts": stop_counts,
    "abstain_by_condition": abstain_by_gt,
}


# ── Print results (for paper) ──────────────────────────────────────
print("\n" + "="*60)
print("TABLE 12 / TABLE 16 — Food Safety Results")
print("="*60)
print(f"  n total                : {n}")
print(f"  Accuracy               : {summary['accuracy']:.2f}%")
print(f"  Balanced Accuracy      : {summary['balanced_accuracy']:.2f}%")
print(f"  Precision (NOT OK)     : {summary['precision_not_ok']:.4f}")
print(f"  Recall    (NOT OK)     : {summary['recall_not_ok']:.4f}")
print(f"  F1        (NOT OK)     : {summary['f1_not_ok']:.4f}")
print(f"  MCC                    : {summary['mcc']:.4f}")

print(f"\n  Confusion matrix:")
print(f"    TP={tp}  FP={fp}")
print(f"    FN={fn}  TN={tn}")

print(f"\n  Three-state breakdown:")
for state, key in [("ANSWER","answer_state"),("PARTIAL","partial_state"),("ABSTAIN","abstain_state")]:
    s = summary[key]
    print(f"    {state:8s}: n={s['n']} ({s['pct']}%)  "
          f"accuracy={s['accuracy']}%  "
          f"mean_max_b={s['mean_max_belief']}")

print(f"\n  Ambiguity reduction:")
print(f"    Mean |H_init|     : {summary['mean_h_init']:.3f}")
print(f"    Mean |H_final|    : {summary['mean_h_final']:.3f}")
print(f"    Mean H(H_init)    : {summary['mean_entropy_init']:.4f} bits")
print(f"    Mean H(H_final)   : {summary['mean_entropy_final']:.4f} bits")
print(f"    Mean H reduction  : {summary['mean_entropy_reduction']:.4f} bits")
print(f"    Mean turns        : {summary['mean_turns']:.3f}")
print(f"    Abstention rate   : {summary['abstention_pct']:.2f}%")

print(f"\n  Stopping conditions:")
for reason, count in stop_counts.items():
    print(f"    {reason:30s}: {count} ({100*count/n:.1f}%)")

print(f"\n  Per-condition performance (top 10):")
for _, r in cond_df.head(10).iterrows():
    print(f"    {r['condition'][:35]:<35} n={r['n']:4d}  "
          f"acc={r['accuracy_pct']:5.1f}%  abs={r['abstain_n']}")

print("="*60)

# ── Save ──────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved evaluated CSV  → {OUTPUT_CSV}")
print(f"Saved summary JSON   → {OUTPUT_JSON}")