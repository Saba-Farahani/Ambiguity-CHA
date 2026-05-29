import ast
import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_CSV = BASE_DIR / "results" / "diagnostic_ambiguity_task" / "test_locked_gpt4_outputs.csv"
CATEGORY_MAP_CSV = BASE_DIR / "data" / "diagnosis_category_map.csv"

OUTPUT_ROW_CSV = BASE_DIR / "results" / "diagnostic_ambiguity_task" / "test_locked_gpt4_outputs_evaluated.csv"
OUTPUT_SUMMARY_JSON = BASE_DIR / "results" / "diagnostic_ambiguity_task" / "test_locked_gpt4_summary.json"


def normalize_text(x):
    if x is None:
        return ""
    return " ".join(str(x).strip().lower().split())


def parse_maybe_json_list(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return x
    s = str(x).strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    return []


def parse_candidate_names(x):
    items = parse_maybe_json_list(x)
    names = []
    for item in items:
        if isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]))
        elif isinstance(item, str):
            names.append(item)
    return names


def get_category_map(path):
    if not path.exists():
        return {}
    cat_df = pd.read_csv(path)
    return {
        normalize_text(r["diagnosis"]): str(r["category"]).strip().lower()
        for _, r in cat_df.iterrows()
    }


def get_category(name, cmap):
    return cmap.get(normalize_text(name), "")


def pct(series):
    return round(100.0 * series.mean(), 2) if len(series) else 0.0


df = pd.read_csv(INPUT_CSV)
category_map = get_category_map(CATEGORY_MAP_CSV)

rows = []

for _, row in df.iterrows():
    gt = str(row.get("ground_truth", ""))
    gt_norm = normalize_text(gt)
    gt_cat = get_category(gt, category_map)

    candidates_init = parse_candidate_names(row.get("candidates_init", ""))
    candidates_final = parse_candidate_names(row.get("candidates_final", ""))
    pred_top1 = str(row.get("prediction_top1", "")).strip()
    pred_top5 = parse_maybe_json_list(row.get("prediction_top5", ""))

    cand_init_norm = [normalize_text(x) for x in candidates_init]
    cand_final_norm = [normalize_text(x) for x in candidates_final]
    pred_top1_norm = normalize_text(pred_top1)
    pred_top5_norm = [normalize_text(x) for x in pred_top5]

    gt_in_hq_init_exact = gt_norm in cand_init_norm
    gt_in_hq_final_exact = gt_norm in cand_final_norm
    top1_exact_correct = pred_top1_norm == gt_norm
    top5_exact_correct = gt_norm in pred_top5_norm

    cand_init_cats = [get_category(x, category_map) for x in candidates_init if get_category(x, category_map)]
    cand_final_cats = [get_category(x, category_map) for x in candidates_final if get_category(x, category_map)]
    pred_top1_cat = get_category(pred_top1, category_map)
    pred_top5_cats = [get_category(x, category_map) for x in pred_top5 if get_category(x, category_map)]

    gt_in_hq_init_category = gt_cat in cand_init_cats if gt_cat else False
    gt_in_hq_final_category = gt_cat in cand_final_cats if gt_cat else False
    top1_category_correct = (pred_top1_cat == gt_cat) if gt_cat and pred_top1_cat else False
    top5_category_correct = gt_cat in pred_top5_cats if gt_cat else False

    n_h_init = int(row.get("n_hypotheses_init", 0))
    n_h_final = int(row.get("n_hypotheses_final", 0))
    ent_init = float(row.get("entropy_init", 0.0))
    ent_final = float(row.get("entropy_final", 0.0))
    turns = int(row.get("turns", 0))
    abstained = str(row.get("abstained", "")).strip().lower() in {"true", "1", "yes"}
    stop_reason = str(row.get("stop_reason", "")).strip()

    rows.append({
        **row.to_dict(),
        "gt_category": gt_cat,
        "pred_top1_category": pred_top1_cat,
        "gt_in_hq_init_exact": gt_in_hq_init_exact,
        "gt_in_hq_final_exact": gt_in_hq_final_exact,
        "gt_in_hq_init_category": gt_in_hq_init_category,
        "gt_in_hq_final_category": gt_in_hq_final_category,
        "top1_exact_correct": top1_exact_correct,
        "top5_exact_correct": top5_exact_correct,
        "top1_category_correct": top1_category_correct,
        "top5_category_correct": top5_category_correct,
        "candidate_reduction": n_h_init - n_h_final,
        "entropy_reduction": ent_init - ent_final,
        "turns_eval": turns,
        "abstained_eval": abstained,
        "stop_reason_eval": stop_reason,
    })

eval_df = pd.DataFrame(rows)

summary = {
    "n_rows": len(eval_df),
    "nonempty_hq_init_pct": pct(eval_df["n_hypotheses_init"] > 0),
    "gt_in_hq_init_exact_pct": pct(eval_df["gt_in_hq_init_exact"]),
    "gt_in_hq_final_exact_pct": pct(eval_df["gt_in_hq_final_exact"]),
    "gt_in_hq_init_category_pct": pct(eval_df["gt_in_hq_init_category"]),
    "gt_in_hq_final_category_pct": pct(eval_df["gt_in_hq_final_category"]),
    "mean_hq_init": round(eval_df["n_hypotheses_init"].mean(), 3),
    "mean_hq_final": round(eval_df["n_hypotheses_final"].mean(), 3),
    "mean_entropy_init": round(eval_df["entropy_init"].mean(), 4),
    "mean_entropy_final": round(eval_df["entropy_final"].mean(), 4),
    "mean_entropy_reduction": round(eval_df["entropy_reduction"].mean(), 4),
    "mean_turns": round(eval_df["turns_eval"].mean(), 3),
    "abstention_pct": pct(eval_df["abstained_eval"]),
    "top1_exact_pct": pct(eval_df["top1_exact_correct"]),
    "top5_exact_pct": pct(eval_df["top5_exact_correct"]),
    "top1_category_pct": pct(eval_df["top1_category_correct"]),
    "top5_category_pct": pct(eval_df["top5_category_correct"]),
    "single_pct": pct(eval_df["stop_reason_eval"] == "single"),
    "low_ig_pct": pct(eval_df["stop_reason_eval"] == "low_ig"),
    "t_max_pct": pct(eval_df["stop_reason_eval"] == "t_max"),
    "no_cand_pct": pct(eval_df["stop_reason_eval"] == "no_cand"),
    "stop_reason_counts": eval_df["stop_reason_eval"].value_counts(dropna=False).to_dict(),
}

OUTPUT_ROW_CSV.parent.mkdir(parents=True, exist_ok=True)
eval_df.to_csv(OUTPUT_ROW_CSV, index=False)

with open(OUTPUT_SUMMARY_JSON, "w") as f:
    json.dump(summary, f, indent=2)

print("\nTest summary")
print("=" * 60)
for k, v in summary.items():
    print(f"{k}: {v}")

print("\nSaved:")
print(OUTPUT_ROW_CSV)
print(OUTPUT_SUMMARY_JSON)