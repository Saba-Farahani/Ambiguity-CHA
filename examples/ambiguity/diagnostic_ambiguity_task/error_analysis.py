"""
Error analysis for symptom-diagnosis task.
Categorises every test case into one of four failure types
and quantifies each category.

Run from agent/CHA:
    python error_analysis.py
"""

import json, numpy as np
import pandas as pd

df  = pd.read_csv('results/diagnostic_ambiguity_task/test_pomdp_outputs.csv')
cat = pd.read_csv('data/diagnosis_category_map.csv')
df['family_name'] = df['family_name'].fillna('')

cat_map = {r['diagnosis'].strip().lower(): r['category'].strip().lower()
           for _, r in cat.iterrows()}

def norm(x): return ' '.join(str(x).strip().lower().split()) if x else ''
def get_cat(name): return cat_map.get(norm(name), '')
def parse5(x):
    try: return [norm(i) for i in json.loads(x)]
    except (json.JSONDecodeError, TypeError, ValueError): return []
def parse_list(x):
    try: return json.loads(x)
    except (json.JSONDecodeError, TypeError, ValueError): return []

df['gt_cat']   = df['ground_truth'].apply(get_cat)
df['pred_cat'] = df['prediction_top1'].apply(get_cat)
df['pred5']    = df['prediction_top5'].apply(
    lambda x: [get_cat(n) for n in parse5(x)])

# ── Column name detection ─────────────────────────────────────
# Try common names for the initial hypothesis list
HINIT_COL = None
for c in ['hypotheses_init', 'h_init', 'initial_hypotheses',
          'candidates_init', 'initial_candidates']:
    if c in df.columns:
        HINIT_COL = c
        break

HFINAL_COL = None
for c in ['hypotheses_final', 'h_final', 'final_hypotheses',
          'candidates_final', 'final_candidates']:
    if c in df.columns:
        HFINAL_COL = c
        break

print(f"Columns in CSV: {list(df.columns)}\n")
print(f"H_init column : {HINIT_COL}")
print(f"H_final column: {HFINAL_COL}\n")

total  = len(df)
errors = []  # one dict per error case

for _, r in df.iterrows():
    gt   = norm(r['ground_truth'])
    pred = norm(r.get('prediction_top1', ''))
    gt_c = r['gt_cat']
    pc   = r['pred_cat']
    state = r['agent_state']
    p5   = r['pred5']

    # ── Correct cases ──────────────────────────────────────────
    top1_correct = (gt_c != '' and pc == gt_c)
    r5_correct   = (gt_c != '' and gt_c in p5)

    if state == 'ABSTAIN':
        errors.append({'type': 'abstain', 'gt': gt, 'state': state})
        continue

    if top1_correct:
        continue  # correct — skip

    # ── Failure categorisation ─────────────────────────────────

    # 1. GT absent from H(0)
    gt_in_hinit = True  # assume present unless we can check
    if HINIT_COL and r.get(HINIT_COL):
        hinit = [norm(h) for h in parse_list(r[HINIT_COL])]
        gt_in_hinit = any(gt in h or h in gt for h in hinit)

    # Use n_hypotheses_init as proxy if list not available
    if HINIT_COL is None:
        # If GT not in H_init, the agent would have gotten 0%
        # We detect this via: gt_cat == '' (no category mapping found)
        # or n_hypotheses_init == 0
        n_init = r.get('n_hypotheses_init', -1)
        if n_init == 0:
            gt_in_hinit = False
        # For cases where n_init > 0, we can't tell without the list
        # Use heuristic: if gt_cat is empty, KG didn't find it
        if gt_c == '':
            gt_in_hinit = False

    if not gt_in_hinit:
        errors.append({
            'type': 'gt_absent_hinit',
            'gt': gt, 'state': state,
            'gt_cat': gt_c, 'pred_cat': pc
        })
        continue

    # 2. GT eliminated during clarification
    gt_in_hfinal = True
    if HFINAL_COL and r.get(HFINAL_COL):
        hfinal = [norm(h) for h in parse_list(r[HFINAL_COL])]
        gt_in_hfinal = any(gt in h or h in gt for h in hfinal)
    elif HINIT_COL is None:
        # proxy: if belief concentrated on wrong candidate
        # and answer state, candidate was likely eliminated
        if state == 'ANSWER' and not top1_correct:
            gt_in_hfinal = False  # belief concentrated on wrong one

    if not gt_in_hfinal and gt_in_hinit:
        errors.append({
            'type': 'gt_eliminated',
            'gt': gt, 'state': state,
            'gt_cat': gt_c, 'pred_cat': pc
        })
        continue

    # 3. Within-family near miss (correct family, wrong exact)
    if gt_c != '' and pc == gt_c and not top1_correct:
        errors.append({
            'type': 'within_family_miss',
            'gt': gt, 'state': state,
            'gt_cat': gt_c, 'pred_cat': pc
        })
        continue

    # Also: correct family in top-5 but not top-1
    if gt_c != '' and gt_c in p5 and pc != gt_c:
        errors.append({
            'type': 'within_family_miss',
            'gt': gt, 'state': state,
            'gt_cat': gt_c, 'pred_cat': pc
        })
        continue

    # 4. Wrong family prediction (candidate pollution / other)
    errors.append({
        'type': 'wrong_family',
        'gt': gt, 'state': state,
        'gt_cat': gt_c, 'pred_cat': pc
    })

# ── Summary ───────────────────────────────────────────────────
err_df = pd.DataFrame(errors)

# Count by type
correct   = total - len(errors)
abstains  = (df['agent_state'] == 'ABSTAIN').sum()

type_counts = err_df[err_df['type'] != 'abstain']['type'].value_counts()

print("=" * 65)
print("ERROR ANALYSIS SUMMARY")
print("=" * 65)
print(f"Total test cases  : {total}")
print(f"Correct (Top-1)   : {correct}  ({100*correct/total:.1f}%)")
print(f"ABSTAIN           : {abstains} ({100*abstains/total:.1f}%)")
print(f"Errors (excl. abs): {len(errors)-abstains} ({100*(len(errors)-abstains)/total:.1f}%)")
print()
print("-" * 65)
print("FAILURE TYPE BREAKDOWN (excluding ABSTAIN):")
print("-" * 65)

type_labels = {
    'gt_absent_hinit':   'GT absent from H(0)',
    'gt_eliminated':     'GT eliminated during clarification',
    'within_family_miss':'Within-family near miss',
    'wrong_family':      'Wrong family / candidate pollution',
}

for t, label in type_labels.items():
    n = type_counts.get(t, 0)
    pct_of_errors = 100*n/(len(errors)-abstains) if (len(errors)-abstains) > 0 else 0
    pct_of_total  = 100*n/total
    print(f"  {label:<38}: {n:4d}  "
          f"({pct_of_errors:.1f}% of errors, "
          f"{pct_of_total:.1f}% of total)")

print()
print("=" * 65)
print("PER-STATE ERROR BREAKDOWN")
print("=" * 65)
for state in ['ANSWER', 'PARTIAL']:
    sub = err_df[err_df['state'] == state]
    print(f"\n  {state} errors (n={len(sub)}):")
    for t, label in type_labels.items():
        n = (sub['type'] == t).sum()
        if n > 0:
            print(f"    {label:<38}: {n}")

print()
print("=" * 65)
print("TOP MISCLASSIFIED GROUND-TRUTH DIAGNOSES")
print("=" * 65)
err_nabs = err_df[err_df['type'] != 'abstain']
top_miss = err_nabs['gt'].value_counts().head(10)
for dx, cnt in top_miss.items():
    print(f"  {dx:<45}: {cnt} errors")

print()
print("NOTE: If GT_absent and GT_eliminated counts look off,")
print("check that your CSV has 'hypotheses_init'/'hypotheses_final'")
print("columns with the full candidate lists.")
print("Without them, categorisation uses proxies.")
print("\nDONE.")
