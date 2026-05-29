"""
Sensitivity analysis on delta (belief concentration threshold).
Tests delta in {0.01, 0.05, 0.10, 0.20, 0.30, 0.50}.

Does NOT re-run the agent. Re-applies stopping logic to the
existing per-turn belief trajectory stored in the results CSV.

NOTE: This script requires 'belief_history' column in your CSV.
If that column does not exist, it uses a simpler approximation:
re-applies Condition 1 threshold to max_belief at termination.

Run from agent/CHA:
    python sensitivity_delta.py
"""

import json, numpy as np, pandas as pd

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

df['gt_cat']   = df['ground_truth'].apply(get_cat)
df['pred_cat'] = df['prediction_top1'].apply(get_cat)
df['pred5']    = df['prediction_top5'].apply(
    lambda x: [get_cat(n) for n in parse5(x)])

total = len(df)

# Check if belief_history column exists
has_history = 'belief_history' in df.columns

print(f"belief_history column found: {has_history}")
print("Using approximation: re-apply delta to final max_belief\n")

# Approximation: if max_belief >= 1-delta AND original state was
# ANSWER or PARTIAL, it would still be ANSWER under new delta.
# If original state was ANSWER but max_belief < 1-delta under new
# delta, it becomes PARTIAL (if family exists) or ABSTAIN.
# PARTIAL cases: max_belief = 0.375, already stopped at zero_ig.
# ABSTAIN cases: max_belief = 0.047, already stopped at zero_ig.

def simulate_delta(delta):
    rows = []
    for _, r in df.iterrows():
        orig = r['agent_state']
        mb   = r['max_belief']

        if orig == 'ANSWER':
            # With new delta: would Condition 1 have triggered?
            if mb > 1 - delta:
                new_state = 'ANSWER'
            else:
                # belief didn't concentrate enough — falls to PARTIAL or ABSTAIN
                # family_name tells us if IS_A family was found
                fn = str(r.get('family_name', '')).strip()
                new_state = 'PARTIAL' if fn and fn != 'nan' else 'ABSTAIN'

        elif orig == 'PARTIAL':
            # These stopped at zero_ig regardless of delta
            new_state = 'PARTIAL'

        else:  # ABSTAIN
            new_state = 'ABSTAIN'

        rows.append(new_state)

    df2 = df.copy()
    df2['state_new'] = rows

    c1 = sum(r['pred_cat']!='' and r['pred_cat']==r['gt_cat']
             and r2!='ABSTAIN'
             for (_,r), r2 in zip(df.iterrows(), rows))
    c5 = sum(r['gt_cat']!='' and r['gt_cat'] in r['pred5']
             and r2!='ABSTAIN'
             for (_,r), r2 in zip(df.iterrows(), rows))

    mrr_vals = []
    for (_, r), r2 in zip(df.iterrows(), rows):
        if r2 == 'ABSTAIN' or r['gt_cat'] == '':
            mrr_vals.append(0.0); continue
        if r['pred_cat'] == r['gt_cat']:
            mrr_vals.append(1.0)
        else:
            rank = next((i+2 for i,c in enumerate(r['pred5'])
                        if c==r['gt_cat']), None)
            mrr_vals.append(1/rank if rank else 0.0)

    mrr = np.mean(mrr_vals)
    n_ans = rows.count('ANSWER')
    n_par = rows.count('PARTIAL')
    n_abs = rows.count('ABSTAIN')

    return {
        'delta': delta,
        'top1':  100*c1/total,
        'r5':    100*c5/total,
        'mrr':   mrr,
        'answer':  100*n_ans/total,
        'partial': 100*n_par/total,
        'abstain': 100*n_abs/total,
    }

deltas = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]

print("="*78)
print("SENSITIVITY ANALYSIS ON DELTA (belief concentration threshold)")
print("="*78)
print(f"{'delta':>7} {'Top-1':>8} {'R@5':>8} {'MRR':>8} "
      f"{'ANSWER%':>9} {'PARTIAL%':>10} {'ABSTAIN%':>10}")
print("-"*78)

results = []
for d in deltas:
    r = simulate_delta(d)
    results.append(r)
    marker = " <-- paper" if d == 0.05 else ""
    print(f"  {d:.2f}   {r['top1']:>7.2f}%  {r['r5']:>7.2f}%"
          f"  {r['mrr']:>7.4f}  {r['answer']:>8.1f}%"
          f"  {r['partial']:>9.1f}%  {r['abstain']:>9.1f}%{marker}")

print("="*78)
print("\nDONE. Paste this output to Claude for LaTeX table + figure.")
