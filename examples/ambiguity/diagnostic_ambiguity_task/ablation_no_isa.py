"""
Ablation: remove IS_A hierarchy so every Condition-2 termination
becomes ABSTAIN instead of PARTIAL.

Run from agent/CHA directory:
    python ablation_no_isa.py
"""

import json, numpy as np
import pandas as pd

df  = pd.read_csv('results/diagnostic_ambiguity_task/test_pomdp_outputs.csv')
cat = pd.read_csv('data/diagnosis_category_map.csv')
df['family_name'] = df['family_name'].fillna('')

cat_map = {
    r['diagnosis'].strip().lower(): r['category'].strip().lower()
    for _, r in cat.iterrows()
}

def norm(x): return ' '.join(str(x).strip().lower().split()) if x else ''
def get_cat(name): return cat_map.get(norm(name), '')
def parse5(x):
    try: return [norm(i) for i in json.loads(x)]
    except (json.JSONDecodeError, TypeError, ValueError): return []

df['gt_cat']   = df['ground_truth'].apply(get_cat)
df['pred_cat'] = df['prediction_top1'].apply(get_cat)
df['pred5']    = df['prediction_top5'].apply(
    lambda x: [get_cat(n) for n in parse5(x)])

# ABLATION: PARTIAL -> ABSTAIN
df['state_abl'] = df['agent_state'].apply(
    lambda s: 'ABSTAIN' if s == 'PARTIAL' else s
)

total = len(df)

def compute(sub, state_col):
    n = len(sub)
    c1 = sum(r['pred_cat']!='' and r['pred_cat']==r['gt_cat']
             and r[state_col]!='ABSTAIN' for _,r in sub.iterrows())
    c5 = sum(r['gt_cat']!='' and r['gt_cat'] in r['pred5']
             and r[state_col]!='ABSTAIN' for _,r in sub.iterrows())
    mrr_vals = []
    for _,r in sub.iterrows():
        if r[state_col]=='ABSTAIN' or r['gt_cat']=='':
            mrr_vals.append(0.0); continue
        if r['pred_cat']==r['gt_cat']:
            mrr_vals.append(1.0)
        else:
            rank = next((i+2 for i,c in enumerate(r['pred5'])
                        if c==r['gt_cat']), None)
            mrr_vals.append(1/rank if rank else 0.0)
    n_abs = (sub[state_col]=='ABSTAIN').sum()
    return 100*c1/n, 100*c5/n, np.mean(mrr_vals), n_abs

t1_f,  r5_f,  mrr_f,  abs_f  = compute(df, 'agent_state')
t1_a,  r5_a,  mrr_a,  abs_a  = compute(df, 'state_abl')

print("\n" + "="*68)
print("ABLATION SUMMARY TABLE")
print("="*68)
print(f"{'Method':<38} {'Top-1':>7} {'R@5':>7} {'MRR':>7} {'Abstain':>9}")
print("-"*68)
print(f"{'Full model (with IS_A)':<38}"
      f" {t1_f:>6.2f}% {r5_f:>6.2f}% {mrr_f:>7.4f}"
      f" {abs_f:>5} ({100*abs_f/total:.1f}%)")
print(f"{'No IS_A  (PARTIAL -> ABSTAIN)':<38}"
      f" {t1_a:>6.2f}% {r5_a:>6.2f}% {mrr_a:>7.4f}"
      f" {abs_a:>5} ({100*abs_a/total:.1f}%)")
print("-"*68)
print(f"{'IS_A contribution (delta)':<38}"
      f" {t1_f-t1_a:>+6.2f}% {r5_f-r5_a:>+6.2f}% {mrr_f-mrr_a:>+7.4f}"
      f" {abs_f-abs_a:>+5} cases")

print("\n" + "="*68)
print("STATE DISTRIBUTION")
print("="*68)
print(f"{'State':<12} {'Full model':>16} {'No IS_A':>16}")
for s in ['ANSWER','PARTIAL','ABSTAIN']:
    f = (df['agent_state']==s).sum()
    a = (df['state_abl']==s).sum()
    print(f"  {s:<10} {f:>6} ({100*f/total:.1f}%)   {a:>6} ({100*a/total:.1f}%)")

print("\nDONE.")
