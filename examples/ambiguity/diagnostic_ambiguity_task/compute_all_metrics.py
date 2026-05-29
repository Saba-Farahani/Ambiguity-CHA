"""
Compute ALL metrics needed for the complete Results section.
Outputs every number needed to fill the tables.
"""
import json, ast, math
import pandas as pd

df  = pd.read_csv('results/diagnostic_ambiguity_task/test_pomdp_outputs.csv')
cat = pd.read_csv('data/diagnosis_category_map.csv')

df['family_name'] = df['family_name'].fillna('')

cat_map = {
    r['diagnosis'].strip().lower(): r['category'].strip().lower()
    for _, r in cat.iterrows()
}

def norm(x):
    return ' '.join(str(x).strip().lower().split()) if x else ''

def get_cat(name):
    return cat_map.get(norm(name), '')

def parse5(x):
    try: return [norm(i) for i in json.loads(x)]
    except (json.JSONDecodeError, TypeError, ValueError): return []

df['gt_cat']     = df['ground_truth'].apply(get_cat)
df['pred_cat']   = df['prediction_top1'].apply(get_cat)
df['pred5_cats'] = df['prediction_top5'].apply(
    lambda x: [get_cat(n) for n in parse5(x)])

total     = len(df)
answered  = df[df['prediction_top1'] != 'ABSTAIN']
abstained = df[df['agent_state'] == 'ABSTAIN']

# ── Overall metrics ───────────────────────────────────────────
c1_all = sum(r['pred_cat']!='' and r['pred_cat']==r['gt_cat']
             for _,r in df.iterrows())
c5_all = sum(r['gt_cat']!='' and r['gt_cat'] in r['pred5_cats']
             for _,r in df.iterrows())

print("="*60)
print("OVERALL (all 1034, ABSTAIN = wrong)")
print("="*60)
print(f"  Top-1 Category  : {100*c1_all/total:.2f}%  ({c1_all}/{total})")
print(f"  Recall@5 Cat    : {100*c5_all/total:.2f}%  ({c5_all}/{total})")

# ── Per state ─────────────────────────────────────────────────
print("\n" + "="*60)
print("BY AGENT STATE")
print("="*60)
for state in ['ANSWER','PARTIAL']:
    sub = df[df['agent_state']==state]
    c1  = sum(r['pred_cat']!='' and r['pred_cat']==r['gt_cat']
              for _,r in sub.iterrows())
    c5  = sum(r['gt_cat']!='' and r['gt_cat'] in r['pred5_cats']
              for _,r in sub.iterrows())
    n   = len(sub)
    print(f"\n  {state} (n={n}, {100*n/total:.1f}%):")
    print(f"    Top-1 Category  : {100*c1/n:.1f}%")
    print(f"    Recall@5 Cat    : {100*c5/n:.1f}%")

print(f"\n  ABSTAIN (n={len(abstained)}, {100*len(abstained)/total:.1f}%):")
print(f"    Language model not called.")

# ── Ambiguity reduction ───────────────────────────────────────
print("\n" + "="*60)
print("AMBIGUITY REDUCTION")
print("="*60)
gt_in_init = (df['n_hypotheses_init'] > 0).mean() * 100
print(f"  GT in H_init (%): {gt_in_init:.1f}")
print(f"  Mean |H_init|   : {df['n_hypotheses_init'].mean():.3f}")
print(f"  Mean |H_final|  : {df['n_hypotheses_final'].mean():.3f}")
print(f"  Mean H(H_init)  : {df['entropy_init'].mean():.4f} bits")
print(f"  Mean H(H_final) : {df['entropy_final'].mean():.4f} bits")
print(f"  Mean H reduction: {(df['entropy_init']-df['entropy_final']).mean():.4f} bits")
print(f"  Mean turns      : {df['turns'].mean():.3f}")

# ── Stopping ──────────────────────────────────────────────────
print("\n" + "="*60)
print("STOPPING CONDITIONS")
print("="*60)
sc = df['stop_reason'].value_counts()
for reason, count in sc.items():
    print(f"  {reason:30s}: {count:4d} ({100*count/total:.1f}%)")

# ── Abstain breakdown ─────────────────────────────────────────
print("\n" + "="*60)
print("ABSTAIN BREAKDOWN")
print("="*60)
print(abstained['ground_truth'].value_counts().to_string())

# ── Per-diagnosis performance ─────────────────────────────────
print("\n" + "="*60)
print("PER-DIAGNOSIS PERFORMANCE (category level, all cases)")
print("="*60)
answered_df = df[df['agent_state'] != 'ABSTAIN']
dx_perf = []
for dx in sorted(df['ground_truth'].unique()):
    sub = df[df['ground_truth']==dx]
    c1  = sum(r['pred_cat']!='' and r['pred_cat']==r['gt_cat']
              for _,r in sub.iterrows())
    c5  = sum(r['gt_cat']!='' and r['gt_cat'] in r['pred5_cats']
              for _,r in sub.iterrows())
    n   = len(sub)
    n_abs = (sub['agent_state']=='ABSTAIN').sum()
    dx_perf.append({'diagnosis':dx,'n':n,'top1':100*c1/n,
                    'r5':100*c5/n,'abstain':n_abs})

dx_df = pd.DataFrame(dx_perf).sort_values('top1',ascending=False)
for _,r in dx_df.iterrows():
    print(f"  {r['diagnosis'][:40]:<40} n={r['n']:4d} "
          f"Top1={r['top1']:5.1f}% R@5={r['r5']:5.1f}% "
          f"Abs={r['abstain']}")

# ── Answered-only metrics ─────────────────────────────────────
print("\n" + "="*60)
print("ANSWERED CASES ONLY (987 cases)")
print("="*60)
an = df[df['agent_state'] != 'ABSTAIN']
c1a = sum(r['pred_cat']!='' and r['pred_cat']==r['gt_cat']
          for _,r in an.iterrows())
c5a = sum(r['gt_cat']!='' and r['gt_cat'] in r['pred5_cats']
          for _,r in an.iterrows())
print(f"  Top-1 Category  : {100*c1a/len(an):.2f}%")
print(f"  Recall@5 Cat    : {100*c5a/len(an):.2f}%")

# ── Belief stats by state ─────────────────────────────────────
print("\n" + "="*60)
print("BELIEF STATS BY STATE")
print("="*60)
for state in ['ANSWER','PARTIAL','ABSTAIN']:
    sub = df[df['agent_state']==state]
    print(f"  {state}: mean max_belief={sub['max_belief'].mean():.3f}  "
          f"mean turns={sub['turns'].mean():.2f}")

print("\nDONE — all metrics computed.")
