"""
compare_to_fp.py
Compare our model's 2026 projections against FantasyPoints published rankings.

Normalise both to FP/game for a fair comparison (FP projects ~15 games, we project 17).
Flag large rank disagreements and diagnose likely causes.
"""

import sys, os, re
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / 'outputs'

# ── Load ──────────────────────────────────────────────────────────────────────

our = pd.read_csv(OUTPUT_DIR / 'projections_2026.csv')
fp  = pd.read_csv(
    'C:/Users/mjgin/Downloads/2026 NFL Fantasy Football Season Rankings  Projections  Fantasy Points.csv',
    encoding='utf-8-sig'
)

# ── Clean FP file ─────────────────────────────────────────────────────────────

fp = fp[fp['POS'].isin(['WR', 'TE', 'RB'])].copy()
fp['FPTS']   = pd.to_numeric(fp['FPTS'],   errors='coerce')
fp['FPTS/G'] = pd.to_numeric(fp['FPTS/G'], errors='coerce')
fp['G']      = pd.to_numeric(fp['G'],      errors='coerce').fillna(15)

# Compute FP positional rank from their FPTS (descending within POS)
fp['FP_pos_rank'] = (
    fp.groupby('POS')['FPTS']
      .rank(ascending=False, method='min')
      .astype(int)
)
fp['FP_overall_rank'] = fp['RK']

# ── Name normalisation ────────────────────────────────────────────────────────

_SUFFIX_RE = re.compile(r"\s+(jr|sr|ii|iii|iv)$")

def norm(s):
    """Lowercase, strip punctuation and name suffixes for fuzzy matching."""
    s = str(s).strip().lower()
    s = s.replace("’", "’").replace("’", "’")
    s = s.replace(".", "").replace("  ", " ")
    # Strip suffixes so "Kenneth Walker III" == "kenneth walker"
    s = _SUFFIX_RE.sub("", s.strip())
    return s

our['name_key'] = our['Name'].apply(norm)
fp['name_key']  = fp['Name'].apply(norm)

# ── Merge ─────────────────────────────────────────────────────────────────────

fp_slim = fp[['name_key', 'POS', 'FP_pos_rank', 'FP_overall_rank',
              'FPTS', 'FPTS/G', 'G', 'Team']].rename(columns={
    'FPTS':   'FP_fpts',
    'FPTS/G': 'FP_fptsg',
    'G':      'FP_games',
    'Team':   'FP_team',
})

# Use adjusted FP if available (post expert adjustments), else base projection
fp_col = 'Proj_FP_Adj' if 'Proj_FP_Adj' in our.columns else 'Proj_FP_2026'
our['our_fptsg']    = (our[fp_col] / 17).round(2)
our['our_pos_rank'] = our['POS_Rank'].astype(int)

merged = our.merge(fp_slim, on=['name_key', 'POS'], how='outer')

# Fill names for FP-only rows
merged['Name'] = merged['Name'].fillna(merged['name_key'].str.title())
merged['Team'] = merged['Team'].fillna(merged['FP_team'])
merged['POS']  = merged['POS']

# ── Rank difference (positive = we rank higher than FP) ──────────────────────

merged['rank_diff'] = merged['FP_pos_rank'] - merged['our_pos_rank']
merged['abs_diff']  = merged['rank_diff'].abs()

def verdict(row):
    if pd.isna(row.get('FP_pos_rank')) and pd.notna(row.get('our_pos_rank')):
        return 'our_only'
    if pd.notna(row.get('FP_pos_rank')) and pd.isna(row.get('our_pos_rank')):
        return 'fp_only'
    if pd.isna(row.get('rank_diff')):
        return 'no_match'
    d = row['rank_diff']
    if d >=  8: return 'we_rank_higher'
    if d <= -8: return 'fp_ranks_higher'
    return 'agreement'

merged['verdict'] = merged.apply(verdict, axis=1)

# Save full comparison
save_cols = ['POS', 'Name', 'Team', 'our_pos_rank', 'FP_pos_rank', 'rank_diff',
             'our_fptsg', 'FP_fptsg', 'Proj_FP_2026', 'FP_fpts', 'FP_games',
             'Games', 'OC_Stable', 'FP_overall_rank', 'verdict']
merged[[c for c in save_cols if c in merged.columns]] \
    .sort_values(['POS', 'our_pos_rank']) \
    .to_csv(OUTPUT_DIR / 'comparison_vs_fp.csv', index=False)

# ── Print helpers ─────────────────────────────────────────────────────────────

HDR = f"  {'Name':<24} {'Ours':>5} {'FP':>5} {'Diff':>6}  {'Our/G':>6} {'FP/G':>6}  {'25 FP':>7} {'25G':>4} {'OC':>3}"
SEP = f"  {'-'*24} {'-'*5} {'-'*5} {'-'*6}  {'-'*6} {'-'*6}  {'-'*7} {'-'*4} {'-'*3}"

def fmt_row(r, pos_col_our='our_pos_rank', pos_col_fp='FP_pos_rank'):
    our_r  = f"{int(r[pos_col_our])}"  if pd.notna(r.get(pos_col_our))  else '--'
    fp_r   = f"{int(r[pos_col_fp])}"   if pd.notna(r.get(pos_col_fp))   else '--'
    diff   = f"{int(r['rank_diff'])}"  if pd.notna(r.get('rank_diff'))  else '--'
    oFPG   = f"{r['our_fptsg']:.2f}"   if pd.notna(r.get('our_fptsg'))  else '--'
    fFPG   = f"{r['FP_fptsg']:.2f}"    if pd.notna(r.get('FP_fptsg'))   else '--'
    fp25   = f"{r['FP_total']:.1f}"    if pd.notna(r.get('FP_total'))   else '--'
    g25    = f"{int(r['Games'])}"      if pd.notna(r.get('Games'))      else '--'
    oc     = f"{int(r['OC_Stable'])}"  if pd.notna(r.get('OC_Stable'))  else '-'
    return (f"  {str(r['Name']):<24} {our_r:>5} {fp_r:>5} {diff:>6}  "
            f"{oFPG:>6} {fFPG:>6}  {fp25:>7} {g25:>4} {oc:>3}")

# ── Summary ───────────────────────────────────────────────────────────────────

print("=" * 70)
print("  MODEL COMPARISON  —  Our Projections vs FantasyPoints 2026")
print("=" * 70)

for pos in ['WR', 'TE', 'RB']:
    sub = merged[merged['POS'] == pos]
    counts = sub['verdict'].value_counts()
    print(f"\n  {pos}  "
          f"agree={counts.get('agreement',0)}  "
          f"fp_higher={counts.get('fp_ranks_higher',0)}  "
          f"we_higher={counts.get('we_rank_higher',0)}  "
          f"fp_only={counts.get('fp_only',0)}  "
          f"our_only={counts.get('our_only',0)}")

# ── FP much more bullish (they're higher on these players) ───────────────────
print("\n\n" + "=" * 70)
print("  FP RANKS SIGNIFICANTLY HIGHER  (we are too low on these players)")
print("=" * 70)
for pos in ['WR', 'TE', 'RB']:
    sub = (merged[(merged['POS'] == pos) & (merged['verdict'] == 'fp_ranks_higher')]
           .sort_values('rank_diff'))
    if sub.empty:
        continue
    print(f"\n  {pos}:   Ours → FP → Diff   Our/G   FP/G   2025FP  G  OC")
    print(SEP)
    for _, r in sub.iterrows():
        print(fmt_row(r))

# ── We are much more bullish ──────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("  WE RANK SIGNIFICANTLY HIGHER  (we may be overvaluing these players)")
print("=" * 70)
for pos in ['WR', 'TE', 'RB']:
    sub = (merged[(merged['POS'] == pos) & (merged['verdict'] == 'we_rank_higher')]
           .sort_values('rank_diff', ascending=False))
    if sub.empty:
        continue
    print(f"\n  {pos}:   Ours → FP → Diff   Our/G   FP/G   2025FP  G  OC")
    print(SEP)
    for _, r in sub.iterrows():
        print(fmt_row(r))

# ── FP top-24 that we missed or rank 25+ ─────────────────────────────────────
print("\n\n" + "=" * 70)
print("  FP TOP-24 WE RANK 25+ OR HAVE NO PROJECTION FOR")
print("  (likely new players, team changes, or structural model gaps)")
print("=" * 70)
for pos in ['WR', 'TE', 'RB']:
    sub = merged[
        (merged['POS'] == pos) &
        (merged['FP_pos_rank'] <= 24) &
        ((merged['our_pos_rank'] > 24) | merged['our_pos_rank'].isna())
    ].sort_values('FP_pos_rank')
    if sub.empty:
        continue
    print(f"\n  {pos}:")
    print(SEP)
    for _, r in sub.iterrows():
        print(fmt_row(r))

# ── Agreement zone: top-12 per position ──────────────────────────────────────
print("\n\n" + "=" * 70)
print("  AGREEMENT ZONE  —  Top 12 per position, ranked by our model")
print("=" * 70)
for pos in ['WR', 'TE', 'RB']:
    sub = (merged[
        (merged['POS'] == pos) &
        merged['our_pos_rank'].notna() &
        merged['FP_pos_rank'].notna()
    ].sort_values('our_pos_rank').head(12))
    print(f"\n  {pos}:   Ours → FP → Diff   Our/G   FP/G   2025FP  G  OC")
    print(SEP)
    for _, r in sub.iterrows():
        print(fmt_row(r))

print(f"\n\nFull table saved -> outputs/comparison_vs_fp.csv")
