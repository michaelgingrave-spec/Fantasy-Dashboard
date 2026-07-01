"""
main.py
Fantasy Football Best Ball Projection Model — Full Pipeline

Outputs (written to outputs/):
  projections_2026.csv          — Season FP draft rankings for 2026
  weekly_schedule_2026.csv      — Week-by-week matchup scores + boom probs (all 18 weeks)
  boom_summary_2025.csv         — Per-player boom week summary (2025 back-test)
  boom_weekly_2025.csv          — Per-week boom probabilities (2025 back-test)
  cv_metrics.txt                — Cross-validation results
  feature_importance.csv        — XGBoost feature importances
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from pathlib import Path

from data_loader  import (load_receiving, load_rushing, load_passing,
                          load_man_vs_zone, load_coverage_matrix,
                          load_coordinators, load_2026_schedule,
                          get_combined_rb_stats, get_season_stats,
                          get_qb_season_stats)
from features     import build_yoy_pairs, build_weekly_features
from season_model import train_and_evaluate, project_2026, get_feature_importance
from boom_model   import train_boom_models, project_boom_weeks
from matchup           import (build_smoothed_def_profiles,
                               build_2026_weekly_schedule,
                               build_oc_continuity)
from expert_adjustments import apply_all_adjustments

OUTPUT_DIR = Path(__file__).parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def save_cv_metrics(trained_season_models, trained_boom_models):
    lines = []

    lines.append("=" * 60)
    lines.append("SEASON FP MODEL  --  Leave-One-Year-Out CV")
    lines.append("=" * 60)
    for pos in ['WR', 'TE', 'RB']:
        r = trained_season_models[pos]
        lines.append(f"\n{pos}  (n_train={r['n_train']})")
        lines.append(f"  {'Year':>6}  {'N':>5}  {'MAE':>7}  {'R2':>6}")
        lines.append(f"  {'-'*6}  {'-'*5}  {'-'*7}  {'-'*6}")
        for row in r['cv_results']:
            lines.append(
                f"  {row['holdout_year']:>6}  {row['n_test']:>5}"
                f"  {row['MAE']:>7.1f}  {row['R2']:>6.3f}")

    lines.append("\n\n" + "=" * 60)
    lines.append("BOOM WEEK MODEL  --  Leave-One-Year-Out CV")
    lines.append("(Boom = weekly FP >= position 90th percentile)")
    lines.append("=" * 60)
    for pos in ['WR', 'TE', 'RB']:
        r = trained_boom_models[pos]
        thresh = r['boom_threshold']
        lines.append(f"\n{pos}  (threshold >= {thresh} FP)")
        lines.append(f"  {'Year':>6}  {'N':>6}  {'Boom%':>6}  {'AUC':>6}  {'AvgPrec':>8}")
        lines.append(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}")
        for row in r['cv_results']:
            lines.append(
                f"  {row['holdout_year']:>6}  {row['n_test']:>6}"
                f"  {row['boom_rate']:>6.1%}  {row['AUC']:>6.3f}"
                f"  {row['AvgPrecision']:>8.3f}")

    text = "\n".join(lines)
    print(text)
    with open(OUTPUT_DIR / 'cv_metrics.txt', 'w') as f:
        f.write(text)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():

    # ── 1. Load all data ──────────────────────────────────────────────────────
    print_section("Loading data")

    receiving    = load_receiving()
    rushing      = load_rushing()
    passing      = load_passing()
    mvz          = load_man_vs_zone()
    coverage_mat = load_coverage_matrix()
    coordinators = load_coordinators()
    schedule_26  = load_2026_schedule()

    print(f"  Receiving rows : {len(receiving):,}")
    print(f"  Rushing rows   : {len(rushing):,}")
    print(f"  Passing rows   : {len(passing):,}")
    print(f"  ManvsZone rows : {len(mvz):,}  (season-level)")
    print(f"  Coverage matrix: {len(coverage_mat):,}  (team-season rows)")
    print(f"  Coordinators   : {len(coordinators):,}")
    print(f"  2026 schedule  : {len(schedule_26):,} games")

    # ── 2. Combine RB rushing + receiving ─────────────────────────────────────
    print_section("Combining RB rushing + receiving")
    rb_combined = get_combined_rb_stats(receiving, rushing)
    print(f"  RB combined weekly rows: {len(rb_combined):,}")

    # ── 3. Season aggregation ─────────────────────────────────────────────────
    print_section("Aggregating season stats")
    season_stats = get_season_stats(receiving, rb_combined)
    print(f"  Season-player rows: {len(season_stats):,}")
    for pos in ['WR', 'TE', 'RB']:
        n = len(season_stats[season_stats['POS'] == pos])
        print(f"    {pos}: {n}")

    qb_stats = get_qb_season_stats(passing)
    print(f"  QB season rows: {len(qb_stats):,}")

    # ── 4. Coordinator / OC continuity ───────────────────────────────────────
    print_section("OC continuity 2025->2026")
    oc_continuity = build_oc_continuity(coordinators)
    changed = oc_continuity[oc_continuity['oc_same_2526'] == 0]
    print(f"  Teams with OC change 2025->2026: {len(changed)}")
    if len(changed):
        for _, r in changed.iterrows():
            print(f"    {r['team_abbrev']:4s}  "
                  f"{r.get('oc_2025','?')} -> {r.get('oc_2026','?')}")

    # ── 5. Season projection model ────────────────────────────────────────────
    print_section("Building YoY training pairs")
    pairs = build_yoy_pairs(season_stats, mvz=mvz, oc_df=oc_continuity)
    print(f"  Total pairs: {len(pairs):,}")
    for pos in ['WR', 'TE', 'RB']:
        print(f"    {pos}: {len(pairs[pairs['POS']==pos]):,}")

    print_section("Training season projection models (XGBoost, LOYO-CV)")
    trained_season = {}
    for pos in ['WR', 'TE', 'RB']:
        print(f"  Training {pos}...", end='  ')
        result = train_and_evaluate(pairs, pos)
        trained_season[pos] = result
        avg_mae = np.mean([r['MAE'] for r in result['cv_results']])
        avg_r2  = np.mean([r['R2']  for r in result['cv_results']])
        print(f"avg MAE={avg_mae:.1f}  avg R2={avg_r2:.3f}")

    # ── 6. 2026 projections ───────────────────────────────────────────────────
    print_section("Generating 2026 draft rankings")

    # Attach OC continuity to 2025 stats before projecting
    stats_2025 = season_stats[season_stats['Year'] == 2025].copy()
    stats_2025 = stats_2025.merge(
        oc_continuity[['team_abbrev', 'oc_same_2526', 'pc_same_2526']],
        left_on='Team', right_on='team_abbrev', how='left'
    )
    stats_2025['oc_same_2526'] = stats_2025['oc_same_2526'].fillna(1)

    # Attach coverage splits to 2025 stats
    mvz_25 = mvz[mvz['Year'] == 2025][
        ['Name', 'POS', 'man_YPRR', 'zone_YPRR', '1hi_YPRR', '2hi_YPRR',
         'man_FPRR', 'zone_FPRR', 'ov_YPRR']
    ]
    stats_2025 = stats_2025.merge(mvz_25, on=['Name', 'POS'], how='left')

    # Attach trend features: compare 2025 per-game stats vs 2024 per-game stats
    stats_2024 = season_stats[season_stats['Year'] == 2024][
        ['Name', 'POS', 'FPG_mean', 'XFPG_mean', 'TGT_per_game']
    ].rename(columns={
        'FPG_mean': 'prior_FPG', 'XFPG_mean': 'prior_XFPG',
        'TGT_per_game': 'prior_TGT'
    })
    stats_2025 = stats_2025.merge(stats_2024, on=['Name', 'POS'], how='left')
    stats_2025['FPG_trend']  = (stats_2025['FPG_mean']
                                 - stats_2025['prior_FPG'].fillna(stats_2025['FPG_mean']))
    stats_2025['XFPG_trend'] = (stats_2025['XFPG_mean']
                                 - stats_2025['prior_XFPG'].fillna(stats_2025['XFPG_mean']))
    stats_2025['TGT_trend']  = (stats_2025['TGT_per_game']
                                 - stats_2025['prior_TGT'].fillna(stats_2025['TGT_per_game']))
    stats_2025 = stats_2025.drop(
        columns=['prior_FPG', 'prior_XFPG', 'prior_TGT'], errors='ignore'
    )

    # Drop players with fewer than 6 games — insufficient data to project
    # (catches 1-5 game backups whose per-game FP_per17 is wildly inflated)
    stats_2025 = stats_2025[stats_2025['Games'] >= 6].copy()
    print(f"  Players with 6+ games in 2025: {len(stats_2025)} "
          f"({len(season_stats[season_stats['Year']==2025]) - len(stats_2025)} removed as small-sample)")

    projections = project_2026(stats_2025, trained_season,
                               season_stats_full=season_stats, verbose=True)

    # Merge OC continuity flag into projections for display
    projections = projections.merge(
        oc_continuity[['team_abbrev', 'oc_same_2526']],
        left_on='Team', right_on='team_abbrev', how='left'
    )
    projections['OC_Stable'] = projections['oc_same_2526'].fillna(1).astype(int)
    projections = projections.drop(columns=['team_abbrev', 'oc_same_2526'], errors='ignore')

    # ── Apply expert adjustments (scheme transfers + manual takes) ────────────
    print_section("Applying expert adjustments")
    projections = apply_all_adjustments(projections, season_stats, verbose=True)

    # ── Assign positional tiers based on natural gaps in Proj_FP_Adj ────────────
    def assign_tiers(group: pd.DataFrame) -> pd.Series:
        """
        Gap-based tier assignment per position, capped at 18 tiers (matching FP).

        Gap threshold scales with projected FP so that low-value depth players
        are grouped into wide tiers rather than hundreds of micro-tiers:
          ≥ 200 FP  → break at  7% gap or 15 FP absolute
          ≥ 100 FP  → break at 10% gap or 12 FP absolute
          ≥  50 FP  → break at 18% gap or 20 FP absolute
          <  50 FP  → break at 35% gap or 30 FP absolute (large buckets at bottom)

        Max tier size: 8 for tiers 1-5, 12 thereafter.
        All players beyond tier 18 are capped at tier 18.
        """
        MAX_TIERS  = 18
        sorted_g   = group.sort_values('Proj_FP_Adj', ascending=False).copy()
        fps        = sorted_g['Proj_FP_Adj'].values
        tiers      = []
        current    = 1
        tier_count = 0

        for i, fp in enumerate(fps):
            if i == 0:
                tiers.append(current)
                tier_count = 1
                continue

            prev_fp  = fps[i - 1]
            gap      = prev_fp - fp
            pct_gap  = gap / prev_fp if prev_fp > 0 else 0

            if   prev_fp >= 200: min_gap, min_pct = 15, 0.07
            elif prev_fp >= 100: min_gap, min_pct = 12, 0.10
            elif prev_fp >=  50: min_gap, min_pct = 20, 0.18
            else:                min_gap, min_pct = 30, 0.35

            size_cap  = 8 if current <= 5 else 12
            gap_break = (gap >= min_gap or pct_gap >= min_pct)
            size_break= (tier_count >= size_cap)

            if (gap_break or size_break) and current < MAX_TIERS:
                current    += 1
                tier_count  = 0

            tiers.append(current)
            tier_count += 1

        sorted_g['Tier'] = tiers
        return sorted_g['Tier']

    projections['Tier'] = (
        projections.groupby('POS', group_keys=False)
                   .apply(assign_tiers)
    )

    projections.to_csv(OUTPUT_DIR / 'projections_2026.csv', index=False)
    print(f"\n  {len(projections)} players saved -> outputs/projections_2026.csv")

    print("\n  Top 25 overall projected 2026 (post-adjustment):")
    top25 = projections.sort_values('Overall_Rank').head(25)
    print(top25[['Overall_Rank', 'Name', 'Team', 'POS', 'POS_Rank',
                 'Proj_FP_Adj', 'Total_Adj', 'OC_Stable',
                 'TGT_per_game', 'aDOT_mean']].to_string(index=False))

    # ── 7. Defensive profiles for matchup scoring ─────────────────────────────
    print_section("Building defensive matchup profiles")
    def_profiles = build_smoothed_def_profiles(coverage_mat, alpha=0.6)
    print(f"  Defensive profiles built for {len(def_profiles)} teams")

    # Spot-check
    for team in ['KC', 'SF', 'BLT', 'TB']:
        if team in def_profiles['team_abbrev'].values:
            row = def_profiles[def_profiles['team_abbrev'] == team].iloc[0]
            print(f"    {team:4s}  MAN={row.get('MAN %', '?'):.1f}%  "
                  f"ZONE={row.get('ZONE %', '?'):.1f}%  "
                  f"zone_FPDB={row.get('zone_FPDB', '?'):.3f}")

    # ── 8. 2026 weekly schedule with matchup scores ───────────────────────────
    print_section("Building 2026 weekly schedule + matchup scores")
    weekly_26 = build_2026_weekly_schedule(
        schedule_26, projections, def_profiles, mvz
    )
    weekly_26.to_csv(OUTPUT_DIR / 'weekly_schedule_2026.csv', index=False)
    print(f"  {len(weekly_26):,} player-week rows -> outputs/weekly_schedule_2026.csv")

    # Best matchup weeks per position
    print("\n  Top matchup weeks in 2026 (WR, matchup_adv > 15%):")
    top_wr = (weekly_26[(weekly_26['POS'] == 'WR') &
                         (weekly_26['matchup_adv'] >= 0.15)]
              .sort_values('matchup_adv', ascending=False)
              .head(10))
    if not top_wr.empty:
        print(top_wr[['Name', 'Team', 'Week', 'Opponent',
                       'matchup_adv', 'def_man_pct', 'def_fpdb',
                       'week_proj_FP']].to_string(index=False))

    print("\n  Top matchup weeks in 2026 (TE, matchup_adv > 15%):")
    top_te = (weekly_26[(weekly_26['POS'] == 'TE') &
                         (weekly_26['matchup_adv'] >= 0.15)]
              .sort_values('matchup_adv', ascending=False)
              .head(10))
    if not top_te.empty:
        print(top_te[['Name', 'Team', 'Week', 'Opponent',
                       'matchup_adv', 'def_man_pct', 'def_fpdb',
                       'week_proj_FP']].to_string(index=False))

    # ── 9. Boom week model ────────────────────────────────────────────────────
    print_section("Training boom week models (XGBoost, LOYO-CV)")

    # Build unified weekly df for boom model (RBs use combined FP)
    rec_for_boom = receiving.copy()
    rb_ids = rb_combined[['Name', 'Team', 'Year', 'WEEK', 'FP', 'XFP',
                           'TGT', 'REC', 'RTE', 'aDOT', 'AY Share',
                           'TGT %', 'YPRR', 'YAC']].copy()
    rb_ids['POS'] = 'RB'

    wr_te_boom = rec_for_boom[rec_for_boom['POS'].isin(['WR', 'TE'])][[
        'Name', 'Team', 'POS', 'Year', 'WEEK', 'FP', 'XFP',
        'TGT', 'REC', 'RTE', 'aDOT', 'AY Share', 'TGT %', 'YPRR', 'YAC'
    ]]

    boom_weekly_input = pd.concat([wr_te_boom, rb_ids], ignore_index=True)

    trained_boom = train_boom_models(boom_weekly_input, season_stats)
    for pos in ['WR', 'TE', 'RB']:
        cv = trained_boom[pos]['cv_results']
        avg_auc = np.mean([r['AUC'] for r in cv])
        avg_ap  = np.mean([r['AvgPrecision'] for r in cv])
        print(f"  {pos}: avg AUC={avg_auc:.3f}  avg AvgPrec={avg_ap:.3f}")

    # ── 10. Boom projections for 2025 back-test ───────────────────────────────
    print_section("Boom week back-test (2025 season)")
    weekly_boom, boom_summary = project_boom_weeks(
        boom_weekly_input, season_stats, trained_boom, target_year=2025
    )
    weekly_boom.to_csv(OUTPUT_DIR / 'boom_weekly_2025.csv', index=False)
    boom_summary.sort_values('Exp_Boom_Weeks', ascending=False)\
                .to_csv(OUTPUT_DIR / 'boom_summary_2025.csv', index=False)

    print(f"  Per-week boom probs -> outputs/boom_weekly_2025.csv")
    print(f"  Boom summary        -> outputs/boom_summary_2025.csv")

    print("\n  Top 15 by expected boom weeks (2025 back-test):")
    top_boom = boom_summary.sort_values('Exp_Boom_Weeks', ascending=False).head(15)
    print(top_boom[['Name', 'Team', 'POS', 'Weeks_Played',
                    'Exp_Boom_Weeks', 'Actual_Boom_Weeks',
                    'Avg_Boom_Prob']].to_string(index=False))

    # ── 11. Merge boom probs into 2026 weekly schedule ────────────────────────
    print_section("Merging boom probabilities into 2026 weekly schedule")
    # Use player-level avg boom prob from 2025 back-test as a 2026 prior
    boom_prior = boom_summary[['Name', 'POS', 'Avg_Boom_Prob', 'Max_Boom_Prob']].copy()
    boom_prior = boom_prior.rename(columns={
        'Avg_Boom_Prob': 'base_boom_prob',
        'Max_Boom_Prob': 'ceiling_boom_prob'
    })

    weekly_26 = weekly_26.merge(boom_prior, on=['Name', 'POS'], how='left')

    # Adjust boom prob by matchup: scale base probability by matchup advantage
    weekly_26['adj_boom_prob'] = (
        weekly_26['base_boom_prob'] * (1 + weekly_26['matchup_adv'])
    ).clip(0, 0.95).round(3)

    weekly_26.to_csv(OUTPUT_DIR / 'weekly_schedule_2026.csv', index=False)
    print(f"  Updated {len(weekly_26):,} rows -> outputs/weekly_schedule_2026.csv")

    # Show best boom weeks for 2026
    print("\n  Top 2026 boom week opportunities (adj_boom_prob):")
    top_boom_26 = (weekly_26.dropna(subset=['adj_boom_prob'])
                   .sort_values('adj_boom_prob', ascending=False)
                   .head(15))
    print(top_boom_26[['Name', 'POS', 'Week', 'Opponent',
                        'base_boom_prob', 'matchup_adv', 'adj_boom_prob',
                        'week_proj_FP']].to_string(index=False))

    # ── 12. CV metrics + feature importance ──────────────────────────────────
    print_section("Model evaluation summary")
    save_cv_metrics(trained_season, trained_boom)

    fi = get_feature_importance(trained_season)
    fi.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)
    print("\n  Top 8 features per position:")
    for pos in ['WR', 'TE', 'RB']:
        top_fi = fi[fi['POS'] == pos].head(8)
        print(f"\n  {pos}:")
        print(top_fi[['Feature', 'Importance']].to_string(index=False))

    print("\n\nAll outputs written to outputs/")


if __name__ == '__main__':
    main()
