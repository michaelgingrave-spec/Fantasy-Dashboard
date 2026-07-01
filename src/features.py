"""
features.py
Build year-over-year training pairs for the season projection model,
and rolling weekly features for the boom week model.
"""

import pandas as pd
import numpy as np

# ── Season-level feature columns used to predict next year's FP ──────────────
SEASON_FEATURES = [
    # Games played — important context for the model
    'Games',
    # Games-normalised season totals (injury-proof volume signals).
    # These replace raw count totals so a 9-game season with 5 TGT/game
    # looks identical to a 17-game equivalent — not like a low-volume player.
    'FP_per17', 'XFP_per17',
    'TGT_per17', 'REC_per17', 'RTE_per17', 'YDS_per17',
    'YAC_per17', 'TD_per17', 'i20TGT_per17', 'EZTGT_per17',
    # Per-game rates (also game-count independent)
    'TGT_per_game', 'RTE_per_game', 'FPG_mean', 'XFPG_mean',
    # Efficiency
    'aDOT_mean', 'AYshare_mean', 'TGT_pct_mean', 'CR_pct_mean',
    'YPRR_mean', 'REC_rate', 'TD_per_REC', 'YDS_per_RTE',
    # Opportunity quality (rate-based; not penalised by game count)
    'XFP_pct',
    # Consistency / ceiling
    'FP_std', 'FP_max', 'FP_median',
    'FP_boom_rate_10',   # fraction of weeks with 10+ FP (replaces raw count)
    'FP_boom_rate_20',   # fraction of weeks with 20+ FP (replaces raw count)
    # Career stage (proxy for age — 5 = multi-year veteran, 1 = newcomer)
    'years_in_data',
    # Year-over-year trend features (added in build_yoy_pairs)
    'FPG_trend',       # FP/game this year minus FP/game previous year
    'XFPG_trend',      # XFP/game trend
    'TGT_trend',       # targets/game trend
    # Rushing context (RBs; ~0 for WR/TE) — per17 for injury robustness
    'rush_share', 'ATT_per_game', 'rush_YPC', 'ATT_per17', 'rush_YDS_per17',
    # Coverage split efficiency (ManvsZone)
    'man_YPRR', 'zone_YPRR', '1hi_YPRR', '2hi_YPRR',
    'man_FPRR', 'zone_FPRR',
    # OC continuity (1 = same OC, 0 = new OC)
    'oc_same_2526',
]

BOOM_THRESHOLDS = {
    'WR': 18.1,   # 90th percentile across 2021-2025 (receiving FP = total)
    'TE': 12.3,
    'RB': 20.2,   # receiving file FP already = total PPR FP for RBs
}


def _attach_mvz(pairs: pd.DataFrame, mvz: pd.DataFrame) -> pd.DataFrame:
    """Attach ManvsZone coverage split stats (year N) to YoY pairs."""
    mvz_sel = mvz[['Name', 'POS', 'Year',
                   'man_YPRR', 'zone_YPRR', '1hi_YPRR', '2hi_YPRR',
                   'man_FPRR', 'zone_FPRR', 'ov_YPRR']].copy()
    return pairs.merge(mvz_sel, on=['Name', 'POS', 'Year'], how='left')


def _attach_oc_continuity(pairs: pd.DataFrame,
                           oc_df: pd.DataFrame) -> pd.DataFrame:
    """Set OC continuity flag. Historical training pairs assume continuity (=1)."""
    pairs = pairs.copy()
    if 'oc_same_2526' not in pairs.columns:
        pairs['oc_same_2526'] = 1
    return pairs


def _add_trend_features(pairs: pd.DataFrame,
                         season_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Add year-over-year trend features to the training pairs:
      FPG_trend  = FPG(year N) - FPG(year N-1)
      XFPG_trend = XFPG(year N) - XFPG(year N-1)
      TGT_trend  = TGT/game(year N) - TGT/game(year N-1)

    These help the model distinguish a player on an upward trajectory
    from a veteran declining — without needing explicit age data.
    """
    # Prior year lookup
    prior = season_stats[['Name', 'POS', 'Year', 'FPG_mean', 'XFPG_mean',
                           'TGT_per_game']].copy()
    prior['Year_next'] = prior['Year'] + 1
    prior = prior.rename(columns={
        'FPG_mean':    'prior_FPG',
        'XFPG_mean':   'prior_XFPG',
        'TGT_per_game': 'prior_TGT',
    })

    pairs = pairs.merge(
        prior[['Name', 'POS', 'Year_next', 'prior_FPG', 'prior_XFPG', 'prior_TGT']],
        left_on=['Name', 'POS', 'Year'],
        right_on=['Name', 'POS', 'Year_next'],
        how='left'
    ).drop(columns=['Year_next'])

    pairs['FPG_trend']  = pairs['FPG_mean']    - pairs['prior_FPG'].fillna(pairs['FPG_mean'])
    pairs['XFPG_trend'] = pairs['XFPG_mean']   - pairs['prior_XFPG'].fillna(pairs['XFPG_mean'])
    pairs['TGT_trend']  = pairs['TGT_per_game']- pairs['prior_TGT'].fillna(pairs['TGT_per_game'])

    pairs = pairs.drop(columns=['prior_FPG', 'prior_XFPG', 'prior_TGT'], errors='ignore')
    return pairs


def build_yoy_pairs(season_stats: pd.DataFrame,
                    mvz: pd.DataFrame = None,
                    oc_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    For each player who appeared in consecutive seasons, create a training row:
      features = season N stats  →  target = season N+1 FP_total
    """
    rows = []
    years = sorted(season_stats['Year'].unique())

    for yr in years[:-1]:
        yr1 = season_stats[season_stats['Year'] == yr].copy()
        yr2 = season_stats[season_stats['Year'] == yr + 1][
            ['Name', 'POS', 'FP_total', 'Games']
        ].rename(columns={'FP_total': 'FP_next', 'Games': 'Games_next'})

        merged = yr1.merge(yr2, on=['Name', 'POS'], how='inner')
        merged['Year'] = yr
        rows.append(merged)

    pairs = pd.concat(rows, ignore_index=True)
    pairs = pairs[(pairs['Games'] >= 8) & (pairs['Games_next'] >= 8)]

    # ── Injury-adjust the target ──────────────────────────────────────────────
    # When year N+1 had fewer than 14 games, scale FP_next to a 17-game pace.
    # This prevents injury-shortened seasons from being read as talent regressions.
    # Example: Bowers 2025 (12g, 176 FP) → target = 249.6 instead of 176.
    short_season = pairs['Games_next'] < 14
    pairs.loc[short_season, 'FP_next'] = (
        pairs.loc[short_season, 'FP_next']
        * (17.0 / pairs.loc[short_season, 'Games_next'])
    ).round(1)

    # Trend features (requires prior-year lookup into season_stats)
    pairs = _add_trend_features(pairs, season_stats)

    # ManvsZone coverage splits
    if mvz is not None:
        pairs = _attach_mvz(pairs, mvz)

    # OC continuity
    pairs = _attach_oc_continuity(pairs, oc_df)

    return pairs


def build_weekly_features(weekly_df: pd.DataFrame,
                           season_stats: pd.DataFrame) -> pd.DataFrame:
    """
    For the boom week model: attach rolling weekly features and boom label.
    """
    df = weekly_df.copy().sort_values(['Name', 'POS', 'Year', 'WEEK'])

    numeric = ['FP', 'XFP', 'TGT', 'REC', 'YDS', 'TD', 'RTE',
               'aDOT', 'AY Share', 'TGT %', 'YPRR', 'YAC']
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    grp = df.groupby(['Name', 'POS', 'Year'])
    for col in ['FP', 'XFP', 'TGT', 'REC', 'RTE', 'aDOT', 'AY Share', 'TGT %']:
        if col in df.columns:
            df[f'{col}_roll3']      = grp[col].transform(
                lambda x: x.shift(1).rolling(3, min_periods=1).mean())
            df[f'{col}_season_avg'] = grp[col].transform(
                lambda x: x.shift(1).expanding().mean())

    df['FP_roll3_std']  = grp['FP'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).std())
    df['FP_season_std'] = grp['FP'].transform(
        lambda x: x.shift(1).expanding().std())

    # Boom label per position
    df['boom'] = 0
    for pos, thresh in BOOM_THRESHOLDS.items():
        mask = df['POS'] == pos
        df.loc[mask, 'boom'] = (df.loc[mask, 'FP'] >= thresh).astype(int)

    return df[df['WEEK'] > 1].copy()


WEEKLY_FEATURES = [
    'WEEK',
    'FP_roll3', 'XFP_roll3', 'TGT_roll3', 'REC_roll3', 'RTE_roll3',
    'aDOT_roll3', 'AY Share_roll3', 'TGT %_roll3',
    'FP_season_avg', 'XFP_season_avg', 'TGT_season_avg', 'RTE_season_avg',
    'aDOT_season_avg', 'AY Share_season_avg',
    'FP_roll3_std', 'FP_season_std',
]
