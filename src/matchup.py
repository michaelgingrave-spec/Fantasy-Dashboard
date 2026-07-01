"""
matchup.py
Build coverage-based matchup scores for WR/TE/RB vs opposing defenses.

Core logic:
  Each player has efficiency splits by coverage type (man/zone/1-hi/2-hi)
  from ManvsZone data. Each defense has a coverage tendency profile from
  the Coverage Matrix. We combine them to estimate how a player should
  perform against a specific defense, relative to their baseline.

  matchup_YPRR = player_man_YPRR  * def_MAN_pct/100
               + player_zone_YPRR * def_ZONE_pct/100

  matchup_adv  = matchup_YPRR / player_overall_YPRR - 1
  (positive = favorable, negative = tough)

Also handles:
  - OC continuity flags (same OC 2025→2026?)
  - 2026 weekly schedule with per-week matchup scores per player
"""

import pandas as pd
import numpy as np
from team_utils import full_to_abbrev, abbrev_to_full


# ── Build defensive profiles ──────────────────────────────────────────────────

def build_def_profiles(coverage_matrix: pd.DataFrame,
                       years: list = None) -> pd.DataFrame:
    """
    Compute a defensive coverage profile per team per year.
    Uses the coverage matrix; optionally filters to specific years.
    If years=['latest'], returns only the most recent year per team.
    """
    df = coverage_matrix.copy()
    if years and years != ['latest']:
        df = df[df['Year'].isin(years)]

    cols = ['team_abbrev', 'Year', 'MAN %', 'ZONE %',
            '1-HI/MOF C %', '2-HI/MOF O %',
            'man_FPDB', 'zone_FPDB', '1hi_FPDB', '2hi_FPDB',
            'COVER 0 %', 'COVER 1 %', 'COVER 2 %', 'COVER 2 MAN %',
            'COVER 3 %', 'COVER 4 %', 'COVER 6 %', 'DB']

    avail = [c for c in cols if c in df.columns]
    return df[avail].copy()


def build_smoothed_def_profiles(coverage_matrix: pd.DataFrame,
                                 alpha: float = 0.6) -> pd.DataFrame:
    """
    For projecting 2026: build a smoothed defensive profile per team
    using a weighted average of 2024 (40%) and 2025 (60%).
    Returns one row per team_abbrev.
    """
    numeric_cols = ['MAN %', 'ZONE %', '1-HI/MOF C %', '2-HI/MOF O %',
                    'man_FPDB', 'zone_FPDB', '1hi_FPDB', '2hi_FPDB',
                    'COVER 0 %', 'COVER 1 %', 'COVER 2 %', 'COVER 2 MAN %',
                    'COVER 3 %', 'COVER 4 %', 'COVER 6 %']
    avail = [c for c in numeric_cols if c in coverage_matrix.columns]

    y24 = coverage_matrix[coverage_matrix['Year'] == 2024].set_index('team_abbrev')[avail]
    y25 = coverage_matrix[coverage_matrix['Year'] == 2025].set_index('team_abbrev')[avail]

    # Align
    all_teams = y25.index.union(y24.index)
    y25 = y25.reindex(all_teams)
    y24 = y24.reindex(all_teams)

    # Weighted blend: use 2025 if available, fall back to 2024
    smoothed = y25.copy()
    for t in all_teams:
        has_25 = not y25.loc[t].isna().all()
        has_24 = not y24.loc[t].isna().all() if t in y24.index else False
        if has_25 and has_24:
            smoothed.loc[t] = alpha * y25.loc[t] + (1 - alpha) * y24.loc[t]
        elif has_24 and not has_25:
            smoothed.loc[t] = y24.loc[t]

    smoothed = smoothed.reset_index().rename(columns={'index': 'team_abbrev'})
    return smoothed


# ── Player matchup scoring ────────────────────────────────────────────────────

def compute_player_matchup_score(player_mvz: pd.Series,
                                  def_profile: pd.Series) -> dict:
    """
    Given one player's ManvsZone splits and one defense's coverage profile,
    compute a matchup advantage score.

    Returns dict with:
      matchup_yprr     : expected YPRR vs this specific defense
      matchup_adv      : % above/below player's baseline YPRR
      man_adv          : man coverage advantage component
      zone_adv         : zone coverage advantage component
      def_man_pct      : how often defense plays man
      def_zone_pct     : how often defense plays zone
      def_fpdb         : weighted FP allowed per dropback
    """
    man_pct  = float(def_profile.get('MAN %',       0) or 0) / 100
    zone_pct = float(def_profile.get('ZONE %',      0) or 0) / 100
    hi1_pct  = float(def_profile.get('1-HI/MOF C %', 0) or 0) / 100
    hi2_pct  = float(def_profile.get('2-HI/MOF O %', 0) or 0) / 100

    p_man_yprr  = float(player_mvz.get('man_YPRR',  0) or 0)
    p_zone_yprr = float(player_mvz.get('zone_YPRR', 0) or 0)
    p_1hi_yprr  = float(player_mvz.get('1hi_YPRR',  0) or 0)
    p_2hi_yprr  = float(player_mvz.get('2hi_YPRR',  0) or 0)
    p_ov_yprr   = float(player_mvz.get('ov_YPRR',   0) or 0)

    # Weighted matchup YPRR using detailed coverage splits
    matchup_yprr = (p_man_yprr * man_pct +
                    p_1hi_yprr * hi1_pct * zone_pct +
                    p_2hi_yprr * hi2_pct * zone_pct)

    # Fallback to simple man/zone split if sub-splits are zero
    if matchup_yprr == 0 and (p_man_yprr > 0 or p_zone_yprr > 0):
        matchup_yprr = p_man_yprr * man_pct + p_zone_yprr * zone_pct

    matchup_adv = (matchup_yprr / p_ov_yprr - 1) if p_ov_yprr > 0 else 0.0

    # Weighted FP/DB the defense allows
    man_fpdb  = float(def_profile.get('man_FPDB',  0) or 0)
    zone_fpdb = float(def_profile.get('zone_FPDB', 0) or 0)
    def_fpdb  = man_pct * man_fpdb + zone_pct * zone_fpdb

    return {
        'matchup_yprr':  round(matchup_yprr, 3),
        'matchup_adv':   round(matchup_adv,  3),
        'man_adv':       round(p_man_yprr  - p_ov_yprr, 3),
        'zone_adv':      round(p_zone_yprr - p_ov_yprr, 3),
        'def_man_pct':   round(man_pct  * 100, 1),
        'def_zone_pct':  round(zone_pct * 100, 1),
        'def_fpdb':      round(def_fpdb, 3),
    }


def build_player_matchup_table(mvz: pd.DataFrame,
                                def_profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-join each player's 2025 ManvsZone stats with every team's
    2026 projected defensive profile. Returns one row per player-opponent.
    """
    mvz_25 = mvz[mvz['Year'] == 2025].copy()
    def_idx = def_profiles.set_index('team_abbrev')

    rows = []
    for _, player in mvz_25.iterrows():
        for team_abbrev, def_row in def_idx.iterrows():
            score = compute_player_matchup_score(player, def_row)
            rows.append({
                'Name':       player['Name'],
                'Team':       player['Team'],
                'POS':        player.get('POS', ''),
                'Opponent':   team_abbrev,
                **score,
            })

    return pd.DataFrame(rows)


# ── 2026 weekly schedule ──────────────────────────────────────────────────────

def build_2026_weekly_schedule(schedule: pd.DataFrame,
                                players_2025: pd.DataFrame,
                                def_profiles: pd.DataFrame,
                                mvz: pd.DataFrame) -> pd.DataFrame:
    """
    For each player who appeared in 2025, generate one row per
    scheduled 2026 week showing:
      - their opponent that week
      - the opponent's defensive coverage profile
      - their matchup advantage score

    players_2025 : season stats aggregated for 2025
    def_profiles : smoothed defensive profiles (output of build_smoothed_def_profiles)
    mvz          : full ManvsZone table (we use 2025 season)
    """
    mvz_25 = mvz[mvz['Year'] == 2025].set_index(['Name', 'POS'])
    def_idx = def_profiles.set_index('team_abbrev')

    # Build team -> weekly opponent map from schedule
    # Each row has home/away — create both directions
    sched_rows = []
    for _, g in schedule.iterrows():
        wk = g['week']
        home = g['home_abbrev']
        away = g['away_abbrev']
        home_dc = g.get('home_dc', '')
        away_dc = g.get('away_dc', '')
        sched_rows.append({'team': home, 'week': wk,
                           'opponent': away, 'opp_dc': away_dc})
        sched_rows.append({'team': away, 'week': wk,
                           'opponent': home, 'opp_dc': home_dc})

    team_schedule = pd.DataFrame(sched_rows)

    # Build player weekly rows
    result_rows = []
    for _, player in players_2025.iterrows():
        name = player['Name']
        pos  = player['POS']
        team = player['Team']

        player_weeks = team_schedule[team_schedule['team'] == team]
        if player_weeks.empty:
            continue

        # Get player's coverage splits
        key = (name, pos)
        if key in mvz_25.index:
            p_mvz = mvz_25.loc[key]
            if isinstance(p_mvz, pd.DataFrame):
                p_mvz = p_mvz.iloc[0]  # take first if duplicates
        else:
            p_mvz = pd.Series(dtype=float)

        for _, wk_row in player_weeks.iterrows():
            opp = wk_row['opponent']
            if opp in def_idx.index:
                def_row = def_idx.loc[opp]
            else:
                def_row = pd.Series(dtype=float)

            score = compute_player_matchup_score(p_mvz, def_row)

            result_rows.append({
                'Name':          name,
                'Team':          team,
                'POS':           pos,
                'Week':          int(wk_row['week']),
                'Opponent':      opp,
                'Opp_DC':        wk_row['opp_dc'],
                'Proj_FP_2026':  player.get('Proj_FP_2026', np.nan),
                **score,
            })

    weekly = pd.DataFrame(result_rows).sort_values(
        ['Name', 'Week']).reset_index(drop=True)

    # Add per-week projected FP = (season proj / 17) * matchup scaling
    weekly['week_base_FP'] = (weekly['Proj_FP_2026'] / 17).round(2)
    weekly['week_proj_FP'] = (weekly['week_base_FP'] *
                               (1 + weekly['matchup_adv'])).round(2)

    return weekly


# ── OC continuity ─────────────────────────────────────────────────────────────

def build_oc_continuity(coordinators: pd.DataFrame) -> pd.DataFrame:
    """
    Flag teams where the offensive coordinator / play caller changed
    between 2025 and 2026. Returns one row per team with:
      oc_same_2526   : bool - same OC both years
      pc_same_2526   : bool - same play caller both years
    """
    y25 = coordinators[coordinators['season'] == 2025][
        ['team_abbrev', 'offensive_coordinator', 'oc_play_caller']
    ].rename(columns={
        'offensive_coordinator': 'oc_2025',
        'oc_play_caller': 'pc_2025'
    })

    y26 = coordinators[coordinators['season'] == 2026][
        ['team_abbrev', 'offensive_coordinator', 'oc_play_caller']
    ].rename(columns={
        'offensive_coordinator': 'oc_2026',
        'oc_play_caller': 'pc_2026'
    })

    df = y25.merge(y26, on='team_abbrev', how='outer')
    df['oc_same_2526'] = (df['oc_2025'] == df['oc_2026']).astype(int)
    df['pc_same_2526'] = (df['pc_2025'] == df['pc_2026']).astype(int)

    return df
