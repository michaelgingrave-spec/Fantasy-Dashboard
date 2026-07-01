"""
data_loader.py
Load and combine all data sources into clean DataFrames.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from team_utils import abbrev_to_full, full_to_abbrev

DATA_DIR = Path(__file__).resolve().parent.parent
YEARS = [2021, 2022, 2023, 2024, 2025]

# ── Common numeric columns per data type ─────────────────────────────────────

REC_NUMERIC = [
    'TGT', 'REC', 'YDS', 'TD', 'FP', 'XFP',
    'RTE', 'aDOT', 'AY', 'AY Share', 'TGT %', 'CR %',
    'YPRR', 'YPR', 'YAC', 'YAC/REC',
    'i20 TGT', 'EZTGT', 'EZTD', 'DP TGT', '1READ', 'MTF',
    'FP/G', 'XFP/G', 'FP/RR', 'XFP/RR', 'RecXFP',
    'WEEK', 'G', 'Rank',
    'WIDE RTE %', 'SLOT RTE %', 'INLINE RTE %', 'BACK RTE %',
    'TM YDS %', 'TM TD %', 'RATE', 'THREAT', 'YPTOE',
    '1D', 'DRP', 'CTGT', 'DESIGN', 'CT', 'CC',
]

RUSH_NUMERIC = [
    'ATT', 'YDS', 'YPC', 'TD', 'FUM', '1D',
    'EXP YDS', 'EXP YDS %', 'i5 %', 'TD RATE',
    'Success %', 'STUFF %', 'MTF', 'MTF/ATT',
    'YACO', 'YACO/ATT', 'YACO %', 'YBCO/ATT',
    'FP/G', 'FP', 'XFP', 'XFP/G',
    'WEEK', 'G', 'Rank',
    # Zone/Gap concept cols (.1 = zone, .2 = gap)
    'ATT.1', 'ATT %', 'YDS.1', 'TD.1', 'YPC.1', 'Success %.1',
    'ATT.2', 'ATT %.1', 'YDS.2', 'TD.2', 'YPC.2', 'Success %.2',
]

PASS_NUMERIC = [
    'DB', 'ATT', 'CMP', 'CMP %', 'YDS', 'YDS/G', 'YPA',
    'TD', 'INT', '1D', 'RATE', 'SACK', 'ANY/A',
    'SCRM', 'CPOE', 'aDOT', 'AY', 'Deep Throw %',
    'YAC %', 'ADJ CMP %', '1Read %', 'ACC %',
    'EZATT', 'DROP %', 'TTT', 'TTP', 'TTSK',
    'PRESS %', 'PRESS SK %', 'CHK %', 'RPO %',
    'FP/DB', 'FP/OPP', 'FP/G', 'FP', 'OPP',
    'WEEK', 'G', 'Rank',
]


def _load_yearly(folder: str, prefix: str, header_row: int = 1,
                 numeric_cols: list = None) -> pd.DataFrame:
    """Generic yearly CSV loader."""
    frames = []
    for yr in YEARS:
        path = DATA_DIR / folder / f"{yr}{prefix}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, header=header_row, encoding='utf-8-sig')
        df['Year'] = yr
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    if numeric_cols:
        for col in numeric_cols:
            if col in combined.columns:
                combined[col] = pd.to_numeric(combined[col], errors='coerce')

    combined['Name'] = combined['Name'].astype(str).str.strip()
    combined['Team'] = combined['Team'].astype(str).str.strip()
    if 'POS' in combined.columns:
        combined['POS'] = combined['POS'].astype(str).str.strip()

    return combined.reset_index(drop=True)


# ── Public loaders ────────────────────────────────────────────────────────────

def load_receiving() -> pd.DataFrame:
    df = _load_yearly('Receiving Stats', 'Receiving', header_row=1,
                      numeric_cols=REC_NUMERIC)
    df = df.dropna(subset=['Name', 'POS', 'WEEK'])
    df = df[df['POS'].isin(['WR', 'TE', 'RB'])]
    return df


def load_rushing() -> pd.DataFrame:
    df = _load_yearly('Rushing Stats', 'Rushing', header_row=1,
                      numeric_cols=RUSH_NUMERIC)
    df = df.dropna(subset=['Name', 'WEEK'])
    return df


def load_passing() -> pd.DataFrame:
    df = _load_yearly('Passing Stats', 'Passing', header_row=1,
                      numeric_cols=PASS_NUMERIC)
    df = df.dropna(subset=['Name', 'WEEK'])
    return df


def load_man_vs_zone() -> pd.DataFrame:
    """
    Season-level player splits by coverage type.
    Columns renamed: Overall / Man / Zone / Single-High / Two-High
    """
    frames = []
    for yr in YEARS:
        path = DATA_DIR / 'Receiving Stats' / f"{yr}ManvsZone.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, header=1, encoding='utf-8-sig')
        df['Year'] = yr
        frames.append(df)

    mvz = pd.concat(frames, ignore_index=True)

    # Rename the duplicate .1/.2/.3/.4 columns to meaningful names
    col_map = {
        'RTE':    'ov_RTE',   'TPRR':    'ov_TPRR',   'YPRR':    'ov_YPRR',   'FP/RR':    'ov_FPRR',
        'RTE.1':  'man_RTE',  'TPRR.1':  'man_TPRR',  'YPRR.1':  'man_YPRR',  'FP/RR.1':  'man_FPRR',
        'RTE.2':  'zone_RTE', 'TPRR.2':  'zone_TPRR', 'YPRR.2':  'zone_YPRR', 'FP/RR.2':  'zone_FPRR',
        'RTE.3':  '1hi_RTE',  'TPRR.3':  '1hi_TPRR',  'YPRR.3':  '1hi_YPRR',  'FP/RR.3':  '1hi_FPRR',
        'RTE.4':  '2hi_RTE',  'TPRR.4':  '2hi_TPRR',  'YPRR.4':  '2hi_YPRR',  'FP/RR.4':  '2hi_FPRR',
    }
    mvz = mvz.rename(columns=col_map)

    str_cols = {'Name', 'Team', 'POS', 'Year', 'Rank', 'Season'}
    for c in mvz.columns:
        if c not in str_cols:
            mvz[c] = pd.to_numeric(mvz[c], errors='coerce')

    mvz['Name'] = mvz['Name'].astype(str).str.strip()
    mvz['Team'] = mvz['Team'].astype(str).str.strip()
    return mvz.reset_index(drop=True)


def load_separation() -> pd.DataFrame:
    """
    Season-level separation by alignment (Wide / Slot / Inline / Backfield).
    Only available 2022-2025.
    """
    frames = []
    for yr in YEARS:
        path = DATA_DIR / 'Receiving Stats' / f"{yr}receivingSeparationByAlignmentExport.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, header=1, encoding='utf-8-sig')
        df['Year'] = yr
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    sep = pd.concat(frames, ignore_index=True)

    # Drop category header rows
    sep = sep[sep['Name'] != 'Name'].copy()

    # Rename split columns
    col_map = {}
    prefixes = ['ov', 'wide', 'slot', 'inline', 'back']
    base_cols = ['RTE', 'SEP SCORE', 'YPRR', 'TPRR', 'WIN RATE',
                 '+1 Rate', '+2 Rate', '+3 Rate', 'Neg Rate',
                 'TGT', 'REC', 'YDS', 'TD', 'AY']

    # The file has Overall then Wide then Slot then Inline then Backfield
    # Each section has the same set of columns but with .N suffixes after the first
    # We'll use the raw column names and just coerce numerics
    for c in sep.columns:
        if c not in ['Name', 'Team', 'POS', 'Year', 'Rank', 'G', 'Season']:
            sep[c] = pd.to_numeric(sep[c], errors='coerce')

    sep['Name'] = sep['Name'].astype(str).str.strip()
    sep['Team'] = sep['Team'].astype(str).str.strip()
    return sep.reset_index(drop=True)


def load_coverage_matrix() -> pd.DataFrame:
    """
    Team-level season defensive coverage tendencies.
    One row per team per year.
    """
    frames = []
    for yr in YEARS:
        path = DATA_DIR / 'Team Stats' / f"{yr}CoverageMatrix.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path, header=0, encoding='utf-8-sig')
        # Row 0 is the real column header
        real_cols = raw.iloc[0].tolist()
        df = raw.iloc[1:].copy()
        df.columns = real_cols
        df['Year'] = yr
        frames.append(df)

    cov = pd.concat(frames, ignore_index=True)

    # Rename duplicate FP/DB columns
    # Columns: Rank, Name, G, Season, Location, Team Name, DB,
    #          MAN %, FP/DB, ZONE %, FP/DB, 1-HI/MOF C %, FP/DB,
    #          2-HI/MOF O %, FP/DB, COVER 0-6 %
    cols = list(cov.columns)
    seen = {}
    new_cols = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    cov.columns = new_cols

    # Rename to clear names
    rename = {
        'FP/DB':   'man_FPDB',
        'FP/DB_1': 'zone_FPDB',
        'FP/DB_2': '1hi_FPDB',
        'FP/DB_3': '2hi_FPDB',
    }
    cov = cov.rename(columns=rename)

    numeric_cols = ['DB', 'MAN %', 'man_FPDB', 'ZONE %', 'zone_FPDB',
                    '1-HI/MOF C %', '1hi_FPDB', '2-HI/MOF O %', '2hi_FPDB',
                    'COVER 0 %', 'COVER 1 %', 'COVER 2 %', 'COVER 2 MAN %',
                    'COVER 3 %', 'COVER 4 %', 'COVER 6 %', 'G']
    for c in numeric_cols:
        if c in cov.columns:
            cov[c] = pd.to_numeric(cov[c], errors='coerce')

    # Add team abbreviation for joining
    cov['team_abbrev'] = cov['Name'].apply(full_to_abbrev)
    cov['Name'] = cov['Name'].astype(str).str.strip()

    return cov.reset_index(drop=True)


def load_coordinators() -> pd.DataFrame:
    """OC/DC history by team 2021-2026."""
    path = DATA_DIR / 'Team Stats' / 'nfl_coordinators_2021_2026.csv'
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['team_abbrev'] = df['team'].apply(full_to_abbrev)
    return df


def load_2026_schedule() -> pd.DataFrame:
    """2026 week-by-week schedule with coordinators attached."""
    path = DATA_DIR / 'Team Stats' / 'nfl_2026_schedule_with_coordinators.csv'
    df = pd.read_csv(path, encoding='utf-8-sig')
    # Add abbreviations for both teams
    df['home_abbrev'] = df['home_team'].apply(full_to_abbrev)
    df['away_abbrev'] = df['away_team'].apply(full_to_abbrev)
    return df


# ── Aggregators ───────────────────────────────────────────────────────────────

def get_combined_rb_stats(receiving: pd.DataFrame,
                          rushing: pd.DataFrame) -> pd.DataFrame:
    """
    For RBs: the FP column in BOTH the rushing and receiving files already
    represents the player's TOTAL weekly fantasy points (the platform reports
    total FP from both analytical views). We use the receiving file as the
    base (it contains receiving-specific features) and LEFT-JOIN rushing stats
    purely for additional rushing features (ATT, YPC, Success%, etc.).
    FP is NOT summed — we keep receiving file's FP which is already total PPR FP.
    """
    rb_rec = receiving[receiving['POS'] == 'RB'][
        ['Name', 'Team', 'POS', 'Year', 'WEEK', 'FP', 'XFP',
         'TGT', 'REC', 'YDS', 'TD', 'RTE', 'aDOT', 'AY Share', 'TGT %',
         'CR %', 'YPRR', 'YAC', 'i20 TGT', 'EZTGT', 'FP/G', 'XFP/G',
         'RecXFP']
    ].copy()
    # YDS/TD kept as receiving yards/TDs (consistent naming with WR/TE)

    rb_rush = rushing[rushing['POS'] == 'RB'][
        ['Name', 'Team', 'Year', 'WEEK',
         'ATT', 'YDS', 'YPC', 'TD',
         'Success %', 'STUFF %', 'MTF', 'YACO', 'i5 %']
    ].copy()
    rb_rush = rb_rush.rename(columns={
        'YDS': 'rush_YDS', 'TD': 'rush_TD',
    })

    # Left join: keep all receiving rows; attach rushing stats as features
    rb = rb_rec.merge(rb_rush, on=['Name', 'Team', 'Year', 'WEEK'], how='left')

    # FP and XFP from receiving file = total PPR FP (rushing + receiving)
    # No summation needed
    rb['rush_YDS']   = rb['rush_YDS'].fillna(0)
    rb['rush_TD']    = rb['rush_TD'].fillna(0)
    rb['ATT']        = rb['ATT'].fillna(0)
    rb['rush_share'] = rb['rush_YDS'] / (rb['rush_YDS'] + rb['YDS'] + 1e-6)
    rb['POS']        = 'RB'

    return rb


def get_season_stats(receiving: pd.DataFrame,
                     rb_combined: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate weekly data into per-player per-season totals.
    RBs use combined rushing+receiving FP; WR/TE use receiving FP.
    """
    # Build unified weekly df
    wr_te = receiving[receiving['POS'].isin(['WR', 'TE'])].copy()
    wr_te['rec_FP']  = wr_te['FP']
    wr_te['rush_FP'] = 0.0
    wr_te['rec_XFP'] = wr_te['XFP']
    wr_te['rush_XFP'] = 0.0
    wr_te['rush_YDS'] = 0.0
    wr_te['rush_TD']  = 0.0
    wr_te['ATT']      = 0.0

    # Shared columns that exist in both wr_te and rb_combined
    shared = ['Name', 'Team', 'POS', 'Year', 'WEEK',
              'FP', 'XFP', 'TGT', 'REC', 'YDS', 'TD', 'RTE', 'aDOT', 'AY Share',
              'TGT %', 'CR %', 'YPRR', 'YAC', 'i20 TGT', 'EZTGT',
              'FP/G', 'XFP/G', 'RecXFP']

    wr_te_cols  = [c for c in shared if c in wr_te.columns]
    rb_cols     = [c for c in shared + ['ATT', 'rush_YDS', 'rush_TD', 'rush_share',
                                         'Success %', 'YPC']
                   if c in rb_combined.columns]

    combined = pd.concat([
        wr_te[wr_te_cols],
        rb_combined[rb_cols],
    ], ignore_index=True)

    # Fill rushing columns with 0 for WR/TE rows
    for col in ['ATT', 'rush_YDS', 'rush_TD', 'rush_share', 'Success %', 'YPC']:
        if col not in combined.columns:
            combined[col] = 0.0
        combined[col] = combined[col].fillna(0.0)

    # ── Primary team per player-year (for OC continuity lookups) ────────────────
    # When a player splits a season between teams, we want a single row with
    # combined stats. Track which team they played the most games for.
    weeks_by_team = (combined.groupby(['Name', 'POS', 'Year', 'Team'])
                              .size()
                              .reset_index(name='_n'))
    primary_team = (weeks_by_team
                    .sort_values('_n', ascending=False)
                    .drop_duplicates(subset=['Name', 'POS', 'Year'])
                    [['Name', 'POS', 'Year', 'Team']])

    # Group by player-season only (not Team) so split-team seasons merge cleanly
    grp = combined.groupby(['Name', 'POS', 'Year'])

    agg = grp.agg(
        Games           = ('WEEK',       'count'),
        FP_total        = ('FP',         'sum'),
        FP_mean         = ('FP',         'mean'),
        FP_std          = ('FP',         'std'),
        FP_max          = ('FP',         'max'),
        FP_median       = ('FP',         'median'),
        XFP_total       = ('XFP',        'sum'),
        XFP_mean        = ('XFP',        'mean'),
        TGT_total       = ('TGT',        'sum'),
        TGT_mean        = ('TGT',        'mean'),
        REC_total       = ('REC',        'sum'),
        YDS_total       = ('YDS',        'sum'),
        TD_total        = ('TD',         'sum'),
        RTE_total       = ('RTE',        'sum'),
        YAC_total       = ('YAC',        'sum'),
        i20TGT_total    = ('i20 TGT',    'sum'),
        EZTGT_total     = ('EZTGT',      'sum'),
        aDOT_mean       = ('aDOT',       'mean'),
        AYshare_mean    = ('AY Share',   'mean'),
        TGT_pct_mean    = ('TGT %',      'mean'),
        CR_pct_mean     = ('CR %',       'mean'),
        YPRR_mean       = ('YPRR',       'mean'),
        FPG_mean        = ('FP/G',       'mean'),
        XFPG_mean       = ('XFP/G',      'mean'),
        # Rushing features (non-zero for RBs only)
        ATT_total       = ('ATT',       'sum'),
        rush_YDS_total  = ('rush_YDS',  'sum'),
        rush_TD_total   = ('rush_TD',   'sum'),
        rush_share_mean = ('rush_share','mean'),
        rush_success    = ('Success %', 'mean'),
        FP_weeks_10plus = ('FP', lambda x: (x >= 10).sum()),
        FP_weeks_20plus = ('FP', lambda x: (x >= 20).sum()),
        FP_weeks_25plus = ('FP', lambda x: (x >= 25).sum()),
    ).reset_index()

    # Attach primary team back (used for OC continuity and display)
    agg = agg.merge(primary_team, on=['Name', 'POS', 'Year'], how='left')

    # Derived rate stats
    agg['REC_rate']      = agg['REC_total']      / agg['TGT_total'].replace(0, np.nan)
    agg['TD_per_REC']    = agg['TD_total']        / agg['REC_total'].replace(0, np.nan)
    agg['YDS_per_RTE']   = agg['YDS_total']       / agg['RTE_total'].replace(0, np.nan)
    agg['TGT_per_game']  = agg['TGT_total']       / agg['Games']
    agg['RTE_per_game']  = agg['RTE_total']        / agg['Games']
    agg['ATT_per_game']  = agg['ATT_total']        / agg['Games']
    agg['rush_YPC']      = agg['rush_YDS_total']  / agg['ATT_total'].replace(0, np.nan)
    agg['XFP_pct']       = agg['XFP_total']       / agg['FP_total'].replace(0, np.nan)
    agg['rush_share']    = agg['rush_share_mean'].fillna(0)
    agg['rush_success']  = agg['rush_success'].fillna(0)

    # ── Games-normalised season totals ────────────────────────────────────────
    # Project each season to a full 17-game season so injury-shortened years
    # don't artificially suppress a player's talent level (e.g. Bowers 12g in 2025).
    # These per17 versions replace raw count totals as model features so that
    # a 9-game season with 5 TGT/game looks the same as a full-season equivalent.
    scale = 17.0 / agg['Games'].clip(lower=1)
    agg['FP_per17']      = (agg['FP_total']      * scale).round(1)
    agg['XFP_per17']     = (agg['XFP_total']     * scale).round(1)
    agg['TGT_per17']     = (agg['TGT_total']     * scale).round(1)
    agg['REC_per17']     = (agg['REC_total']     * scale).round(1)
    agg['RTE_per17']     = (agg['RTE_total']     * scale).round(1)
    agg['YDS_per17']     = (agg['YDS_total']     * scale).round(1)
    agg['YAC_per17']     = (agg['YAC_total']     * scale).round(1)
    agg['TD_per17']      = (agg['TD_total']      * scale).round(2)
    agg['i20TGT_per17']  = (agg['i20TGT_total']  * scale).round(1)
    agg['EZTGT_per17']   = (agg['EZTGT_total']   * scale).round(1)
    agg['ATT_per17']     = (agg['ATT_total']     * scale).round(1)
    agg['rush_YDS_per17']= (agg['rush_YDS_total']* scale).round(1)

    # Boom rate (fraction of weeks with a big game — game-count independent)
    agg['FP_boom_rate_10'] = (agg['FP_weeks_10plus'] / agg['Games']).fillna(0).round(3)
    agg['FP_boom_rate_20'] = (agg['FP_weeks_20plus'] / agg['Games']).fillna(0).round(3)

    # ── Career-stage features ─────────────────────────────────────────────────
    # years_in_data: how many of the 5 seasons (2021-2025) the player appeared in.
    # A proxy for career length / age — 5 = veteran, 1 = rookie/newcomer.
    seasons_count = (
        agg.groupby(['Name', 'POS'])['Year']
        .transform('count')
    )
    agg['years_in_data'] = seasons_count

    return agg


def get_qb_season_stats(passing: pd.DataFrame) -> pd.DataFrame:
    """QB passing volume aggregated by season — used as team context for receivers."""
    grp = passing.groupby(['Name', 'Team', 'Year'])
    agg = grp.agg(
        QB_Games    = ('WEEK', 'count'),
        QB_ATT      = ('ATT',  'sum'),
        QB_YDS      = ('YDS',  'sum'),
        QB_TD       = ('TD',   'sum'),
        QB_INT      = ('INT',  'sum'),
        QB_CPOE     = ('CPOE', 'mean'),
        QB_aDOT     = ('aDOT', 'mean'),
        QB_AY       = ('AY',   'sum'),
        QB_DeepPct  = ('Deep Throw %', 'mean'),
        QB_FPG      = ('FP/G', 'mean'),
        QB_FP       = ('FP',   'sum'),
    ).reset_index()
    agg['QB_ATT_per_game'] = agg['QB_ATT'] / agg['QB_Games']
    return agg
