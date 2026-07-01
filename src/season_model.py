"""
season_model.py
Train a year-over-year season FP projection model per position.
Uses XGBoost with leave-one-year-out cross-validation.
Outputs 2026 projected season FP for all players with 2025 data.

For players in their 1st or 2nd year in the data (rookies / sophomores),
a comparable-player system blends XGBoost predictions with the historical
outcomes of statistically similar players in the same career stage.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from features import SEASON_FEATURES


# ── Comp system tuning knobs ──────────────────────────────────────────────────
# How many historical comps to search for per player
N_COMPS = 8

# A historical player must have played this many games to count as a comp
# (filters out backups whose stats don't reflect a real starting opportunity)
COMP_MIN_GAMES = 8

# How much of the final projection to draw from comps vs. XGBoost
# Set to 0.0 to disable comps entirely; 1.0 for comps only.
ROOKIE_COMP_WEIGHT    = 0.60   # career year 1: 60% comp, 40% XGBoost
SOPHOMORE_COMP_WEIGHT = 0.35   # career year 2: 35% comp, 65% XGBoost

# Features used to measure similarity between players (per position)
# These are all per-game or rate-based so game count doesn't distort the match
_SIM_FEATURES = {
    "WR": ["FPG_mean", "YPRR_mean", "TGT_per_game", "aDOT_mean",
           "RTE_per_game", "XFP_per17", "Games"],
    "TE": ["FPG_mean", "YPRR_mean", "TGT_per_game", "aDOT_mean",
           "RTE_per_game", "XFP_per17", "Games"],
    "RB": ["FPG_mean", "XFPG_mean", "ATT_per_game", "TGT_per_game",
           "rush_share", "YPRR_mean", "Games"],
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the feature columns present in df, filling missing with median."""
    cols = [c for c in SEASON_FEATURES if c in df.columns]
    X = df[cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    return X


def _build_comp_pool(season_stats: pd.DataFrame,
                     career_year: int,
                     pos: str) -> pd.DataFrame:
    """
    Build a pool of historical player-seasons in their career_year-th season
    where we also know their NEXT season's FP total (i.e. the target outcome).

    career_year is computed as (Year - first_year_in_data + 1), not years_in_data,
    so a player who appeared 2022-2025 gets career_year=1 for their 2022 row —
    not career_year=4 (which years_in_data would give across the full dataset).

    Returns a DataFrame with all season features plus 'FP_next'.
    """
    # First year each player appears in our data (per position)
    first_years = (
        season_stats.groupby(["Name", "POS"])["Year"]
        .min()
        .reset_index()
        .rename(columns={"Year": "first_year"})
    )

    ss = season_stats.merge(first_years, on=["Name", "POS"], how="left")
    ss["_career_yr"] = ss["Year"] - ss["first_year"] + 1

    # Year-N players in the right career stage
    year_n = ss[
        (ss["POS"] == pos) &
        (ss["_career_yr"] == career_year) &
        (ss["Games"] >= COMP_MIN_GAMES)
    ].copy()

    # For true-rookie comps (career_year == 1): exclude players whose
    # first appearance is 2021 — those may be veterans re-entering our data
    # rather than actual NFL rookies, since our data starts in 2021.
    # Players first appearing in 2022+ are almost certainly genuine rookies.
    if career_year == 1:
        year_n = year_n[year_n["Year"] >= 2022]

    if year_n.empty:
        return pd.DataFrame()

    # Pull next-year FP totals and join by (Name, POS, Year+1)
    next_yr = (
        season_stats[season_stats["POS"] == pos]
        [["Name", "POS", "Year", "FP_total"]]
        .copy()
        .rename(columns={"FP_total": "FP_next"})
    )
    next_yr["Year"] = next_yr["Year"] - 1  # align: next_yr.Year now = the PRIOR year

    pool = year_n.merge(next_yr, on=["Name", "POS", "Year"], how="inner")
    return pool.drop(columns=["_career_yr", "first_year"], errors="ignore")


def _find_comp_projection(player_row: pd.Series,
                           comp_pool: pd.DataFrame,
                           pos: str) -> tuple:
    """
    Find the N most statistically similar players in comp_pool and return
    their inverse-distance-weighted average Year+1 FP.

    Returns (comp_fp: float, comp_names: str, n_used: int).
    Returns (nan, '', 0) when the pool is too small.
    """
    if len(comp_pool) < 4:
        return np.nan, "", 0

    feats = [f for f in _SIM_FEATURES.get(pos, _SIM_FEATURES["WR"])
             if f in comp_pool.columns and f in player_row.index]
    if not feats:
        return np.nan, "", 0

    pool_vals   = comp_pool[feats].fillna(0).values.astype(float)
    player_vals = player_row[feats].fillna(0).values.astype(float)

    # Z-score normalise so each feature contributes equally
    means = pool_vals.mean(axis=0)
    stds  = pool_vals.std(axis=0)
    stds[stds == 0] = 1.0

    pool_z   = (pool_vals - means) / stds
    player_z = (player_vals - means) / stds

    dists = np.sqrt(((pool_z - player_z) ** 2).sum(axis=1))

    # Take top-N closest comps
    top_idx  = np.argsort(dists)[:N_COMPS]
    top_pool = comp_pool.iloc[top_idx].reset_index(drop=True)
    top_dist = dists[top_idx]

    # Inverse-distance weights; cap any single comp at 40% of total weight
    raw_weights = 1.0 / (top_dist + 0.01)
    raw_weights = np.minimum(raw_weights, 0.40 * raw_weights.sum())
    raw_weights /= raw_weights.sum()

    comp_fp = float(np.dot(top_pool["FP_next"].values, raw_weights))

    # Human-readable comp list: "Player (Season): FPnext"
    comp_names = ", ".join(
        f"{r['Name']} ({int(r['Year']) + 1}): {r['FP_next']:.0f}FP"
        for _, r in top_pool.iterrows()
    )

    return round(comp_fp, 1), comp_names, len(top_pool)


# ── Public API ────────────────────────────────────────────────────────────────

def train_and_evaluate(pairs: pd.DataFrame, pos: str) -> dict:
    """
    Train XGBoost on YoY pairs for a given position using leave-one-year-out CV.
    Returns dict with model, feature names, CV metrics, and per-year results.
    """
    sub   = pairs[pairs["POS"] == pos].copy()
    years = sorted(sub["Year"].unique())

    cv_results = []
    for holdout_year in years:
        train = sub[sub["Year"] != holdout_year]
        test  = sub[sub["Year"] == holdout_year]

        X_train = _get_features(train)
        y_train = train["FP_next"]
        X_test  = _get_features(test)
        y_test  = test["FP_next"]

        model = XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = np.clip(model.predict(X_test), 0, None)

        cv_results.append({
            "holdout_year": holdout_year + 1,
            "n_test":       len(test),
            "MAE":          round(mean_absolute_error(y_test, preds), 2),
            "R2":           round(r2_score(y_test, preds), 3),
        })

    # Final model trained on ALL pairs
    X_all = _get_features(sub)
    y_all = sub["FP_next"]
    final_model = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
    )
    final_model.fit(X_all, y_all, verbose=False)

    feature_names = [c for c in SEASON_FEATURES if c in sub.columns]

    return {
        "model":         final_model,
        "feature_names": feature_names,
        "cv_results":    cv_results,
        "pos":           pos,
        "n_train":       len(sub),
    }


def project_2026(season_stats_2025: pd.DataFrame,
                 trained_models: dict,
                 season_stats_full: pd.DataFrame = None,
                 verbose: bool = True) -> pd.DataFrame:
    """
    Use 2025 season stats to project 2026 FP totals for all eligible players.

    season_stats_2025 : pre-filtered 2025 stats (with OC, trend, and coverage features
                        already merged in from main.py).
    season_stats_full : the full 2021-2025 season stats used to build comp pools.
                        When supplied, rookie/sophomore projections are blended with
                        historical-comp outcomes. Pass None to skip comp blending.
    verbose           : if True, print a table of comp-adjusted players.

    Returns a sorted DataFrame with Proj_FP_2026, XGB_FP, Comp_FP, Comp_Names,
    POS_Rank, and Overall_Rank columns.
    """
    # Filter to 2025 if the caller passed the full stats by mistake
    stats_2025 = season_stats_2025[season_stats_2025["Year"] == 2025].copy()

    # Build per-position comp pools up front (only when full stats are available)
    comp_pools = {}   # {(pos, career_year): DataFrame}
    if season_stats_full is not None:
        for pos in ["WR", "TE", "RB"]:
            for cy in [1, 2]:
                pool = _build_comp_pool(season_stats_full, cy, pos)
                comp_pools[(pos, cy)] = pool

    all_projections = []

    for pos in ["WR", "TE", "RB"]:
        sub = stats_2025[stats_2025["POS"] == pos].copy()
        result = trained_models[pos]
        feat_names = result["feature_names"]

        # ── XGBoost base projection ───────────────────────────────────────────
        X = sub[feat_names].copy().fillna(sub[feat_names].median(numeric_only=True))
        preds = np.clip(result["model"].predict(X), 0, None)
        sub["XGB_FP"]      = preds.round(1)
        sub["Proj_FP_2026"] = sub["XGB_FP"]
        sub["Comp_FP"]      = np.nan
        sub["Comp_Names"]   = ""

        # ── Comp blending for rookies / sophomores ────────────────────────────
        if season_stats_full is not None and "years_in_data" in sub.columns:
            for idx, row in sub.iterrows():
                career_yr = int(row.get("years_in_data", 99))
                if career_yr > 2:
                    continue   # veteran — XGBoost handles these well

                pool = comp_pools.get((pos, career_yr), pd.DataFrame())
                comp_fp, comp_names, n_used = _find_comp_projection(row, pool, pos)

                if np.isnan(comp_fp) or n_used < 4:
                    continue   # not enough comps — fall back to pure XGBoost

                weight = (ROOKIE_COMP_WEIGHT if career_yr == 1
                          else SOPHOMORE_COMP_WEIGHT)
                blended = weight * comp_fp + (1 - weight) * row["XGB_FP"]

                sub.loc[idx, "Comp_FP"]      = comp_fp
                sub.loc[idx, "Comp_Names"]   = comp_names
                sub.loc[idx, "Proj_FP_2026"] = round(blended, 1)

        # Rank within position
        sub = sub.sort_values("Proj_FP_2026", ascending=False)
        sub["POS_Rank"] = range(1, len(sub) + 1)

        keep = ["Name", "Team", "POS", "POS_Rank",
                "Games", "FP_total", "XFP_total",
                "TGT_total", "TGT_per_game", "aDOT_mean", "AYshare_mean",
                "years_in_data",
                "XGB_FP", "Comp_FP", "Comp_Names", "Proj_FP_2026"]
        all_projections.append(sub[[c for c in keep if c in sub.columns]])

    projections = pd.concat(all_projections, ignore_index=True)
    projections = projections.sort_values("Proj_FP_2026", ascending=False)
    projections.insert(0, "Overall_Rank", range(1, len(projections) + 1))

    # ── Print comp adjustments ────────────────────────────────────────────────
    if verbose and season_stats_full is not None:
        comp_adj = projections[projections["Comp_FP"].notna()].copy()
        comp_adj["comp_delta"] = comp_adj["Proj_FP_2026"] - comp_adj["XGB_FP"]
        comp_adj = comp_adj.sort_values("comp_delta", ascending=False)

        if not comp_adj.empty:
            print(f"\n  Rookie/Sophomore comp adjustments  "
                  f"(rookieWt={ROOKIE_COMP_WEIGHT:.0%}  "
                  f"sophWt={SOPHOMORE_COMP_WEIGHT:.0%}  "
                  f"n_comps={N_COMPS}):")
            print(f"  {'Player':<22} {'POS':<3} {'Yr':>2}  "
                  f"{'XGB':>6} {'Comp':>6} {'Final':>6}  {'Chg':>6}  Top comps")
            print(f"  {'-'*22} {'-'*3} {'-'*2}  "
                  f"{'-'*6} {'-'*6} {'-'*6}  {'-'*6}  {'-'*50}")
            for _, r in comp_adj.iterrows():
                delta = r["comp_delta"]
                cnames = str(r["Comp_Names"])[:70] if pd.notna(r["Comp_Names"]) else ""
                yr = int(r.get("years_in_data", 0)) if pd.notna(r.get("years_in_data")) else 0
                print(f"  {r['Name']:<22} {r['POS']:<3} {yr:>2}  "
                      f"{r['XGB_FP']:>6.1f} {r['Comp_FP']:>6.1f} "
                      f"{r['Proj_FP_2026']:>6.1f}  {delta:>+6.1f}  {cnames}")

    return projections


def get_feature_importance(trained_models: dict) -> pd.DataFrame:
    """Return a combined feature importance table across all positions."""
    rows = []
    for pos, result in trained_models.items():
        model = result["model"]
        feat  = result["feature_names"]
        scores = model.feature_importances_
        for f, s in zip(feat, scores):
            rows.append({"POS": pos, "Feature": f, "Importance": round(s, 4)})
    df = pd.DataFrame(rows).sort_values(["POS", "Importance"], ascending=[True, False])
    return df
