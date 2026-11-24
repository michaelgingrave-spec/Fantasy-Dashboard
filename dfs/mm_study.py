"""Millionaire Maker study, part two: does building to the winners' structure pay before kickoff?

`dfs.hindsight` finds the lineup that scored the most in every stored week and describes its structure. That is selection on the outcome: a pattern can be common among the best
lineups only because the weeks' best players happened to fall that way. This module asks the forward-looking question instead. For each week it builds lineups the way an
ordinary optimizer would, from information available before kickoff, with and without a structural rule (a QB stack, a bring-back, a cap on players per team, ...), and
compares what those lineups then scored.

The pre-game expected points are the salary-implied projection: for every position, the points a player at that salary has averaged in the *other* seasons (a quadratic in salary,
refit with each season held out), so no week's own outcome leaks into its projection. Under it every lineup that spends the cap has about the same expected score, which makes
the comparison a clean test of structure at equal expected points: a rule can only win here by changing how the scores are *distributed* (a stack's correlation lifts the upper
tail), not by choosing better players. The 2026 weeks repeat the test with the FantasyPoints projections. Each lineup's objective is its projection with a random jitter, so
the lineups of a week differ the way a field's do; every rule set sees the same jitter.

A lineup is a top-5% (top-1%) lineup when it scores at least the 95th (99th) percentile of every lineup built that week by every rule set pooled, a stand-in for a field with many kinds of
lineups, and a rule's lift is its rate over the unconstrained baseline's. The yardstick matters: against the baseline's own lineups (alike, one objective and a jitter) nearly every rule set
looked better, opposite rules included, because any rule whose players differ from the baseline's lands above or below it by the luck of which players scored, and a bounded tail rate
averages above its nominal level under that luck (`_hits`).

Rebuilding the study (each run saves a CSV under data/dfs/history, named for its grid and projection source):

    python -m dfs.hindsight                                      # the best lineups and their features
    python -m dfs.mm_study --grid main --n 40                    # structure, shape and spend rule sets (about an hour on 6 cores)
    python -m dfs.mm_study --grid env --n 40                     # game totals and opposing defenses; `--shard 0/2` and `--shard 1/2` split it into halves
    python -m dfs.mm_study --grid main --source fp --n 60        # the 2026 weeks again with the FantasyPoints projections; so does `--grid env --source fp`
    python -m dfs.mm_study --evidence                            # combine the saved runs into mm_evidence.json, the file the screen reads
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from dfs import hindsight as HS
from dfs import history as H
from dfs.lineup_milp import PoolModel, Rules, solve_portfolio

POSITIONS = ("QB", "RB", "WR", "TE", "DST")


# ── Pre-game expected points ───────────────────────────────────────────────────────────────────────────
def fit_pricing(pools: list[pd.DataFrame]) -> dict[str, np.ndarray]:
    """Per position, quadratic coefficients (highest power first) of actual points on salary, over the given weeks' pools."""
    df = pd.concat([p[["pos", "salary", "pts"]] for p in pools], ignore_index=True)
    out = {}
    for pos in POSITIONS:
        d = df[df["pos"] == pos]
        out[pos] = np.polyfit(d["salary"].to_numpy(float) / 1000.0, d["pts"].to_numpy(float), 2)
    return out


def salary_projection(pool: pd.DataFrame, coefs: dict[str, np.ndarray]) -> np.ndarray:
    """What a player at this salary and position has scored on average (never below 0.5)."""
    proj = np.zeros(len(pool))
    for pos, c in coefs.items():
        m = (pool["pos"] == pos).to_numpy()
        proj[m] = np.polyval(c, pool.loc[m, "salary"].to_numpy(float) / 1000.0)
    return np.maximum(proj, 0.5)


def vegas_adjust(pool: pd.DataFrame, proj: np.ndarray, beta: float) -> np.ndarray:
    """Tilt the projection by the team's implied total against the slate's mean: proj * (1 + beta * (team_total / mean - 1)). `beta` 0 leaves it alone."""
    tt = pool["team_total"].to_numpy(float)
    mean = np.nanmean(tt)
    return proj * (1.0 + beta * (tt / mean - 1.0))


# ── The lineup population ──────────────────────────────────────────────────────────────────────────────
def prepare(pool: pd.DataFrame) -> pd.DataFrame:
    """The pool with the columns `Rules` reads: the game-total rank and, where there is one, the ownership estimate."""
    p = pool.reset_index(drop=True).copy()
    p["total_rank"] = p["game_total_rank"]
    p["own"] = p["own_pct"] if "own_pct" in p.columns else np.nan
    return p


def build_population(pool: pd.DataFrame, proj: np.ndarray, rules: Rules, noise: np.ndarray, min_uniques: int = 1) -> list[np.ndarray]:
    """One lineup per row of `noise` (n_lineups x n_players, a multiplicative jitter): the highest jittered-projection lineup the rules allow."""
    model = PoolModel(prepare(pool), rules)
    return solve_portfolio(model, lambda k: proj * (1.0 + noise[k]), n=len(noise), min_uniques=min_uniques)


def lineup_scores(pool: pd.DataFrame, sets: list[np.ndarray]) -> np.ndarray:
    pts = pool["pts"].to_numpy(float)
    return np.array([pts[s].sum() for s in sets])


def population_features(pool: pd.DataFrame, sets: list[np.ndarray]) -> pd.DataFrame:
    pool = pool.reset_index(drop=True)
    return pd.DataFrame([HS.lineup_features(pool["dk_id"].iloc[s].tolist(), pool) for s in sets])


# ── Baseline vs. rule sets ──────────────────────────────────────────────────────────────────────────────
def _week_job(args) -> list[dict]:
    season, week, coefs, rule_sets, n, jitter, beta, seed, with_features, source = args
    pool = HS.pool_context(season, week).reset_index(drop=True)
    if source == "fp":                                         # the FantasyPoints projection (2026 weeks); a player it does not list is worth nothing
        proj = pool["proj"].fillna(0.0).to_numpy(float)
    else:
        proj = vegas_adjust(pool, salary_projection(pool, coefs), beta)
    rng = np.random.default_rng([seed, season, week])
    noise = rng.uniform(-jitter, jitter, size=(n, len(pool)))
    rows = []
    for name, rules in rule_sets.items():
        sets = build_population(pool, proj, rules, noise)
        scores = lineup_scores(pool, sets)
        feats = population_features(pool, sets) if with_features else None
        for k, (s, sc) in enumerate(zip(sets, scores)):
            row = {"season": season, "week": week, "rules": name, "k": k, "score": float(sc), "proj": float(proj[s].sum()), "salary": float(pool["salary"].to_numpy()[s].sum())}
            if feats is not None:
                row.update({c: v for c, v in feats.iloc[k].items() if c not in ("qb_team", "qb_name", "score", "n_20plus", "n_tds", "top_player_share", "salary")})
            rows.append(row)
    return rows


def compare(rule_sets: dict[str, Rules], weeks: list[tuple[int, int]] | None = None, n: int = 60, jitter: float = 0.35, beta: float = 0.0, seed: int = 7,
            workers: int = 6, with_features: bool = False, source: str = "salary", fit_weeks: list[tuple[int, int]] | None = None) -> pd.DataFrame:
    """Lineups for every rule set in every week, all from the same projection and the same jitter. The pricing curve a week uses is fit on the other seasons only (of `fit_weeks`,
    default `weeks`: a run over part of the weeks passes them all, so its projections, and so its baseline lineups, match the full run's).
    `source='fp'` uses the FantasyPoints projection instead (only the 2026 weeks have one)."""
    weeks = weeks or HS.study_weeks()
    seasons = sorted({s for s, _ in weeks})
    pools = {(s, w): HS.pool_context(s, w) for s, w in (fit_weeks or weeks)}
    coefs_by_season = {}
    for s in seasons:
        others = [p for (s2, _), p in pools.items() if s2 != s] or list(pools.values())
        coefs_by_season[s] = fit_pricing(others)
    jobs = [(s, w, coefs_by_season[s], rule_sets, n, jitter, beta, seed, with_features, source) for s, w in weeks]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            parts = list(ex.map(_week_job, jobs, chunksize=2))
    else:
        parts = [_week_job(j) for j in jobs]
    return pd.DataFrame([r for part in parts for r in part])


# ── Summaries ──────────────────────────────────────────────────────────────────────────────────────────
REFERENCES = ("pooled", "baseline")


def _hits(df: pd.DataFrame, baseline: str, q: float, reference: str = "pooled") -> pd.Series:
    """1.0 for every lineup that scored at least the week's `q` quantile of the reference lineups, else 0.0 (aligned with `df`).

    The reference is every lineup built that week by every rule set ('pooled'): a field with the variety of all the ways the lineups were built. Measuring against the baseline's own
    lineups ('baseline') is kept for diagnosis and should not be used to compare rules: the baseline's lineups are alike (one optimizer, one objective, a jitter), so their
    week-quantile is a narrow yardstick, and any rule set whose players merely differ from the baseline's lands above or below it by the luck of which players scored. Because a
    rate is bounded and convex in that gap, the luck lifts the average rate above `1 - q` for every rule set, helpful or not (a cheap-QB rule and an expensive-QB rule both looked
    'better' that way)."""
    ref = df if reference == "pooled" else df[df["rules"] == baseline]
    thr = ref.groupby(["season", "week"])["score"].quantile(q).rename("_thr")
    return (df["score"] >= df.join(thr, on=["season", "week"])["_thr"]).astype(float)


def tail_rates(df: pd.DataFrame, baseline: str, qs=(0.5, 0.75, 0.9, 0.95, 0.99), reference: str = "pooled") -> pd.DataFrame:
    """For each rule set, the share of its lineups that scored at least the week's `q` quantile of the reference lineups (see `_hits`): 5% is what a lineup drawn from the whole
    field would get at q = 0.95, and a rule set at twice the baseline's rate lands twice as many lineups in a week's top tail."""
    hits = {q: _hits(df, baseline, q, reference) for q in qs}
    rows = []
    for name, g in df.groupby("rules"):
        row = {"rules": name, "lineups": len(g), "mean": g["score"].mean(), "sd": g["score"].std(), "mean_proj": g["proj"].mean()}
        for q in qs:
            row[f"top{round((1 - q) * 100)}%"] = float(hits[q].loc[g.index].mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("rules")


def week_bootstrap(df: pd.DataFrame, baseline: str, rule: str, q: float = 0.95, reps: int = 2000, seed: int = 1, reference: str = "pooled") -> tuple[float, float, float]:
    """Top-tail rate of `rule` divided by the baseline's (both against the same week quantile of the reference lineups), with a 95% interval from resampling WEEKS (a week's lineups
    share one slate, so they are not independent). Only the weeks in which both rule sets left a lineup count."""
    d = df.assign(hit=_hits(df, baseline, q, reference))
    per = d.groupby(["rules", "season", "week"])["hit"].mean().unstack("rules")[list(dict.fromkeys([baseline, rule]))].dropna()
    a, b = per[rule].to_numpy(), per[baseline].to_numpy()
    n = len(per)
    if n == 0 or b.mean() == 0:
        return float("nan"), float("nan"), float("nan")
    ix = np.random.default_rng(seed).integers(0, n, size=(reps, n))
    with np.errstate(divide="ignore", invalid="ignore"):
        boots = a[ix].mean(axis=1) / b[ix].mean(axis=1)
    lo, hi = np.nanpercentile(boots[np.isfinite(boots)], [2.5, 97.5]) if np.isfinite(boots).any() else (float("nan"), float("nan"))
    return float(a.mean() / b.mean()), float(lo), float(hi)


# ── Structural definitions and their prevalence ────────────────────────────────────────────────────────
FEATURE_DEFS = {
    "QB + 1 or more WR/TE teammates": lambda d: d["stack_wr_te"] >= 1,
    "QB + 2 or more WR/TE teammates": lambda d: d["stack_wr_te"] >= 2,
    "QB + a teammate RB, no WR/TE": lambda d: (d["stack_rb"] >= 1) & (d["stack_wr_te"] == 0),
    "no QB teammate at all": lambda d: d["stack_skill"] == 0,
    "bring-back: an opposing WR/TE": lambda d: d["bring_back_wr_te"] >= 1,
    "stack (1+) with a bring-back": lambda d: (d["stack_wr_te"] >= 1) & (d["bring_back_wr_te"] >= 1),
    "stack (2+) with a bring-back": lambda d: (d["stack_wr_te"] >= 2) & (d["bring_back_wr_te"] >= 1),
    "3+ players from the QB's game": lambda d: d["qb_game_players"] >= 3,
    "at most 3 offensive players from one team": lambda d: d["max_from_team"] <= 3,
    "4+ from one team": lambda d: d["max_from_team"] >= 4,
    "a second pair from another team": lambda d: d["n_mini_stacks"] >= 1,
    "6 to 8 teams": lambda d: d["n_teams"].between(6, 8),
    "5 to 7 games": lambda d: d["n_games"].between(5, 7),
    "DST 3,000 or less": lambda d: d["sal_dst"] <= 3000,
    "no player facing the lineup's DST": lambda d: d["dst_faces_lineup"] == 0,
    "DST does not face the QB": lambda d: d["dst_faces_qb"] == 0,
    "no player at 8,000+": lambda d: d["n_8k"] == 0,
    "2+ players at 8,000+": lambda d: d["n_8k"] >= 2,
    "2+ skill players at 4,000 or less": lambda d: d["n_le_4k"] >= 2,
    "RB in the FLEX": lambda d: d["flex_pos"] == "RB",
    "WR in the FLEX": lambda d: d["flex_pos"] == "WR",
    "TE in the FLEX": lambda d: d["flex_pos"] == "TE",
    "QB at 6,000 or less": lambda d: d["sal_qb"] <= 6000,
    "QB at 7,000 or more": lambda d: d["sal_qb"] >= 7000,
    "QB's game among the top 3 totals": lambda d: d["qb_total_rank"] <= 3,
    "QB's team a favorite": lambda d: d["qb_fav"] == 1,
    "2+ players from a top-3 total game": lambda d: d["n_top3_games"] >= 2,
    # how the cap is spent at each position
    "TE at 4,500 or less": lambda d: d["sal_te"] <= 4500,
    "TE at 5,500 or more": lambda d: d["sal_te"] >= 5500,
    "RBs 16,000 or more in all": lambda d: d["sal_rb"] >= 16000,
    "RBs 12,000 or less in all": lambda d: d["sal_rb"] <= 12000,
    "WRs 21,000 or more in all": lambda d: d["sal_wr"] >= 21000,
    "WRs 17,000 or less in all": lambda d: d["sal_wr"] <= 17000,
    "DST at 2,800 or less": lambda d: d["sal_dst"] <= 2800,
    "DST at 3,300 or more": lambda d: d["sal_dst"] >= 3300,
    # the opponents and offenses behind the players (as of kickoff; rank 1 = the weakest opponent for the position)
    "QB faces one of the 10 weakest defenses": lambda d: d["qb_fpa_rank"] <= 10,
    "QB faces one of the 8 toughest defenses": lambda d: d["qb_fpa_rank"] >= 25,
    "4+ players face a bottom-8 defense for their position": lambda d: d["n_weak_opp"] >= 4,
    "no player faces a top-8 defense": lambda d: d["n_strong_opp"] == 0,
    "DST faces one of the 10 weakest offenses": lambda d: d["dst_opp_off_rank"] <= 10,
    "4+ players from teams with a 25+ implied total": lambda d: d["n_high_team_total"] >= 4,
    "no player from a team at 20 or below": lambda d: d["n_low_team_total"] == 0,
    "QB's team an underdog": lambda d: d["qb_spread"] > 0,
    "4+ players from a top-8 offense": lambda d: d["n_top_offense"] >= 4,
}
_GROUPS = {
    "Stacking and teams": ["QB + 1 or more WR/TE teammates", "QB + 2 or more WR/TE teammates", "QB + a teammate RB, no WR/TE", "no QB teammate at all", "bring-back: an opposing WR/TE",
                           "stack (1+) with a bring-back", "stack (2+) with a bring-back", "3+ players from the QB's game", "at most 3 offensive players from one team",
                           "4+ from one team", "a second pair from another team", "6 to 8 teams", "5 to 7 games"],
    "Spend by position": ["QB at 6,000 or less", "QB at 7,000 or more", "TE at 4,500 or less", "TE at 5,500 or more", "RBs 16,000 or more in all", "RBs 12,000 or less in all",
                          "WRs 21,000 or more in all", "WRs 17,000 or less in all", "DST at 2,800 or less", "DST at 3,300 or more"],
    "Salary shape": ["no player at 8,000+", "2+ players at 8,000+", "2+ skill players at 4,000 or less", "RB in the FLEX", "WR in the FLEX", "TE in the FLEX"],
    "Defense pairing": ["no player facing the lineup's DST", "DST does not face the QB"],
    "Game environment": ["QB's game among the top 3 totals", "QB's team a favorite", "QB's team an underdog", "2+ players from a top-3 total game",
                         "4+ players from teams with a 25+ implied total", "no player from a team at 20 or below", "4+ players from a top-8 offense"],
    "Opponent quality (as of kickoff)": ["QB faces one of the 10 weakest defenses", "QB faces one of the 8 toughest defenses", "4+ players face a bottom-8 defense for their position",
                                         "no player faces a top-8 defense", "DST faces one of the 10 weakest offenses"],
}
FEATURE_GROUP = {name: group for group, names in _GROUPS.items() for name in names}
# The lineups a salary-only optimizer builds have habits of their own where the cap is split (a TE in the FLEX more than half the time, RBs priced higher than the winners'), so
# they are not a fair "ordinary" comparison for these groups: the best lineups' prevalence is shown on its own, and the forward test is what says whether a split pays.
NO_COMPARATOR_GROUPS = ("Spend by position", "Salary shape")
FEATURE_DEFS.pop("DST 3,000 or less", None)                    # superseded by the two DST spend traits
assert set(FEATURE_DEFS) == set(FEATURE_GROUP), set(FEATURE_DEFS) ^ set(FEATURE_GROUP)


def prevalence(feat: pd.DataFrame, defs: dict | None = None) -> pd.Series:
    """Share of lineups meeting each definition, averaged over weeks so every week counts once however many lineups it has."""
    out = {}
    for name, f in (defs or FEATURE_DEFS).items():
        out[name] = feat.assign(_m=f(feat).astype(float)).groupby(["season", "week"])["_m"].mean().mean()
    return pd.Series(out)


def prevalence_ci(feat: pd.DataFrame, defs: dict | None = None, reps: int = 1000, seed: int = 3) -> pd.DataFrame:
    """`prevalence` with a 95% interval from resampling weeks."""
    rng = np.random.default_rng(seed)
    per = pd.DataFrame({name: feat.assign(_m=f(feat).astype(float)).groupby(["season", "week"])["_m"].mean() for name, f in (defs or FEATURE_DEFS).items()})
    n = len(per)
    boots = np.array([per.iloc[rng.integers(0, n, n)].mean().to_numpy() for _ in range(reps)])
    return pd.DataFrame({"share": per.mean(), "lo": np.percentile(boots, 2.5, axis=0), "hi": np.percentile(boots, 97.5, axis=0)})


# ── The grid of rule sets the forward test compares ────────────────────────────────────────────────────
BASE = Rules(min_salary=49000)
RULE_GRID: dict[str, Rules] = {
    "base": BASE,
    "cap3": Rules(min_salary=49000, max_from_team=3),
    "stack1": Rules(min_salary=49000, qb_stack=1),
    "stack2": Rules(min_salary=49000, qb_stack=2),
    "stack1+back": Rules(min_salary=49000, qb_stack=1, bring_back=1),
    "stack2+back": Rules(min_salary=49000, qb_stack=2, bring_back=1),
    "stack1+back+cap3": Rules(min_salary=49000, qb_stack=1, bring_back=1, max_from_team=3),
    "stack2+back+cap3": Rules(min_salary=49000, qb_stack=2, bring_back=1, max_from_team=3),
    "stack2+back+dst": Rules(min_salary=49000, qb_stack=2, bring_back=1, dst_vs="lineup"),
    "stack2+back+spread": Rules(min_salary=49000, qb_stack=2, bring_back=1, min_games=5, min_teams=6),
}
# the salary-shape traits the best lineups showed (a cheaper QB, at most one $8,000+ player, no TE in the FLEX), alone and on top of the best structure
SHAPE = dict(max_qb_salary=6500, max_stars=1, star_salary=8000, max_te=1)
RULE_GRID_SHAPE: dict[str, Rules] = {
    "base": BASE,
    "qb<=6500": Rules(min_salary=49000, max_qb_salary=6500),
    "stars<=1": Rules(min_salary=49000, max_stars=1),
    "te=1": Rules(min_salary=49000, max_te=1),
    "shape": Rules(min_salary=49000, **SHAPE),
    "stack1+back+shape": Rules(min_salary=49000, qb_stack=1, bring_back=1, **SHAPE),
    "stack2+back+shape": Rules(min_salary=49000, qb_stack=2, bring_back=1, **SHAPE),
}
# how the cap is spent at each position: the best lineups under-spent at QB and TE, so each is tried cheap and expensive, plus stars-and-scrubs against balance and the FLEX position
RULE_GRID_SPEND: dict[str, Rules] = {
    "base": BASE,
    "qb<=6500": Rules(min_salary=49000, max_qb_salary=6500),
    "qb>=7000": Rules(min_salary=49000, price_window=(("QB", 7000, 99999),)),
    "te<=4500": Rules(min_salary=49000, price_window=(("TE", 0, 4500),)),
    "te>=5500": Rules(min_salary=49000, price_window=(("TE", 5500, 99999),)),
    "te=1": Rules(min_salary=49000, max_te=1),
    "stars<=1": Rules(min_salary=49000, max_stars=1),
    "stars=0": Rules(min_salary=49000, max_stars=0),
    "stars>=2": Rules(min_salary=49000, price_counts=(("SKILL", 8000, 99999, 2, 9),)),
    "rb>=16k": Rules(min_salary=49000, spend=(("RB", 16000, 50000),)),
    "rb<=12k": Rules(min_salary=49000, spend=(("RB", 0, 12000),)),
    "wr>=21k": Rules(min_salary=49000, spend=(("WR", 21000, 50000),)),
    "wr<=17k": Rules(min_salary=49000, spend=(("WR", 0, 17000),)),
    "flex=RB": Rules(min_salary=49000, counts=(("RB", 3, 3),)),
    "flex=WR": Rules(min_salary=49000, counts=(("WR", 4, 4),)),
    "flex=TE": Rules(min_salary=49000, counts=(("TE", 2, 2),)),
}
# the game around the players: totals, favorites, and how weak the opposing defense (offense, for the DST) was before kickoff. `env_fpa_rank` 1 = the weakest opponent for the player's position.
RULE_GRID_ENV: dict[str, Rules] = {
    "base": BASE,
    "qb_top5_total": Rules(min_salary=49000, max_qb_total_rank=5),
    "qb_underdog": Rules(min_salary=49000, env_bounds=(("QB", "spread", 1.0, 99.0),)),
    "qb_weak_opp": Rules(min_salary=49000, env_bounds=(("QB", "env_fpa_rank", 1, 10),)),
    "weak_opps>=5": Rules(min_salary=49000, env_counts=(("OFFENSE", "env_fpa_rank", 1, 12, 5, 8),)),
    "tough_opps<=1": Rules(min_salary=49000, env_counts=(("OFFENSE", "env_fpa_rank", 25, 32, 0, 1),)),
    "high_tt>=5": Rules(min_salary=49000, env_counts=(("OFFENSE", "team_total", 24.0, 99.0, 5, 8),)),
    "low_tt<=1": Rules(min_salary=49000, env_counts=(("OFFENSE", "team_total", 0.0, 20.0, 0, 1),)),
    "dst_weak_off": Rules(min_salary=49000, env_bounds=(("DST", "env_fpa_rank", 1, 10),)),
    "dst_fav": Rules(min_salary=49000, env_bounds=(("DST", "spread", -99.0, -1.0),)),
    "stack2+back+weak_opps": Rules(min_salary=49000, qb_stack=2, bring_back=1, env_counts=(("OFFENSE", "env_fpa_rank", 1, 12, 5, 8),)),
    "stack2+back+high_tt": Rules(min_salary=49000, qb_stack=2, bring_back=1, env_counts=(("OFFENSE", "team_total", 24.0, 99.0, 5, 8),)),
}
# a small grid for the robustness reruns (other jitter levels): the baseline and the stack rules the builder defaults to
RULE_GRID_CHECK: dict[str, Rules] = {k: RULE_GRID[k] for k in ("base", "stack1+back", "stack2", "stack2+back")}
GRIDS = {"structure": RULE_GRID, "shape": RULE_GRID_SHAPE, "spend": RULE_GRID_SPEND, "env": RULE_GRID_ENV, "check": RULE_GRID_CHECK}
GRIDS["main"] = {**RULE_GRID, **RULE_GRID_SHAPE, **RULE_GRID_SPEND}                  # every structure, shape and spend rule set once, one shared baseline
RULE_LABELS = {
    "base": "No structure rule (what an ordinary optimizer builds)",
    "cap3": "At most 3 from one team",
    "stack1": "QB + 1 pass catcher",
    "stack2": "QB + 2 pass catchers",
    "stack1+back": "QB + 1 pass catcher + a bring-back",
    "stack2+back": "QB + 2 pass catchers + a bring-back",
    "stack1+back+cap3": "QB + 1 + bring-back, at most 3 from a team",
    "stack2+back+cap3": "QB + 2 + bring-back, at most 3 from a team",
    "stack2+back+dst": "QB + 2 + bring-back, nobody facing the defense",
    "stack2+back+spread": "QB + 2 + bring-back, 5+ games, 6+ teams",
    "qb<=6500": "QB at $6,500 or less",
    "stars<=1": "At most one $8,000+ player",
    "te=1": "One TE (the FLEX is an RB or WR)",
    "shape": "Cheaper QB + at most one $8,000+ + one TE",
    "stack1+back+shape": "QB + 1 + bring-back + the salary shape",
    "stack2+back+shape": "QB + 2 + bring-back + the salary shape",
    "qb>=7000": "QB at $7,000 or more",
    "te<=4500": "TE at $4,500 or less",
    "te>=5500": "TE at $5,500 or more",
    "stars=0": "No $8,000+ player",
    "stars>=2": "Two or more $8,000+ players",
    "rb>=16k": "$16,000+ on RBs (RB-heavy)",
    "rb<=12k": "$12,000 or less on RBs (RB-light)",
    "wr>=21k": "$21,000+ on WRs (WR-heavy)",
    "wr<=17k": "$17,000 or less on WRs (WR-light)",
    "flex=RB": "FLEX is a third RB",
    "flex=WR": "FLEX is a fourth WR",
    "flex=TE": "FLEX is a second TE",
    "qb_top5_total": "QB's game among the 5 highest totals",
    "qb_underdog": "QB's team is an underdog",
    "qb_weak_opp": "QB faces one of the 10 weakest defenses against QBs",
    "weak_opps>=5": "5+ players face one of the 12 weakest defenses for their position",
    "tough_opps<=1": "At most 1 player faces one of the 8 toughest defenses for his position",
    "high_tt>=5": "5+ players from teams with an implied total of 24+",
    "low_tt<=1": "At most 1 player from a team with an implied total of 20 or less",
    "dst_weak_off": "DST faces one of the 10 weakest offenses",
    "dst_fav": "DST's team is a favorite by a point or more",
    "stack2+back+weak_opps": "QB + 2 + bring-back, 5+ players against weak defenses",
    "stack2+back+high_tt": "QB + 2 + bring-back, 5+ players from implied totals of 24+",
}
# which question each rule set answers (the screen shows one table per group)
RULE_GROUPS = {
    "structure": ("cap3", "stack1", "stack2", "stack1+back", "stack2+back", "stack1+back+cap3", "stack2+back+cap3", "stack2+back+dst", "stack2+back+spread"),
    "spend": ("qb<=6500", "qb>=7000", "te<=4500", "te>=5500", "te=1", "stars<=1", "stars=0", "stars>=2", "rb>=16k", "rb<=12k", "wr>=21k", "wr<=17k", "flex=RB", "flex=WR", "flex=TE"),
    "environment": ("qb_top5_total", "qb_underdog", "qb_weak_opp", "weak_opps>=5", "tough_opps<=1", "high_tt>=5", "low_tt<=1", "dst_weak_off", "dst_fav"),
    "combined": ("shape", "stack1+back+shape", "stack2+back+shape", "stack2+back+weak_opps", "stack2+back+high_tt"),
}
GROUP_TITLES = {"structure": "Stacks and teams", "spend": "Spending by position", "environment": "Game totals and opposing defenses", "combined": "Combinations"}
_ALL_RULES = {k for grid in GRIDS.values() for k in grid} - {"base"}
assert _ALL_RULES == {k for keys in RULE_GROUPS.values() for k in keys}, _ALL_RULES ^ {k for keys in RULE_GROUPS.values() for k in keys}


def rule_group(key: str) -> str:
    return next((g for g, keys in RULE_GROUPS.items() if key in keys), "")


EVIDENCE_FILE = H.HISTORY_DIR / "mm_evidence.json"


def combine_runs(*runs: pd.DataFrame) -> pd.DataFrame:
    """Forward runs made over the same weeks with the same lineups per week, seed and jitter draw the same jitter, so their `base` lineups are the same lineups. This stacks the
    rule sets of every run under the first run's baseline (a rule set that appears twice is kept once). Raises when two runs' baselines differ: lifts against different
    baselines cannot share a table."""
    first = runs[0]
    key = ["season", "week", "k"]
    base0 = first[first["rules"] == "base"].set_index(key)["score"].sort_index()
    out, seen = [first], set(first["rules"])
    for run in runs[1:]:
        base = run[run["rules"] == "base"].set_index(key)["score"].sort_index()
        if not base.index.equals(base0.index) or not np.allclose(base.to_numpy(), base0.to_numpy()):
            raise ValueError("these forward runs do not share a baseline (different weeks, lineups per week, seed or jitter)")
        new = run[~run["rules"].isin(seen)]
        out.append(new)
        seen |= set(new["rules"])
    return pd.concat(out, ignore_index=True)


def _finite(x: float) -> float | None:
    return float(x) if x is not None and np.isfinite(x) else None


def forward_table(df: pd.DataFrame, baseline: str = "base", reps: int = 1000) -> dict:
    """The forward test as one table: each rule set's mean score and its top-5% and top-1% rates against the week's pooled lineups (every rule set's), and the lift over the
    baseline's rate at each, with week-bootstrap intervals. `verdict` reads the top-5% lift (the better-powered one), `verdict1` the top-1% lift."""
    tr = tail_rates(df, baseline)
    rows = []
    for name in tr.index:
        est, lo, hi = (1.0, 1.0, 1.0) if name == baseline else week_bootstrap(df, baseline, name, q=0.95, reps=reps)
        est1, lo1, hi1 = (1.0, 1.0, 1.0) if name == baseline else week_bootstrap(df, baseline, name, q=0.99, reps=reps)
        rows.append({"rules": RULE_LABELS.get(name, name), "key": name, "group": rule_group(name), "lineups": int(tr.loc[name, "lineups"]),
                     "weeks": int(df[df["rules"] == name].groupby(["season", "week"]).ngroups),
                     "mean": float(tr.loc[name, "mean"]), "sd": float(tr.loc[name, "sd"]), "mean_proj": float(tr.loc[name, "mean_proj"]),
                     "top5": float(tr.loc[name, "top5%"]), "top1": float(tr.loc[name, "top1%"]),
                     "lift": _finite(est), "lo": _finite(lo), "hi": _finite(hi), "lift1": _finite(est1), "lo1": _finite(lo1), "hi1": _finite(hi1),
                     "verdict": "baseline" if name == baseline else verdict(est, lo, hi), "verdict1": "baseline" if name == baseline else verdict(est1, lo1, hi1)})
    weeks = df.groupby(["season", "week"]).ngroups
    return {"n_weeks": int(weeks), "lineups_per_rule": int(len(df) // max(1, df["rules"].nunique())), "reference": "pooled", "rows": rows}


def hindsight_table(hind: pd.DataFrame, base: pd.DataFrame, reps: int = 1000) -> list[dict]:
    """Every structural trait: the share of weeks whose single best lineup had it (with an interval), the share among each week's best 25, and the share among typical lineups."""
    r1 = hind[hind["rank"] == 1]
    ci = prevalence_ci(r1, reps=reps)
    try:
        p25, typ = prevalence(hind), prevalence(base)
    except KeyError as ex:
        raise ValueError(f"the lineup table lacks the feature column {ex}: the baseline lineups must come from a forward run made with the current `lineup_features` "
                         "(put that run first in `combine_runs`)") from ex
    rows = []
    for name in FEATURE_DEFS:
        t = float(typ[name])
        fair = FEATURE_GROUP.get(name, "") not in NO_COMPARATOR_GROUPS
        rows.append({"trait": name, "group": FEATURE_GROUP.get(name, ""), "rank1": float(ci.loc[name, "share"]), "rank1_lo": float(ci.loc[name, "lo"]), "rank1_hi": float(ci.loc[name, "hi"]), "top25": float(p25[name]),
                     "typical": t if fair else None, "lift": float(p25[name] / t) if fair and t > 0 else None})
    return rows


def verdict(lift: float, lo: float, hi: float) -> str:
    """Whether a rule set's top-tail lift cleared 1.0 by more than the week-to-week noise: 'better', 'worse' or 'unclear'."""
    if lift is None or lo is None or hi is None or not (np.isfinite(lift) and np.isfinite(lo) and np.isfinite(hi)):
        return "unclear"
    return "better" if lo > 1.0 else "worse" if hi < 1.0 else "unclear"


def _forward_summary(table: dict, label: str) -> list[str]:
    rows = [r for r in table["rows"] if r["key"] != "base"]
    out = []
    better = sorted((r for r in rows if r.get("verdict") == "better"), key=lambda r: -r["lift"])
    worse = [r for r in rows if r.get("verdict") == "worse"]
    unclear = [r for r in rows if r.get("verdict") not in ("better", "worse")]
    if better:
        top = better[0]
        top1 = f"; {top['lift1']:.1f}x in the top 1%" if top.get("lift1") else ""
        out.append(f"{label}: {top['rules'].lower()} put {top['lift']:.2f}x as many lineups in a week's top 5% as ordinary lineups (95% interval {top['lo']:.2f}-{top['hi']:.2f}{top1}); "
                   f"{len(better)} of {len(rows)} rule sets cleared 1.0.")
    else:
        out.append(f"{label}: no rule set cleared 1.0 on the top-5% rate by more than week-to-week noise.")
    if worse:
        out.append("Rules that hurt: " + "; ".join(f"{r['rules'].lower()} ({r['lift']:.2f}x)" for r in worse) + ".")
    if unclear:
        names = "; ".join(r["rules"].lower() for r in unclear) if len(unclear) <= 6 else f"{len(unclear)} of them"
        out.append(f"{len(unclear)} rule set(s) were indistinguishable from ordinary lineups: {names}." if len(unclear) <= 6 else f"{len(unclear)} rule sets were indistinguishable from ordinary lineups.")
    return out


CAVEATS = [
    "The best lineups are chosen after the scores are known, so a trait can be common among them only because of which players happened to score. The forward test is the check on that.",
    "Salaries are on file for 2014-2021 and 2026 only (138 main slates). DraftKings pricing and the league's scoring environment have changed since, so older weeks count as evidence, not as a forecast.",
    "The forward test projects players from their salary (what a player at that price has averaged in the other seasons), so every lineup that spends the cap has about the same expected points. "
    "It isolates what structure does to the spread of scores. It says nothing about whether your projections beat the salary-implied ones.",
    "A lineup counts as a top-5% (top-1%) lineup when it beats the 95th (99th) percentile of every lineup built that week by every rule set, a stand-in for a field with many kinds of lineups. "
    "The lift is a rule set's rate over the unconstrained optimizer's. Measured against the unconstrained lineups alone, nearly every rule looked better (even opposite ones), because those lineups are too alike to "
    "serve as a yardstick; against the pooled field, rules that only change which players are used (the spending rules) show no lift.",
    "Forty-odd rule sets were tested, so a couple would clear a 95% interval by chance alone: a single 'better' that is not a stack, and does not repeat on 2026 with the FantasyPoints projections, is not evidence.",
    "Ownership rules are not tested: there is no historical ownership data, so the screen's ownership figures are a heuristic ordering.",
    "The Millionaire Maker pays its top prize to roughly one lineup in a hundred thousand. This study measures landing in the top 5% and 1% of ordinary lineups, the closest proxy a sample this size allows, not contest results.",
]


def build_evidence(forward: pd.DataFrame, fp: pd.DataFrame | None = None, hind: pd.DataFrame | None = None, pools: pd.DataFrame | None = None,
                   reps: int = 1000, write: bool = True) -> dict:
    """The study as one JSON-ready dict the screen shows: the hindsight traits, the forward test over every rule set (`forward`: `combine_runs` of the structure / spend /
    environment runs, each from `compare(..., with_features=True)`), the 2026 FantasyPoints rerun (`fp`), how the cap is split by position and what each price tier returned
    and whether the game is already in the price (those two need `pools`, `mm_spend.load_pools()`), and the caveats. `hind` is the committed hindsight feature file."""
    import json
    from datetime import datetime

    from dfs import mm_spend as MS
    hind = HS.read_features() if hind is None else hind
    typical = forward[forward["rules"] == "base"]
    ev: dict = {"built": datetime.now().strftime("%Y-%m-%d"), "weeks": {"hindsight": int(hind.groupby(["season", "week"]).ngroups), "forward": int(forward.groupby(["season", "week"]).ngroups)}}
    ev["hindsight"] = hindsight_table(hind, typical, reps=reps)
    fwd = forward_table(forward, "base", reps=reps)
    ev["forward"] = fwd
    ev["group_titles"] = GROUP_TITLES
    ev["spend"] = {"slots": MS.spend_table(hind, reps=reps)}
    if pools is not None:
        ev["spend"]["tiers"] = MS.price_tier_table(pools, reps=reps)
        ev["environment"] = {"effects": MS.environment_effects(pools)}
    summary = [f"Best possible lineup of each of {ev['weeks']['hindsight']} past weeks: " + "; ".join(
        f"{r['trait']} in {100 * r['rank1']:.0f}% (ordinary lineups {100 * r['typical']:.0f}%)" for r in ev["hindsight"]
        if r["trait"] in ("QB + 1 or more WR/TE teammates", "stack (1+) with a bring-back", "at most 3 offensive players from one team")) + "."]
    for group, title in GROUP_TITLES.items():
        part = {**fwd, "rows": [r for r in fwd["rows"] if r["group"] == group]}
        if part["rows"]:
            summary += _forward_summary(part, f"{title}, {fwd['n_weeks']} weeks before kickoff")
    if fp is not None:
        ev["weeks"]["fp"] = int(fp.groupby(["season", "week"]).ngroups)
        ev["forward_fp"] = forward_table(fp, "base", reps=reps)
        summary += _forward_summary(ev["forward_fp"], f"With the FantasyPoints projections ({ev['weeks']['fp']} weeks of 2026)")
    ev["summary"] = summary
    ev["caveats"] = CAVEATS
    if write:
        EVIDENCE_FILE.write_text(json.dumps(ev, indent=1), encoding="utf-8")
    return ev


def run_file(grid: str, source: str = "salary", shard: str = "0/1"):
    """Where a forward run is saved by default: one file per grid and projection source (`mm_forward_main_local.csv`, `mm_forward_fpenv_local.csv`), the shards of a split run side by side."""
    i, n = (int(x) for x in shard.split("/"))
    return H.HISTORY_DIR / f"mm_forward_{'fp' if source == 'fp' else ''}{grid}{f'{i}of{n}' if n > 1 else ''}_local.csv"


# the grids whose saved runs make up the evidence (`check` is a robustness rerun at another jitter, so it has its own baseline and stays out)
EVIDENCE_GRIDS = ("env", "main", "structure", "shape", "spend")


def read_runs(source: str = "salary") -> pd.DataFrame | None:
    """Every saved forward run of one projection source, as one table under one baseline (None when there is none). The shards of a split run are stacked; the grids are combined
    with `combine_runs`, which keeps each rule set once, and the run with the most columns goes first because its baseline lineups are the ones the hindsight table reads."""
    prefix = "fp" if source == "fp" else ""
    runs = []
    for grid in EVIDENCE_GRIDS:
        files = sorted(H.HISTORY_DIR.glob(f"mm_forward_{prefix}{grid}*_local.csv"))
        if files:
            runs.append(pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates(["season", "week", "rules", "k"]))
    if not runs:
        return None
    runs.sort(key=lambda r: -r.shape[1])
    return combine_runs(*runs)


def evidence_main() -> None:
    """Combine the saved forward runs with the hindsight features and the price / environment pools into the evidence file the screen reads."""
    from dfs import mm_spend as MS
    forward = read_runs("salary")
    if forward is None:
        raise SystemExit(f"no forward runs under {H.HISTORY_DIR}: run `python -m dfs.mm_study --grid main` and `--grid env` first")
    ev = build_evidence(forward, fp=read_runs("fp"), hind=HS.read_features(), pools=MS.load_pools(), write=True)
    print(f"wrote {EVIDENCE_FILE}")
    for line in ev["summary"]:
        print("-", line)


def forward_main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Forward test: lineups built with and without each structural rule, scored on what really happened.")
    ap.add_argument("--n", type=int, default=60, help="lineups per rule set per week")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--jitter", type=float, default=0.35)
    ap.add_argument("--beta", type=float, default=0.0, help="Vegas tilt on the salary-implied projection")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None, help="where to save the run (default: data/dfs/history/mm_forward_<grid>_local.csv, 'fp' before the grid for the FantasyPoints rerun)")
    ap.add_argument("--grid", choices=sorted(GRIDS), default="structure")
    ap.add_argument("--source", choices=("salary", "fp"), default="salary")
    ap.add_argument("--shard", default="0/1", help="'i/n': run every n-th study week starting at i (the projections are still fit on all weeks, so shards stack into the full run)")
    ap.add_argument("--evidence", action="store_true", help="instead of a run: combine the saved runs into data/dfs/history/mm_evidence.json, the file the screen reads")
    args = ap.parse_args(argv)
    if args.evidence:
        return evidence_main()
    args.out = args.out or str(run_file(args.grid, args.source, args.shard))
    weeks = [sw for sw in HS.study_weeks() if sw[0] == 2026] if args.source == "fp" else None      # the FantasyPoints projections exist for the 2026 weeks only
    fit_weeks = None
    i, n_shards = (int(x) for x in args.shard.split("/"))
    if n_shards > 1:
        fit_weeks = weeks or HS.study_weeks()
        weeks = fit_weeks[i::n_shards]
    df = compare(GRIDS[args.grid], weeks=weeks, n=args.n, jitter=args.jitter, beta=args.beta, seed=args.seed, workers=args.workers, with_features=True, source=args.source, fit_weeks=fit_weeks)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} lineups to {args.out}")
    print(tail_rates(df, "base").round(3).to_string())


if __name__ == "__main__":
    forward_main()
