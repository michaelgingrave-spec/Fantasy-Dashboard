"""The forward-test helpers behind the Millionaire Maker evidence: pricing curve, tail rates, bootstrap, structural prevalence, and one real end-to-end comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_hindsight import tiny_pool

from dfs import hindsight as HS
from dfs import mm_study as S
from dfs.lineup_milp import Rules


def test_pricing_curve_recovers_a_known_relation():
    rng = np.random.default_rng(0)
    rows = []
    for pos, slope in (("QB", 3.0), ("RB", 2.0), ("WR", 2.0), ("TE", 2.0), ("DST", 2.0)):
        sal = rng.integers(2500, 9000, 400)
        rows.append(pd.DataFrame({"pos": pos, "salary": sal, "pts": slope * sal / 1000 + rng.normal(0, 2, 400)}))
    coefs = S.fit_pricing([pd.concat(rows, ignore_index=True)])
    pool = pd.DataFrame({"pos": ["QB", "RB", "DST"], "salary": [6000, 6000, 3000]})
    proj = S.salary_projection(pool, coefs)
    assert proj == pytest.approx([18.0, 12.0, 6.0], abs=0.8)
    assert S.salary_projection(pd.DataFrame({"pos": ["WR"], "salary": [100]}), coefs)[0] >= 0.5        # never below the floor


def test_vegas_adjust_tilts_by_team_total():
    pool = pd.DataFrame({"team_total": [30.0, 20.0]})
    proj = np.array([10.0, 10.0])
    assert S.vegas_adjust(pool, proj, 0.0).tolist() == [10.0, 10.0]
    tilted = S.vegas_adjust(pool, proj, 0.5)
    assert tilted[0] > 10 > tilted[1] and tilted.mean() == pytest.approx(10.0, abs=0.2)


def _runs():
    rows = []
    rng = np.random.default_rng(1)
    for w in range(1, 41):
        for name, shift in (("base", 0.0), ("good", 6.0)):
            for k in range(50):
                rows.append({"season": 2020, "week": w, "rules": name, "k": k, "score": float(rng.normal(120 + shift, 15)), "proj": 130.0})
    return pd.DataFrame(rows)


def test_tail_rates_and_bootstrap():
    df = _runs()
    own = S.tail_rates(df, "base", reference="baseline")
    assert own.loc["base", "top5%"] == pytest.approx(0.05, abs=0.02)                 # against its own quantile the baseline is at 5% by construction
    t = S.tail_rates(df, "base")
    assert t.loc["good", "top5%"] > 0.055 and t.loc["base", "top5%"] < 0.045 and t.loc["good", "mean"] > t.loc["base", "mean"] + 4      # pooled with a better rule set, the baseline falls below 5%
    est, lo, hi = S.week_bootstrap(df, "base", "good", q=0.95, reps=300)
    assert lo < est < hi and lo > 1.0                                                 # a real shift is detected
    est0, lo0, hi0 = S.week_bootstrap(df, "base", "base", q=0.95, reps=300)
    assert est0 == pytest.approx(1.0) and lo0 <= 1.0 <= hi0


def _placebo_runs(sd_by_rule: dict | None = None, n_weeks: int = 120, n_rules: int = 6, seed: int = 3) -> pd.DataFrame:
    """Rule sets that differ from each other only in WHICH players they happened to use: each week every rule set (the baseline included) gets its own lucky or unlucky shift of its
    mean. A rule's spread within the week can be set to make it a real tail-fattener."""
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(1, n_weeks + 1):
        for name in ["base"] + [f"r{i}" for i in range(n_rules)]:
            shift = rng.normal(0, 10)
            sd = (sd_by_rule or {}).get(name, 8.0)
            for k in range(40):
                rows.append({"season": 2020, "week": w, "rules": name, "k": k, "score": float(120 + shift + rng.normal(0, sd)), "proj": 120.0})
    return pd.DataFrame(rows)


def test_a_rule_that_only_changes_which_players_are_used_shows_no_lift():
    df = _placebo_runs()
    rules = [f"r{i}" for i in range(6)]
    pooled = [S.week_bootstrap(df, "base", r, q=0.95, reps=300)[0] for r in rules]
    narrow = [S.week_bootstrap(df, "base", r, q=0.95, reps=300, reference="baseline")[0] for r in rules]
    assert 0.8 < float(np.mean(pooled)) < 1.2                                         # none of them does anything: no lift against the field
    assert float(np.mean(narrow)) > 1.5                                               # the old yardstick (the baseline's own lineups) called all of them better
    table = S.forward_table(df, "base", reps=200)
    assert sum(r["verdict"] == "better" for r in table["rows"]) <= 1                  # at most the odd chance clearing, not the whole grid


def test_a_rule_that_widens_the_spread_shows_a_lift():
    df = _placebo_runs({"r0": 16.0})
    est, lo, hi = S.week_bootstrap(df, "base", "r0", q=0.95, reps=300)
    assert lo > 1.0 and est > 1.2
    est1, lo1, hi1 = S.week_bootstrap(df, "base", "r0", q=0.99, reps=300)
    assert est1 > est                                                                 # fatter tails matter more the further out you go


def test_prevalence_matches_the_raw_columns():
    pool = tiny_pool(5)
    sets = HS.best_index_sets(pool, 6, min_uniques=2)
    feats = pd.DataFrame([HS.lineup_features(pool["dk_id"].iloc[s].tolist(), pool) for s in sets]).assign(season=2020, week=1)
    prev = S.prevalence(feats)
    assert prev["QB + 1 or more WR/TE teammates"] == pytest.approx((feats["stack_wr_te"] >= 1).mean())
    assert prev["at most 3 offensive players from one team"] == pytest.approx((feats["max_from_team"] <= 3).mean())
    assert ((prev >= 0) & (prev <= 1)).all() and len(prev) == len(S.FEATURE_DEFS)
    ci = S.prevalence_ci(feats, reps=50)
    assert (ci["lo"] <= ci["share"] + 1e-9).all() and (ci["share"] <= ci["hi"] + 1e-9).all()


def test_end_to_end_comparison_on_two_real_weeks():
    weeks = [(2026, 10), (2026, 11)]
    rules = {"base": Rules(min_salary=49000), "stack2+back": Rules(min_salary=49000, qb_stack=2, bring_back=1)}
    df = S.compare(rules, weeks=weeks, n=4, jitter=0.3, workers=1, with_features=True, source="fp")
    assert len(df) == 2 * 2 * 4 and set(df["rules"]) == set(rules)
    stacked = df[df["rules"] == "stack2+back"]
    assert (stacked["stack_wr_te"] >= 2).all() and (stacked["bring_back_wr_te"] >= 1).all()
    assert (df["salary"] >= 49000).all() and (df["salary"] <= 50000).all()
    again = S.compare(rules, weeks=weeks, n=4, jitter=0.3, workers=1, with_features=False, source="fp")
    assert again["score"].tolist() == df["score"].tolist()                           # the same seed gives the same lineups


def _feature_frames(n_weeks: int = 12):
    """Best-lineup features for a few identical tiny weeks, plus a forward-test frame (base and a shifted rule set) built on the same feature columns."""
    pool = tiny_pool(5)
    sets = HS.best_index_sets(pool, 6, min_uniques=2)
    base = pd.DataFrame([HS.lineup_features(pool["dk_id"].iloc[s].tolist(), pool) for s in sets])
    rng = np.random.default_rng(2)
    hind, fwd = [], []
    for w in range(1, n_weeks + 1):
        hind.append(base.assign(season=2020, week=w, rank=range(1, len(base) + 1)))
        for name, shift in (("base", 0.0), ("stack2+back", 8.0)):
            fwd.append(base.assign(season=2020, week=w, rules=name, k=range(len(base)), score=rng.normal(120 + shift, 12, len(base)), proj=130.0))
    return pd.concat(hind, ignore_index=True), pd.concat(fwd, ignore_index=True)


def test_verdict_needs_the_interval_to_clear_one():
    assert S.verdict(1.4, 1.1, 1.8) == "better" and S.verdict(0.7, 0.5, 0.95) == "worse" and S.verdict(1.2, 0.8, 1.7) == "unclear"


def test_forward_and_hindsight_tables():
    hind, fwd = _feature_frames()
    ft = S.forward_table(fwd, "base", reps=100)
    rows = {r["key"]: r for r in ft["rows"]}
    assert ft["n_weeks"] == 12 and ft["lineups_per_rule"] == len(fwd) // 2
    assert rows["base"]["lift"] == rows["base"]["lo"] == rows["base"]["hi"] == 1.0
    assert rows["stack2+back"]["mean"] > rows["base"]["mean"] + 4 and rows["stack2+back"]["rules"] == S.RULE_LABELS["stack2+back"]
    assert rows["stack2+back"]["lo"] <= rows["stack2+back"]["lift"] <= rows["stack2+back"]["hi"]
    ht = S.hindsight_table(hind, fwd[fwd["rules"] == "base"], reps=50)
    assert [r["trait"] for r in ht] == list(S.FEATURE_DEFS)
    one = next(r for r in ht if r["trait"] == "QB + 1 or more WR/TE teammates")
    assert one["rank1_lo"] <= one["rank1"] <= one["rank1_hi"] and 0 <= one["typical"] <= 1


def test_build_evidence_is_json_ready_and_reads_the_forward_results(tmp_path, monkeypatch):
    import json
    hind, fwd = _feature_frames()
    monkeypatch.setattr(S, "EVIDENCE_FILE", tmp_path / "ev.json")
    ev = S.build_evidence(fwd, fp=fwd, hind=hind, reps=60)
    assert json.loads((tmp_path / "ev.json").read_text(encoding="utf-8"))["weeks"] == ev["weeks"]
    assert ev["weeks"] == {"hindsight": 12, "forward": 12, "fp": 12}
    assert set(ev) >= {"built", "weeks", "hindsight", "forward", "forward_fp", "spend", "group_titles", "summary", "caveats"}
    assert {r["slot"] for r in ev["spend"]["slots"]} >= {"QB", "RBs (all)", "WRs (all)", "TEs (all)", "DST"} and "tiers" not in ev["spend"]
    assert {r["key"]: r["group"] for r in ev["forward"]["rows"]}["stack2+back"] == "structure"
    assert any("5%" in line for line in ev["summary"]) and ev["caveats"]


def test_every_rule_set_belongs_to_exactly_one_group():
    keys = [k for ks in S.RULE_GROUPS.values() for k in ks]
    assert len(keys) == len(set(keys))
    assert set(keys) == {k for grid in S.GRIDS.values() for k in grid} - {"base"}
    assert S.rule_group("stack2+back") == "structure" and S.rule_group("qb>=7000") == "spend" and S.rule_group("dst_fav") == "environment" and S.rule_group("nope") == ""


def test_combine_runs_shares_one_baseline_and_rejects_a_different_one():
    _, fwd = _feature_frames()
    a = fwd[fwd["rules"].isin(["base", "stack2+back"])]
    b = fwd[fwd["rules"] == "base"].assign(rules="base")
    extra = fwd[fwd["rules"] == "stack2+back"].assign(rules="qb<=6500", score=lambda d: d["score"] + 3.0)
    run2 = pd.concat([b, extra], ignore_index=True)
    both = S.combine_runs(a, run2)
    assert set(both["rules"]) == {"base", "stack2+back", "qb<=6500"} and (both["rules"] == "base").sum() == (a["rules"] == "base").sum()      # the second baseline is not stacked a second time
    again = S.combine_runs(a, pd.concat([b, fwd[fwd["rules"] == "stack2+back"]], ignore_index=True))
    assert (again["rules"] == "stack2+back").sum() == (a["rules"] == "stack2+back").sum()                                                     # a rule set already present is kept once
    shifted = run2.assign(score=lambda d: d["score"] + (d["rules"] == "base") * 5.0)
    with pytest.raises(ValueError, match="share a baseline"):
        S.combine_runs(a, shifted)


def test_saved_runs_are_stacked_by_grid_and_combined_under_one_baseline(tmp_path, monkeypatch):
    from dfs import history as H
    monkeypatch.setattr(H, "HISTORY_DIR", tmp_path)
    assert S.run_file("main").name == "mm_forward_main_local.csv" and S.run_file("env", "fp", "1/2").name == "mm_forward_fpenv1of2_local.csv"
    assert S.read_runs("salary") is None and S.read_runs("fp") is None

    _, fwd = _feature_frames()
    env = fwd.assign(rules=fwd["rules"].map({"base": "base", "stack2+back": "qb_underdog"}))                       # an environment run: the baseline and one rule set, with the feature columns
    first_half = env["week"] <= 6
    env[first_half].to_csv(S.run_file("env", "salary", "0/2"), index=False)                                         # the two halves of a sharded run
    env[~first_half].to_csv(S.run_file("env", "salary", "1/2"), index=False)
    old = fwd[["season", "week", "rules", "k", "score", "proj", "salary"]]                                           # an older run made before the feature columns existed
    old.to_csv(S.run_file("main"), index=False)
    fwd[fwd["rules"] == "base"].assign(score=lambda d: d["score"] + 40.0).to_csv(tmp_path / "mm_forward_check15_local.csv", index=False)   # a robustness rerun: other baseline, never part of the evidence

    got = S.read_runs("salary")
    assert set(got["rules"]) == {"base", "qb_underdog", "stack2+back"}
    assert (got["rules"] == "base").sum() == (fwd["rules"] == "base").sum()                                          # shards stacked once, the baseline not repeated by the second grid
    assert "stack_wr_te" in got.columns and got[got["rules"] == "base"]["stack_wr_te"].notna().all()                 # the baseline is the one that carries the feature columns
    assert S.read_runs("fp") is None
