import itertools

import numpy as np
import pandas as pd
import pytest

from dfs.config import SALARY_CAP, OptimizerConfig
from dfs.optimizer import OptimizerError, ideal_lineup, optimize, to_dk_upload


def _assert_valid(lu):
    p = lu.players
    assert len(p) == 9
    counts = p["pos"].value_counts().to_dict()
    assert counts.get("QB", 0) == 1
    assert counts.get("DST", 0) == 1
    assert 2 <= counts.get("RB", 0) <= 3
    assert 3 <= counts.get("WR", 0) <= 4
    assert 1 <= counts.get("TE", 0) <= 2
    assert counts.get("RB", 0) + counts.get("WR", 0) + counts.get("TE", 0) == 7
    assert lu.salary <= SALARY_CAP


def test_single_lineup_is_valid(slate_df):
    lus = optimize(slate_df, OptimizerConfig(n_lineups=1, min_salary=45000, stack_qb=False))
    assert len(lus) == 1
    _assert_valid(lus[0])


def test_locked_players_appear_in_every_lineup(slate_df):
    rb = slate_df[slate_df["pos"] == "RB"].iloc[0]["dk_id"]
    wr = slate_df[slate_df["pos"] == "WR"].iloc[0]["dk_id"]
    cfg = OptimizerConfig(n_lineups=5, min_salary=45000, stack_qb=False,
                          locks=[rb, wr], max_exposure=1.0, min_uniques=1)
    lus = optimize(slate_df, cfg)
    assert len(lus) >= 1
    for lu in lus:
        ids = set(lu.players["dk_id"])
        assert rb in ids and wr in ids
        _assert_valid(lu)


def test_excluded_players_never_appear(slate_df):
    top_wr = slate_df[slate_df["pos"] == "WR"].iloc[0]["dk_id"]
    lus = optimize(slate_df, OptimizerConfig(n_lineups=3, min_salary=45000,
                                             stack_qb=False, excludes=[top_wr], min_uniques=1))
    for lu in lus:
        assert top_wr not in set(lu.players["dk_id"])


def test_qb_stack_constraint(slate_df):
    lus = optimize(slate_df, OptimizerConfig(n_lineups=1, min_salary=45000,
                                             stack_qb=True, stack_min=1))
    lu = lus[0]
    qb = lu.players[lu.players["pos"] == "QB"].iloc[0]
    mates = lu.players[(lu.players["team"] == qb["team"]) & (lu.players["pos"].isin(["WR", "TE"]))]
    assert len(mates) >= 1


def test_multi_lineup_diversity_and_exposure(slate_df):
    cfg = OptimizerConfig(n_lineups=10, min_salary=45000, stack_qb=False,
                          min_uniques=2, max_exposure=0.6)
    lus = optimize(slate_df, cfg)
    assert len(lus) >= 5
    sigs = {tuple(sorted(lu.players["dk_id"])) for lu in lus}
    assert len(sigs) == len(lus)  # all distinct
    n = len(lus)
    counts: dict[str, int] = {}
    for lu in lus:
        for i in lu.players["dk_id"]:
            counts[i] = counts.get(i, 0) + 1
    assert max(counts.values()) <= (0.6 * n) + 1  # exposure cap (rounded up)


def test_dk_upload_has_nine_columns(slate_df):
    lus = optimize(slate_df, OptimizerConfig(n_lineups=2, min_salary=45000, stack_qb=False,
                                             min_uniques=1))
    up = to_dk_upload(lus)
    assert list(up.columns) == ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
    assert len(up) == len(lus)


def test_impossible_lock_raises(slate_df):
    qbs = slate_df[slate_df["pos"] == "QB"]["dk_id"].tolist()[:2]
    with pytest.raises(OptimizerError):
        optimize(slate_df, OptimizerConfig(locks=qbs))


def test_variance_keeps_lineups_valid_and_diverse(slate_df):
    cfg = OptimizerConfig(n_lineups=8, min_salary=45000, stack_qb=False,
                          variance=0.4, seed=1, min_uniques=1)
    lus = optimize(slate_df, cfg)
    assert len(lus) == 8
    for lu in lus:
        _assert_valid(lu)
    sigs = {tuple(sorted(lu.players["dk_id"])) for lu in lus}
    assert len(sigs) == 8
    # with real jitter the set should use more distinct players than the zero-variance set
    used_var = {i for lu in lus for i in lu.players["dk_id"]}
    base = optimize(slate_df, OptimizerConfig(n_lineups=8, min_salary=45000,
                                              stack_qb=False, min_uniques=1))
    used_base = {i for lu in base for i in lu.players["dk_id"]}
    assert len(used_var) >= len(used_base)


def test_variance_seed_is_reproducible_and_shuffles(slate_df):
    a = optimize(slate_df, OptimizerConfig(n_lineups=5, min_salary=45000, stack_qb=False,
                                           variance=0.5, seed=42))
    b = optimize(slate_df, OptimizerConfig(n_lineups=5, min_salary=45000, stack_qb=False,
                                           variance=0.5, seed=42))
    c = optimize(slate_df, OptimizerConfig(n_lineups=5, min_salary=45000, stack_qb=False,
                                           variance=0.5, seed=7))
    sig = lambda lus: [tuple(sorted(lu.players["dk_id"])) for lu in lus]
    assert sig(a) == sig(b)          # same seed -> same lineups
    assert sig(a) != sig(c)          # different seed -> different combos


def test_variance_zero_matches_default(slate_df):
    a = optimize(slate_df, OptimizerConfig(n_lineups=4, min_salary=45000, stack_qb=False,
                                           variance=0.0, seed=99))
    b = optimize(slate_df, OptimizerConfig(n_lineups=4, min_salary=45000, stack_qb=False))
    sig = lambda lus: [tuple(sorted(lu.players["dk_id"])) for lu in lus]
    assert sig(a) == sig(b)


def _two_rb_temptation_slate() -> pd.DataFrame:
    """A tiny synthetic slate where team A's two RBs are clearly the two best RBs
    overall, so an unconstrained optimizer wants both — the scenario no_rb_pair exists
    to forbid (same-team RBs split touches and rarely both pay off)."""
    rows = [
        ("qbA", "QB A", "QB", "A", "B", 6000, 20.0),
        ("qbB", "QB B", "QB", "B", "A", 6000, 18.0),
        ("rbA1", "RB A1", "RB", "A", "B", 7000, 18.0),
        ("rbA2", "RB A2", "RB", "A", "B", 6500, 17.0),
        ("rbB1", "RB B1", "RB", "B", "A", 5000, 12.0),
        ("rbC1", "RB C1", "RB", "C", "D", 4500, 11.0),
        ("wrA1", "WR A1", "WR", "A", "B", 6000, 15.0),
        ("wrB1", "WR B1", "WR", "B", "A", 5500, 14.0),
        ("wrC1", "WR C1", "WR", "C", "D", 5000, 13.0),
        ("wrD1", "WR D1", "WR", "D", "C", 4500, 12.0),
        ("teA1", "TE A1", "TE", "A", "B", 4000, 10.0),
        ("teB1", "TE B1", "TE", "B", "A", 3500, 9.0),
        ("dstA", "DST A", "DST", "A", "B", 3000, 8.0),
        ("dstE", "DST E", "DST", "E", "F", 2500, 7.0),
    ]
    df = pd.DataFrame(rows, columns=["dk_id", "name", "pos", "team", "opp", "salary", "proj"])
    df["playable"] = True
    return df


def test_no_rb_pair_forbids_same_team_rbs():
    slate = _two_rb_temptation_slate()
    # Precondition: without the constraint, the optimal lineup really does pair team A's
    # two RBs (otherwise this test wouldn't be exercising anything).
    unconstrained = optimize(slate, OptimizerConfig(n_lineups=1, min_salary=0, stack_qb=False))[0]
    rbs = unconstrained.players[unconstrained.players["pos"] == "RB"]
    assert rbs["team"].value_counts().max() == 2

    constrained = optimize(
        slate, OptimizerConfig(n_lineups=1, min_salary=0, stack_qb=False, no_rb_pair=True)
    )[0]
    rbs = constrained.players[constrained.players["pos"] == "RB"]
    assert rbs["team"].value_counts().max() <= 1
    # still picks team A's better RB over sitting it entirely
    assert "rbA1" in set(constrained.players["dk_id"])


def test_game_stack_min_concentrates_in_one_game(slate_df):
    # DET and NO play each other in the fixture slate; use them as the target game.
    cfg = OptimizerConfig(n_lineups=1, min_salary=45000, stack_qb=False,
                          game_stack_teams=("DET", "NO"), game_stack_min=4)
    lu = optimize(slate_df, cfg)[0]
    in_game = lu.players[lu.players["team"].isin(["DET", "NO"]) & (lu.players["pos"] != "DST")]
    assert len(in_game) >= 4


def test_game_stack_min_zero_is_a_no_op(slate_df):
    """game_stack_teams set but game_stack_min=0 (the default) must not constrain anything."""
    with_zero = optimize(slate_df, OptimizerConfig(
        n_lineups=1, min_salary=45000, stack_qb=False, game_stack_teams=("DET", "NO")))
    without = optimize(slate_df, OptimizerConfig(n_lineups=1, min_salary=45000, stack_qb=False))
    sig = lambda lus: [tuple(sorted(lu.players["dk_id"])) for lu in lus]
    assert sig(with_zero) == sig(without)


# ── order: best projection first ───────────────────────────────────────────────────────────────────────────
def test_lineups_come_back_best_projection_first(slate_df):
    """Variance jitters each solve and the exposure cap / uniqueness rule shape the later ones, so the solver's own order
    goes up and down (this seed built them as 152.8, 154.6, 157.9, 154.1, ...). The list is sorted, best projection first."""
    cfg = OptimizerConfig(n_lineups=8, min_salary=45000, stack_qb=False, variance=0.4, seed=1, min_uniques=1)
    lus = optimize(slate_df, cfg)
    projs = [lu.proj for lu in lus]
    assert len(lus) == 8
    assert projs == sorted(projs, reverse=True)
    assert len(to_dk_upload(lus)) == 8                                   # the upload file follows the same order


# ── the ideal lineup ───────────────────────────────────────────────────────────────────────────────────────
def test_ideal_lineup_is_valid_and_no_built_lineup_beats_it(slate_df):
    ideal = ideal_lineup(slate_df)
    _assert_valid(ideal)
    for cfg in (OptimizerConfig(n_lineups=1, min_salary=45000, stack_qb=False),
                OptimizerConfig(n_lineups=6, variance=0.4, seed=3),                   # stack, salary floor, team cap, DST rule, exposure cap
                OptimizerConfig(n_lineups=4, lean=1.0, bring_back=True, min_salary=45000)):
        for lu in optimize(slate_df, cfg):
            assert lu.proj <= ideal.proj + 1e-9


def test_ideal_lineup_matches_a_brute_force_search():
    """On a pool small enough to enumerate every valid lineup, the ideal one is exactly the best of them -- with a cap that binds."""
    rng = np.random.default_rng(5)
    rows = []
    # salary climbs with projection (plus noise), as DraftKings prices it, so the best players cannot all fit under the cap
    for pos, n, p_lo, p_hi, base, slope in (("QB", 3, 14, 26, 4500, 130), ("RB", 5, 5, 24, 2500, 330), ("WR", 6, 4, 22, 2500, 340),
                                            ("TE", 3, 3, 16, 2500, 300), ("DST", 3, 3, 12, 2000, 110)):
        for k in range(n):
            proj = round(float(rng.uniform(p_lo, p_hi)), 1)
            rows.append({"dk_id": f"{pos}{k}", "name": f"{pos} {k}", "pos": pos, "team": "ABC"[k % 3], "opp": "", "proj": proj,
                         "salary": int(round((base + slope * proj + rng.integers(-500, 500)) / 100)) * 100})
    slate = pd.DataFrame(rows)
    by_pos = {p: slate[slate["pos"] == p].to_dict("records") for p in ("QB", "RB", "WR", "TE", "DST")}

    def best(cap):
        top = -1.0
        for qb in by_pos["QB"]:
            for dst in by_pos["DST"]:
                for skill in itertools.combinations(by_pos["RB"] + by_pos["WR"] + by_pos["TE"], 7):
                    pos = [s["pos"] for s in skill]
                    if not (2 <= pos.count("RB") <= 3 and 3 <= pos.count("WR") <= 4 and 1 <= pos.count("TE") <= 2):
                        continue
                    if qb["salary"] + dst["salary"] + sum(s["salary"] for s in skill) > cap:
                        continue
                    top = max(top, qb["proj"] + dst["proj"] + sum(s["proj"] for s in skill))
        return top

    assert best(SALARY_CAP) < best(10**9), "the cap should bind in this pool, or the check proves nothing"
    ideal = ideal_lineup(slate)
    _assert_valid(ideal)
    assert ideal.proj == pytest.approx(best(SALARY_CAP), abs=0.01)           # Lineup.proj is rounded to 2 decimals


def test_ideal_lineup_leaves_out_excluded_players(slate_df):
    top = ideal_lineup(slate_df)
    gone = top.players.iloc[0]["dk_id"]
    again = ideal_lineup(slate_df, excludes=[gone])
    assert gone not in set(again.players["dk_id"])
    assert again.proj <= top.proj
    _assert_valid(again)
