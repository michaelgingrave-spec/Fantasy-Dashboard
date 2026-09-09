import pytest

from dfs.config import SALARY_CAP, OptimizerConfig
from dfs.optimizer import OptimizerError, optimize, to_dk_upload


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
