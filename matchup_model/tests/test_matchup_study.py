"""The matchup study's machinery: labels that cannot be gamed by level or skew, out-of-fold discipline, and intervals that find a planted effect but not noise."""
import numpy as np
import pandas as pd
import pytest

from matchup_model.opp import matchup_study as S

pytest.importorskip("sklearn")
FAST = {"auc": 80, "tails": 120, "slope": 100, "logit": 100}


def cheap_median():
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(loss="quantile", alpha=0.5, n_estimators=60, max_depth=2, random_state=0)


@pytest.fixture(autouse=True)
def fast_bootstrap(monkeypatch):
    monkeypatch.setattr(S, "REPS", dict(FAST))


def synth(n: int = 4000, seed: int = 0, planted: bool = True, players: int = 400, power: float = 1.15) -> pd.DataFrame:
    """A WR-like table: his projection `b_tgt`, a profile trend `p_trend`, a mismatch `mm_x`, noise defense features -- outcome driven by the first three only if planted."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"season": rng.choice([2022, 2023, 2024], n), "week": rng.integers(5, 18, n), "nk": [f"p{i}" for i in rng.integers(0, players, n)],
                      "position": rng.choice(["WR", "TE"], n), "b_tgt": rng.uniform(2, 10, n), "p_trend": rng.normal(size=n), "mm_x": rng.normal(size=n)})
    d["b_rec"], d["b_rec_yds"], d["is_te"] = d.b_tgt * 0.65, d.b_tgt * 8, (d.position == "TE").astype(float)
    for i in range(6):
        d[f"d_noise{i}_dev"] = rng.normal(size=n)
    shift = 0.30 * d.p_trend + 0.30 * d.mm_x if planted else 0.0
    d["targets"] = d.b_tgt ** power * np.exp(shift + rng.normal(0, 0.5, n))       # skewed, and its median is NOT the projection
    return d


BASE = ["b_tgt", "b_rec", "b_rec_yds", "is_te"]
TARGETS = {"targets": S.Target("targets", BASE)}
SEASONS = (2022, 2023, 2024)


# ── labels ──────────────────────────────────────────────────────────────────────────────────────────────
def test_centred_labels_are_balanced_at_every_level_where_beating_the_projection_is_not():
    d = synth(planted=False)
    rng = np.random.default_rng(1)
    d["targets"] = d.b_tgt ** 1.8 / 4 * np.exp(rng.normal(0, 0.4, len(d)))        # below the projection for small b_tgt, above it for large: skewed and miscalibrated
    f = S.eligible(d, TARGETS["targets"])
    y, _ = S.centred_labels(f, TARGETS["targets"], SEASONS, cheap_median)
    tercile = pd.qcut(f.b_tgt, 3, labels=False)
    centred = y.groupby(tercile).mean()
    naive = (f.targets > f.b_tgt).astype(int).groupby(tercile).mean()            # 'beat the projection': driven by the level
    assert centred.between(0.45, 0.55).all()
    assert naive.max() - naive.min() > 0.5


def test_each_label_comes_from_a_median_fit_without_that_season():
    d = synth(planted=False)
    f = S.eligible(d, TARGETS["targets"])
    seen = []

    class Spy:
        def __init__(self):
            self.m = cheap_median()

        def fit(self, X, y):
            seen.append(len(X))
            self.m.fit(X, y)
            return self

        def predict(self, X):
            return self.m.predict(X)

    _, ytr = S.centred_labels(f, TARGETS["targets"], SEASONS, Spy)
    assert seen == [int((f.season != s).sum()) for s in SEASONS]
    assert all(len(ytr[s]) == (f.season != s).sum() and not (ytr[s].index.isin(f[f.season == s].index)).any() for s in SEASONS)


def _oof_frame(seed: int = 1, n: int = 3000):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({"season": rng.choice(SEASONS, n), "nk": [f"p{i}" for i in range(n)]})
    y = pd.Series(rng.integers(0, 2, n), index=f.index)
    f["real"] = y * 0.8 + rng.normal(size=n)                                      # a modest feature that holds in every season
    return f, y, {s: y[f.season != s] for s in SEASONS}


def test_out_of_fold_predictions_never_come_from_a_model_trained_on_that_season(monkeypatch):
    f, _, ytr = _oof_frame()
    fits = []

    class Spy:
        def fit(self, X, y):
            fits.append((set(X.index), len(y)))
            return self

        def predict_proba(self, X):
            return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])
    monkeypatch.setattr(S, "logit_clf", Spy)
    S.out_of_fold(f, ["real"], "logit", ytr, SEASONS)
    assert len(fits) == len(SEASONS)
    for s, (trained_on, n) in zip(SEASONS, fits):
        assert not trained_on & set(f.index[f.season == s]) and len(trained_on) == n == int((f.season != s).sum())


def test_out_of_fold_scores_carry_a_feature_that_holds_across_seasons():
    f, y, ytr = _oof_frame()
    assert S.auc(y, S.out_of_fold(f, ["real"], "logit", ytr, SEASONS)) > 0.65


# ── intervals and tests ─────────────────────────────────────────────────────────────────────────────────
def test_cluster_weights_resample_whole_clusters():
    groups = np.repeat(np.arange(50), 4)
    gid, W = S.cluster_weights(groups, reps=30, seed=0)
    assert W.shape == (30, 50) and (W.sum(axis=1) == 50).all()
    per_row = W[:, gid]
    assert (per_row.reshape(30, 50, 4) == per_row.reshape(30, 50, 4)[:, :, :1]).all()      # every row of a cluster carries its cluster's weight


def test_benjamini_hochberg_matches_a_hand_computed_example_and_never_lowers_a_p_value():
    q = S.benjamini_hochberg([0.01, 0.04, 0.03, 0.5])
    assert q == pytest.approx([0.04, 0.16 / 3, 0.16 / 3, 0.5])
    p = np.random.default_rng(2).uniform(size=200)
    q = S.benjamini_hochberg(p)
    assert (q >= p - 1e-12).all() and (q <= 1).all()
    assert (np.diff(q[np.argsort(p)]) >= -1e-12).all()                  # monotone in p
    assert S.benjamini_hochberg([1.0, 1.0]).tolist() == [1.0, 1.0]


def test_clustered_slope_recovers_a_planted_slope_and_spans_zero_for_noise():
    rng = np.random.default_rng(3)
    n = 3000
    groups = np.repeat(np.arange(n // 3), 3)
    x = rng.normal(size=n)
    b, lo, hi = S.clustered_slope(x, 2.0 * x + rng.normal(size=n), groups, reps=300)
    assert b == pytest.approx(2.0, abs=0.1) and lo < 2.0 < hi
    b, lo, hi = S.clustered_slope(x, rng.normal(size=n), groups, reps=300)
    assert lo < 0 < hi


def test_clustered_slope_is_wider_when_rows_within_a_cluster_move_together():
    rng = np.random.default_rng(4)
    n_clusters, per = 200, 5
    groups = np.repeat(np.arange(n_clusters), per)
    shared = np.repeat(rng.normal(size=n_clusters), per)
    x = shared + 0.3 * rng.normal(size=n_clusters * per)
    y = shared + 0.3 * rng.normal(size=n_clusters * per)           # pure cluster-level co-movement: effectively 200 observations, not 1000
    _, lo_c, hi_c = S.clustered_slope(x, y, groups, reps=300)
    _, lo_i, hi_i = S.clustered_slope(x, y, np.arange(len(x)), reps=300)
    assert hi_c - lo_c > hi_i - lo_i


def test_delta_auc_clears_zero_for_a_better_predictor_and_not_for_a_noise_one():
    rng = np.random.default_rng(5)
    n = 3000
    y = rng.integers(0, 2, n)
    weak, better, noise = rng.normal(size=n) + 0.2 * y, rng.normal(size=n) + 0.7 * y, rng.normal(size=n)
    groups = np.arange(n) // 2
    d, lo, hi = S.delta_auc(y, weak, better, groups, reps=100)
    assert lo > 0 and d > 0.05
    d, lo, hi = S.delta_auc(y, weak, noise, groups, reps=100)
    assert hi < 0.02 and not lo > 0


def test_tail_gaps_find_a_planted_feature_and_hold_the_false_discovery_rate_on_noise():
    rng = np.random.default_rng(6)
    n = 4000
    f = pd.DataFrame({"season": rng.choice(SEASONS, n), "nk": [f"p{i}" for i in rng.integers(0, 500, n)]})
    f["signal"] = rng.normal(size=n)
    for i in range(40):
        f[f"noise{i}"] = rng.normal(size=n)
    y = (rng.uniform(size=n) < 0.5 + 0.12 * np.clip(f.signal, -1.5, 1.5) / 1.5).astype(int)
    t = S.tail_gaps(f, y, ["signal"] + [f"noise{i}" for i in range(40)], by=None, reps=200)
    t["q"] = S.benjamini_hochberg(t.p.to_numpy())
    sig = t[t.feature == "signal"].iloc[0]
    assert sig.gap > 15 and sig.q < 0.01
    assert t[t.feature != "signal"].q.min() > 0.05


def test_tail_gaps_leave_out_rows_with_a_missing_feature_and_skip_thin_tails():
    rng = np.random.default_rng(7)
    n = 1000
    f = pd.DataFrame({"season": 2023, "nk": [f"p{i}" for i in range(n)], "x": rng.normal(size=n)})
    f.loc[f.index[:900], "x"] = np.nan                               # 100 usable rows: each decile holds ~10, under the minimum
    assert S.tail_gaps(f, rng.integers(0, 2, n), ["x"], by=None, reps=50).empty


# ── the nested study end to end on data where the answer is known ───────────────────────────────────────
def test_nested_study_credits_the_profile_and_the_mismatch_it_was_given_and_not_the_noise_defense_features():
    d = synth()
    res = S.nested_study(d, TARGETS, BASE, SEASONS, kinds=("logit",), median=cheap_median)
    steps = res["steps"].set_index("step")
    assert steps.loc["B0->B1", "lo"] > 0                      # his own profile (the planted p_trend) is real information beyond the level
    assert steps.loc["B1->B3", "lo"] > 0                      # and so is the planted mismatch
    assert steps.loc["B1->B2", "hi"] < steps.loc["B1->B3", "lo"] + 0.02
    t = res["tails"]
    assert t[t.feature == "mm_x"].iloc[0].q < 0.01
    assert t[t.feature.str.startswith("d_noise")].q.min() > 0.05


def test_a_short_pre_specified_list_finds_a_planted_mismatch_that_the_full_set_dilutes_and_not_a_noise_list():
    d = synth()
    res = S.nested_study(d, TARGETS, BASE, SEASONS, kinds=("logit",), median=cheap_median, core=["mm_x", "p_trend"])
    steps = res["steps"].set_index("step")
    assert steps.loc["B1->B4", "lo"] > 0 and "B4" in res["auc"].columns
    noise = S.nested_study(d, TARGETS, BASE, SEASONS, kinds=("logit",), median=cheap_median, core=["d_noise0_dev", "d_noise1_dev"])
    assert noise["steps"].set_index("step").loc["B1->B4", "lo"] <= 0.002
    assert "B4" not in S.nested_study(d, TARGETS, BASE, SEASONS, kinds=("logit",), median=cheap_median)["auc"].columns        # no list, no B4


def test_route_type_sets_take_the_short_list_for_b5_and_every_route_feature_for_b6():
    d = synth()
    d["rm_planted"], d["rp_mix_go"], d["rd_go_yprr_rel"], d["rp_n_prior"] = d.mm_x, np.random.default_rng(3).normal(size=len(d)), np.random.default_rng(4).normal(size=len(d)), 5
    sets = S.feature_sets(d, BASE, BASE, route_core=["rm_planted"])
    assert sets["B5"] == sets["B1"] + ["rm_planted"] and set(sets["B6"]) == set(sets["B1"]) | {"rm_planted", "rp_mix_go", "rd_go_yprr_rel"}      # counts are not features
    assert "B5" not in S.feature_sets(d, BASE, BASE)
    res = S.nested_study(d, TARGETS, BASE, SEASONS, kinds=("logit",), median=cheap_median, route_core=["rm_planted"])
    steps = res["steps"].set_index("step")
    assert steps.loc["B1->B5", "lo"] > 0 and "B1->B6" in steps.index and "B1->B4" not in steps.index
    assert (res["tails"].feature == "rm_planted").any()                                                                 # route mismatches are tail-tested too


def test_nested_study_finds_nothing_when_nothing_is_planted():
    d = synth(planted=False, seed=11)
    res = S.nested_study(d, TARGETS, BASE, SEASONS, kinds=("logit",), median=cheap_median)
    steps = res["steps"].set_index("step")
    assert steps.loc["B1->B2", "lo"] <= 0 and steps.loc["B1->B3", "lo"] <= 0
    assert res["tails"].q.min() > 0.05


# ── RB index, market test, caching, report ──────────────────────────────────────────────────────────────
def test_run_defense_index_is_standardised_on_the_reference_and_points_toward_soft_run_defenses():
    rng = np.random.default_rng(8)
    ref = pd.DataFrame({"d_ypc_rel": rng.normal(0, 0.1, 500), "d_succ_dev": rng.normal(0, 0.05, 500), "d_stuff_dev": rng.normal(0, 0.03, 500)})
    fit = S.fit_run_index(ref)
    idx = S.run_index(ref, fit)
    assert idx.mean() == pytest.approx(0, abs=1e-9) and idx.std() == pytest.approx(1, abs=0.01)
    soft = pd.DataFrame({"d_ypc_rel": [0.2], "d_succ_dev": [0.1], "d_stuff_dev": [-0.06]})
    tough = pd.DataFrame({"d_ypc_rel": [-0.2], "d_succ_dev": [-0.1], "d_stuff_dev": [0.06]})
    assert S.run_index(soft, fit).iloc[0] > 2 > -2 > S.run_index(tough, fit).iloc[0]
    assert S.run_index(pd.DataFrame({"d_ypc_rel": [0.0], "d_succ_dev": [0.0], "d_stuff_dev": [0.1]}), fit).iloc[0] < 0      # more stuffs = tougher


def _lines(n: int = 1600, seed: int = 9, edge: float = 0.0, priced: float = 0.0) -> pd.DataFrame:
    """Graded lines: the result relative to the line (SDs of the stat) depends on `edge` x the signal, and the line sits `priced` x the signal above OUR raw projection."""
    rng = np.random.default_rng(seed)
    sig = rng.normal(size=n)
    res = edge * sig + rng.normal(size=n)
    line, sigma = 50.0, 10.0
    return pd.DataFrame({"player": [f"p{i}" for i in rng.integers(0, 400, n)], "p_mkt_over": rng.uniform(0.4, 0.6, n), "z": rng.normal(size=n) * 0.3,
                         "over": (res > 0).astype(float), "actual": line + sigma * res, "line": line, "sigma": sigma, "sig": sig,
                         "raw": line - sigma * (priced * sig + rng.normal(0, 0.3, n)), "week": rng.integers(1, 6, n)})


def test_signal_vs_market_splits_a_signals_worth_into_what_the_book_priced_and_what_is_left():
    r = S.signal_vs_market(_lines(edge=0.4, priced=0.3), "sig", reps=100)
    assert r["coef"][1] > 0 and r["pts"][0] > 5                                   # predicts the side beyond the market's own price
    assert r["replication"][0] == pytest.approx(0.7, abs=0.1)                     # result minus OUR projection = what the book moved + what is left
    assert r["book_moves"][0] == pytest.approx(0.3, abs=0.05) and r["book_moves"][1] > 0.25 and r["corr"] > 0.5
    assert r["result_vs_line"][0] == pytest.approx(0.4, abs=0.1) and r["result_vs_line"][1] > 0.25 and r["weeks"] == "1-5"
    assert r["result_vs_line"][0] == pytest.approx(r["replication"][0] - r["book_moves"][0], abs=1e-9)
    r = S.signal_vs_market(_lines(seed=10, priced=0.3), "sig", reps=100)          # fully priced: the book moved all of it, nothing is left
    assert abs(r["coef"][0]) < 0.25 and abs(r["result_vs_line"][0]) < 0.12 and r["book_moves"][0] == pytest.approx(0.3, abs=0.05)
    assert S.signal_vs_market(_lines(n=30), "sig", reps=50) == {"n": 30}


def test_baselines_are_cached_per_season_and_invalidated_when_a_shipped_constant_changes(tmp_path, monkeypatch):
    calls = []

    def universe(positions, seasons):
        return pd.DataFrame([{"season": s, "week": 5, "nk": f"{p}{s}", "position": p, "team": "KC", "opp": "LV"} for s in seasons for p in positions])

    def run(chunks):
        calls.append(sorted({int(c.season.iloc[0]) for c in chunks if len(c)}))
        return pd.concat([c.assign(b_tgt=1.0, b_rec=1.0, b_rec_yds=1.0, b_rush_att=np.nan, b_rush_yds=np.nan) for c in chunks], ignore_index=True)
    monkeypatch.setattr(S, "OUT", tmp_path)
    monkeypatch.setattr(S.RC, "universe_rows", universe)
    monkeypatch.setattr(S, "_run_chunks", run)
    assert sorted(S.baselines("receiving", [2023, 2024]).season.unique()) == [2023, 2024] and calls == [[2023, 2024]]
    S.baselines("receiving", [2023, 2024])
    assert calls == [[2023, 2024]]                                              # second read comes from disk
    S.baselines("receiving", [2023, 2024, 2025])
    assert calls[-1] == [2025] and len(calls) == 2                              # only the new season is projected
    monkeypatch.setattr(S.M, "K_YPT", S.M.K_YPT + 1.0)
    S.baselines("receiving", [2023])
    assert calls[-1] == [2023] and len(calls) == 3                              # a changed constant is a different model: re-project
    S.baselines("receiving", [2023], refresh=True)
    assert len(calls) == 4


def _results() -> dict:
    """Everything `build_report` takes, from a small synthetic study (so the report path is exercised without the real data)."""
    d = synth(n=1800, seed=12)
    res = S.nested_study(d, TARGETS, BASE, SEASONS, kinds=("logit",), median=cheap_median)
    pres = pd.DataFrame({"n_low": [200, 200], "low": [0.8, 0.05], "n_high": [150, 150], "high": [1.0, 0.98], "ratio": [0.8, 0.05]}, index=pd.Index([2024, 2025], name="season"))
    inputs = {"presence": pres, "dropped": [2025]}
    usage = pd.DataFrame([{"target": "targets", "AUC": "0.600", "AUC by prior games": "3-4 games: 0.59", "beats-his-normal rate, lowest -> highest score decile": "30 40 50 60"}])
    gain = pd.DataFrame([{"stat": "targets", "RMSE baseline -> with profile": "2.1 -> 2.0", "change": "-3.0%", "by season": "2022: -2.9%"}])
    slopes = pd.DataFrame([{"test": "rush yards vs v3 baseline", "scope": "all seasons", "n": 900, "per +1 SD": 2.4, "lo": 1.1, "hi": 3.6, "unit": "yds"}])
    market = pd.DataFrame([{"signal": "demo signal", "lines": "all weeks", "n": 500, "note": "", "history": (0.1, 0.05, 0.15), **S.signal_vs_market(_lines(), "sig", 60)},
                           {"signal": "demo signal", "lines": "forward (week 6+)", "n": 12, "note": "", "history": (0.1, 0.05, 0.15)}])
    return {"rec": inputs, "rush": inputs, "wr": {k: v for k, v in res.items() if k != "frames"}, "rb": {k: v for k, v in res.items() if k != "frames"}, "gain": gain,
            "usage_wr": usage, "usage_rb": usage, "slopes": slopes, "market": market, "n_rows": {"WR/TE": "1800", "RB": "900"},
            "history": pd.DataFrame([{"signal": "demo signal", "season": y, "n": 900, "history": (0.1 + i / 100, 0.05, 0.15)} for i, y in enumerate((2023, 2024, 2025))])}


def test_the_report_renders_from_study_outputs_and_quotes_the_dated_verdict(tmp_path, monkeypatch):
    verdict = tmp_path / "verdict.md"
    verdict.write_text("## Verdict (demo date)\n\nNothing beats the price.\n", encoding="utf-8")
    monkeypatch.setattr(S, "VERDICT", verdict)
    txt = S.build_report(**_results())
    for needle in ("Verdict (demo date)", "Nothing beats the price.", "Which Data Suite exports can be trusted", "FILTERED - dropped", "AUC increments", "B1->B3",
                   "extreme-decile tests", "pre-specified run-defense", "mismatch features themselves", "demo signal", "too few lines", "forward (week 6+)",
                   "history: result minus OUR projection", "stable across seasons", "Forward-test registry", "Caveats"):
        assert needle in txt
    monkeypatch.setattr(S, "VERDICT", tmp_path / "missing.md")
    assert "Verdict" not in S.build_report(**_results())                # no verdict file, no section


def test_render_forward_and_quick_modes_use_the_cached_run_and_never_overwrite_the_report_with_a_smoke_test(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "RESULTS", tmp_path / "run.pkl")
    monkeypatch.setattr(S, "REPORT", tmp_path / "report.md")
    monkeypatch.setattr(S, "VERDICT", tmp_path / "none.md")
    with pytest.raises(SystemExit):
        S.main(["--render"])                                              # nothing cached yet
    pd.to_pickle(_results(), S.RESULTS)
    S.main(["--render"])
    assert "Forward-test registry" in (tmp_path / "report.md").read_text(encoding="utf-8")
    fresh = pd.DataFrame([{"signal": "fresh signal", "lines": "forward (week 6+)", "n": 5, "note": "", "history": (0.2, 0.1, 0.3)}])
    monkeypatch.setattr(S, "datasets", lambda refresh=False: {})
    monkeypatch.setattr(S, "market_section", lambda ds: (fresh, pd.DataFrame()))
    S.main(["--forward"])                                                 # only the 2026 section is recomputed, the rest comes from the cache
    assert "fresh signal" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert pd.read_pickle(S.RESULTS)["market"].signal.tolist() == ["fresh signal"]
    before = (tmp_path / "report.md").read_text(encoding="utf-8")
    monkeypatch.setattr(S, "market_section", lambda ds: pytest.fail("a render must not recompute"))
    S.main(["--render", "--quick"])
    assert (tmp_path / "report.md").read_text(encoding="utf-8") == before and (tmp_path / "matchup_study_report_quick.md").exists()
