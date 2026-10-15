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
    monkeypatch.setattr(S, "market_section", lambda ds: {"market": fresh, "history": pd.DataFrame()})
    S.main(["--forward"])                                                 # only the 2026 section is recomputed, the rest comes from the cache
    assert "fresh signal" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert pd.read_pickle(S.RESULTS)["market"].signal.tolist() == ["fresh signal"]
    before = (tmp_path / "report.md").read_text(encoding="utf-8")
    monkeypatch.setattr(S, "market_section", lambda ds: pytest.fail("a render must not recompute"))
    S.main(["--render", "--quick"])
    assert (tmp_path / "report.md").read_text(encoding="utf-8") == before and (tmp_path / "matchup_study_report_quick.md").exists()


# ── a full season of closing lines ──────────────────────────────────────────────────────────────────────
def test_devig_removes_the_vig_from_two_american_prices():
    assert S.devig([-110], [-110])[0] == pytest.approx(0.5)
    p = S.devig([-150, 120], [130, -140])
    assert p[0] == pytest.approx(0.6 / (0.6 + 100 / 230)) and p[1] == pytest.approx((100 / 220) / (100 / 220 + 140 / 240)) and p[0] > 0.5 > p[1]


def _signal_and_lines(n: int = 900, seed: int = 0, effect: float = 0.3, priced: float = 0.0):
    """A Signal over a synthetic 2025 history in which (result - our projection) = 34 * effect * signal + noise, and DK / FD closing lines that sit
    `priced` x signal x 34 above our projection (so the book has moved by that much with the signal)."""
    rng = np.random.default_rng(seed)
    sig = rng.normal(size=n)
    hist = pd.DataFrame({"season": 2025, "week": rng.integers(5, 19, n), "nk": [f"player{i}" for i in range(n)], "b_rec_yds": rng.uniform(40, 80, n)})
    hist["receiving_yards"] = (hist.b_rec_yds + 34.0 * (effect * sig + rng.normal(size=n))).round()
    z = pd.Series(sig, index=hist.index)
    sg = S.Signal("test signal", "wrte", "rec yds", hist, z, lambda rows: z, "receiving_yards", "b_rec_yds", "rec_yds")
    pt = (np.floor(hist.b_rec_yds + 34.0 * priced * sig) + 0.5)
    rows = []
    for h, p_ in zip(hist.itertuples(), pt):
        rows.append(dict(season=2025, week=h.week, player=f"Player{h.nk[6:]}", market="rec yds", book="draftkings", point=float(p_), over_price=-110, under_price=-110))
        rows.append(dict(season=2025, week=h.week, player=f"Player{h.nk[6:]}", market="rec yds", book="fanduel", point=float(p_) + 1.0, over_price=-115, under_price=-105))
    return sg, pd.DataFrame(rows)


def test_a_historical_frame_takes_one_books_line_per_player_week_and_drops_pushes_and_other_markets():
    sg, L = _signal_and_lines(200)
    extra = L.iloc[:4].assign(market="receptions", point=3.5)                                    # another market: ignored
    dup = L[L.book == "draftkings"].iloc[:3].assign(point=lambda x: x.point + 20, over_price=-250, under_price=190)      # an alternate-looking line far from a coin flip: not chosen
    d = S.historical_frame(pd.concat([L, extra, dup], ignore_index=True), sg, "draftkings")
    assert d.groupby(["season", "week", "player"]).size().max() == 1 and len(d) <= 200
    h = sg.hist.set_index("nk")
    row = d.iloc[0]
    assert row.line == L[(L.book == "draftkings") & (L.player == "Player" + row.player[6:]) & (L.week == row.week)].point.iloc[0]       # DraftKings' number, not FanDuel's +1
    assert row.p_mkt_over == pytest.approx(0.5) and row.over == float(row.actual > row.line) and row.z == pytest.approx((row.raw - row.line) / row.sigma)
    assert (d.actual != d.line).all()
    fd = S.historical_frame(L, sg, "fanduel")
    assert (fd.line.to_numpy() - d.set_index("player").loc[fd.player].line.to_numpy() == 1.0).all() and fd.p_mkt_over.iloc[0] != 0.5     # FanDuel's shaded prices -> a different no-vig probability


def test_market_tables_split_a_signal_into_what_the_book_priced_and_what_is_left_on_a_season_of_closing_lines():
    sg, L = _signal_and_lines(1500, seed=1, effect=0.3, priced=0.0)               # the book ignores the signal
    t, seasons = S.market_tables([sg], {}, L)
    dk = t[t.lines == "2025 closing, DraftKings"].iloc[0]
    assert dk.n > 1000 and dk.replication[0] == pytest.approx(0.3, abs=0.07) and abs(dk.book_moves[0]) < 0.05 and dk.result_vs_line[1] > 0.15
    sg2, L2 = _signal_and_lines(1500, seed=2, effect=0.3, priced=0.3)             # the book prices all of it
    dk2 = S.market_tables([sg2], {}, L2)[0].iloc[0]
    assert dk2.book_moves[0] == pytest.approx(0.3, abs=0.07) and abs(dk2.result_vs_line[0]) < 0.1 and dk2.result_vs_line[1] < 0 < dk2.result_vs_line[2]
    assert set(t.lines) == {"2025 closing, DraftKings", "2025 closing, FanDuel"} and len(seasons) == 1
    d25 = S.historical_frame(L, sg, "draftkings")
    f26 = {sg.label: d25.iloc[:100].assign(season=2026, week=lambda x: x.week % 5 + 1)}          # pretend 2026 lines: weeks 1-5 only
    t2 = S.market_tables([sg], f26, L)[0].set_index("lines")
    assert t2.loc["2026 weeks 1-5 (all)", "n"] == 100 and t2.loc["2026 forward (week 6+)", "n"] == 0 and t2.loc["2025 DK closing + 2026", "n"] == len(d25) + 100


def _closing_fixture(n: int = 3000, seed: int = 5, shade: float = 0.0, edge: float = 0.0):
    """DK closing lines for a WR/TE rec-yds history. `shade` pushes the results below the lines (a book that leans OVER); `edge` makes the result follow OUR
    projection rather than the line (a projection that beats the close)."""
    rng = np.random.default_rng(seed)
    b = rng.uniform(35, 85, n)
    line = np.floor(b + 12 * rng.normal(size=n)) + 0.5                         # the book's number differs from our projection
    truth = (1 - edge) * line + edge * b                                       # edge = 1: the outcome centres on OUR number
    hist = pd.DataFrame({"season": 2025, "week": rng.integers(5, 19, n), "nk": [f"player{i}" for i in range(n)], "b_rec_yds": b,
                         "receiving_yards": np.round(truth - shade * 34 + 34 * rng.normal(size=n)), "receptions": 3.0, "b_rec": 3.0})
    L = pd.DataFrame({"season": 2025, "week": hist.week, "player": [f"Player{i}" for i in range(n)], "market": "rec yds", "book": "draftkings", "point": line,
                      "over_price": -110, "under_price": -110})
    empty_rb = pd.DataFrame({c: [] for c in ("season", "week", "nk", "rushing_yards", "b_rush_yds", "receptions", "b_rec", "receiving_yards", "b_rec_yds")})
    return L, hist, empty_rb


def test_lean_finds_a_book_that_leans_over_and_edge_finds_a_projection_that_beats_the_close():
    L, hist, rb = _closing_fixture(shade=0.15)
    lean, edge = S.lean_and_edge(L, hist, rb, reps=120)
    r = lean.iloc[0]
    assert r.market == "WR/TE rec yds" and r.book == "draftkings" and r.gap[2] < -3 and r["under roi"][1] > 0 and r["over roi"][2] < 0
    assert abs(edge.iloc[0]["z coef"][0]) < 0.3                                                   # no projection edge was planted
    L2, hist2, rb2 = _closing_fixture(seed=6, edge=0.6)
    lean2, edge2 = S.lean_and_edge(L2, hist2, rb2, reps=120)
    assert edge2.iloc[0]["z coef"][1] > 0 and edge2.iloc[0].pts[0] > 2.4 and abs(lean2.iloc[0].gap[0]) < 4
    assert S.lean_and_edge(L.iloc[:0], hist, rb, reps=50)[0].empty and S.lean_and_edge(L.iloc[:50], hist, rb, reps=50)[1].empty            # too few lines: no row


def test_profit_and_clustered_mean_are_exact_on_a_hand_example():
    assert S.profit([-110, 120, -200], [True, True, False]).tolist() == pytest.approx([100 / 110, 1.2, -1.0])
    x = np.arange(200) / 10.0
    m, lo, hi = S.clustered_mean(x, np.arange(200) // 4, reps=200)
    assert m == pytest.approx(x.mean()) and lo < m < hi and hi - lo < 4


def test_following_a_signal_the_book_ignores_pays_and_following_one_it_priced_does_not():
    sg, L = _signal_and_lines(2500, seed=7, effect=0.4, priced=0.0)
    t = S.follow_table(L, sg, ks=(1.0,), books=("draftkings",)).set_index("side")
    assert t.loc["both", "bets"] > 600 and t.loc["both", "win"] > t.loc["both", "break-even"] + 3 and t.loc["both", "roi"][1] > 0
    assert t.loc["over", "bets"] + t.loc["under", "bets"] == t.loc["both", "bets"]
    sg2, L2 = _signal_and_lines(2500, seed=8, effect=0.4, priced=0.4)                        # the line moved all the way with the signal: nothing to follow
    t2 = S.follow_table(L2, sg2, ks=(1.0,), books=("draftkings",)).set_index("side")
    assert t2.loc["both", "roi"][0] < 2 and t2.loc["both", "roi"][1] < 0                      # about -vig, never clearly profitable
    assert S.follow_table(L.iloc[:40], sg, ks=(1.0,), books=("draftkings",)).empty               # too few bets: no row


def test_the_report_shows_the_lean_the_projection_test_and_the_follow_table_when_they_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "VERDICT", tmp_path / "none.md")
    sg, L = _signal_and_lines(1500, seed=9, effect=0.3)
    lean, edge = S.lean_and_edge(L, sg.hist, pd.DataFrame({c: [] for c in ("season", "week", "nk", "rushing_yards", "b_rush_yds", "receptions", "b_rec", "receiving_yards", "b_rec_yds")}), reps=60)
    res = {**_results(), "lean": lean, "edge": edge, "follow": S.follow_table(L, sg, ks=(1.0,), books=("draftkings",))}
    txt = S.build_report(**res)
    for needle in ("Do the closing lines lean", "blind UNDER roi", "Does our raw projection add anything", "n/a: prices are flat", "following the RB usage-trend score", "break-even", "WR/TE rec yds"):
        assert needle in txt
    assert "no historical lines pulled" in S.build_report(**_results())                         # absent data degrades to a note, not an error


# ── DraftKings vs FanDuel ────────────────────────────────────────────────────────────────────────────────
def _two_books(actuals, dk_pt=54.5, fd_pt=55.5, dk=(-110, -110), fd=(-110, -110)):
    rows, res = [], []
    for i, x in enumerate(actuals):
        for book, pt, (o, u) in (("draftkings", dk_pt, dk), ("fanduel", fd_pt, fd)):
            rows.append(dict(season=2025, week=5, event_id=f"e{i}", player=f"Guy{i}", market="rec yds", book=book, point=pt, over_price=o, under_price=u))
        res.append(dict(season=2025, week=5, nk=f"guy{i}", receiving_yards=float(x), receptions=0.0, rushing_yards=0.0))
    return pd.DataFrame(rows), pd.DataFrame(res)


def test_cross_book_legs_and_the_middle_pay_exactly_what_the_prices_say():
    L, res = _two_books([55.0] * 20 + [70.0] * 20)             # 20 results land BETWEEN 54.5 and 55.5, 20 go over both numbers
    c = S.cross_book(S.book_pairs(L, res), reps=60)
    legs = c["legs"].set_index("leg")
    w = 100 / 110
    assert legs.loc["OVER at the lower number", "roi"][0] == pytest.approx(100 * w) and legs.loc["OVER at the higher number", "roi"][0] == pytest.approx(100 * (20 * w - 20) / 40)
    assert legs.loc["UNDER at the higher number", "roi"][0] == pytest.approx(100 * (20 * w - 20) / 40) and legs.loc["UNDER at the lower number", "roi"][0] == pytest.approx(-100.0)
    mid = c["middle"].iloc[0]
    assert mid["middle hit"] == pytest.approx(50.0) and mid.roi[0] == pytest.approx(100 * (20 * 2 * w + 20 * (w - 1)) / 80)
    assert c["gaps"].iloc[0]["by 1"] == pytest.approx(100.0) and c["arb"] == 0 and c["same_number"] == 0


def test_a_push_is_refunded_and_same_number_arbitrage_is_counted():
    L, res = _two_books([54.5 + 1] * 4, dk_pt=55.0, fd_pt=56.0)                       # a whole-number line: result 55.5 is not a push, 56.0 would be
    res2 = res.assign(receiving_yards=56.0)
    c = S.cross_book(S.book_pairs(L, res2), reps=40)
    assert c["legs"].set_index("leg").loc["UNDER at the higher number", "roi"][0] == pytest.approx(0.0)       # result == the higher number: refunded
    La, resa = _two_books([60.0] * 5, dk_pt=54.5, fd_pt=54.5, dk=(105, -125), fd=(-125, 105))   # over +105 at one book, under +105 at the other: both > even money
    ca = S.cross_book(S.book_pairs(La, resa), reps=40)
    assert ca["arb"] == 5 and ca["same_number"] == 5 and ca["legs"].empty


def test_the_report_shows_the_cross_book_section_and_degrades_without_it():
    L, res = _two_books([55.0] * 30 + [70.0] * 30)
    txt = S.build_report(**{**_results(), "cross": S.cross_book(S.book_pairs(L, res), reps=40)})
    for needle in ("Do DraftKings and FanDuel disagree", "UNDER at the higher number", "Same-number arbitrage", "middle roi per $ staked"):
        assert needle in txt
    assert "Do DraftKings and FanDuel disagree" in S.build_report(**_results())


# ── opening line vs late line ──────────────────────────────────────────────────────────────────────────────
def _timing_frames(n: int = 1000, seed: int = 4, open_moved: float = 0.5, added: float = 0.2, left: float = 0.1):
    """One signal's graded lines (one per week and player, like the real frames) with an opening line that holds `open_moved` SD per SD of signal (relative to our
    projection), a late line that adds `added` more, and a result that beats the late line by `left` per SD."""
    rng = np.random.default_rng(seed)
    sig, sd, raw = rng.normal(size=n), 10.0, 50.0
    opened = raw + sd * (open_moved * sig + rng.normal(0, 0.2, n))
    late = opened + sd * (added * sig + rng.normal(0, 0.2, n))
    f = pd.DataFrame({"week": 1 + np.arange(n) // 200, "nk": [f"p{i % 200}" for i in range(n)], "market": "rec yds", "signal": sig, "raw": raw, "sigma": sd,
                      "line": late, "actual": late + sd * (left * sig + rng.normal(size=n))})
    f["player"] = f.nk
    o = pd.DataFrame({"week": f.week, "nk": f.nk, "market": f.market, "line_open": opened}).drop_duplicates(["week", "nk", "market"])
    return {"demo usage score": f}, o


def test_timing_table_splits_what_the_opening_line_held_from_what_the_market_added_during_the_week():
    frames, o = _timing_frames()
    t = S.timing_table(frames, o, reps=150).iloc[0]
    assert t.n > 600 and t.weeks == "1-5"
    assert t.open_moved[0] == pytest.approx(0.5, abs=0.07) and t.close_moved[0] == pytest.approx(0.7, abs=0.07) and t.added[0] == pytest.approx(0.2, abs=0.07)
    assert t.added[1] > 0.1 and t.left_open[0] == pytest.approx(0.3, abs=0.1) and t.left_close[0] == pytest.approx(0.1, abs=0.1)
    assert t.left_open[0] == pytest.approx(t.left_close[0] + t.added[0], abs=1e-9)           # result - open = (result - late) + (late - open)
    frames, o = _timing_frames(seed=5, added=0.0, left=0.0)                                 # the market learns nothing after the open, nothing is left
    t = S.timing_table(frames, o, reps=150).iloc[0]
    assert t.added[1] < 0 < t.added[2] and abs(t.left_open[0]) < 0.12 and t.open_moved[0] == pytest.approx(t.close_moved[0], abs=0.05)


def test_timing_table_needs_enough_lines_with_both_an_open_and_a_late_line():
    frames, o = _timing_frames()
    assert S.timing_table(frames, o.iloc[:40], reps=50).empty                                # only 40 lines have an opening line
    assert S.timing_table({}, o, reps=50).empty
    f = frames["demo usage score"].assign(signal=np.nan)
    assert S.timing_table({"x": f}, o, reps=50).empty                                        # a signal that is undefined for these rows


def test_openers_are_read_by_player_key_and_a_missing_file_is_an_empty_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "DATA", tmp_path)
    assert S.load_openers().empty
    (tmp_path / "props").mkdir()
    pd.DataFrame({"week": [1, 1, 2], "player": ["Ja'Marr Chase", "Ja'Marr Chase", "A.J. Brown"], "market": ["rec yds"] * 3, "line_open": [80.5, 81.5, 64.5]}).to_csv(tmp_path / "props" / "opening_lines.csv", index=False)
    o = S.load_openers()
    assert list(o.nk) == ["jamarr chase", "aj brown"] and list(o.line_open) == [80.5, 64.5]


def test_the_report_shows_the_timing_section_and_degrades_without_it():
    frames, o = _timing_frames()
    txt = S.build_report(**{**_results(), "timing": S.timing_table(frames, o, reps=60)})
    for needle in ("priced by Wednesday, or during the week", "demo usage score", "added during the week", "left at the OPEN", "What the bet-level tests can and cannot rule out"):
        assert needle in txt
    assert "(no opening lines with a signal)" in S.build_report(**_results())


# ── FantasyPoints' own projection (H5) ─────────────────────────────────────────────────────────────────
def _fp_history(weeks=(2, 3, 4, 5), per_week: int = 400, seed: int = 8, info: float = 0.3, priced: float = 0.0, shift: float = 0.0, with_ratio: bool = True) -> pd.DataFrame:
    """Graded line_history rows: FantasyPoints' number holds `info` SD per SD of the gap beyond our projection, the book's line sits `priced` per SD toward it
    (`shift` moves the level of every gap, as a model-version change would). The book also knows something of its own (`u`, in its line AND in the truth), as a real
    line does: that is what makes the yes/no coefficient measure what is left after the line rather than what FantasyPoints knows beyond us."""
    rng = np.random.default_rng(seed)
    n, line = per_week * len(weeks), 60.5
    s0 = float(S._sigma("rec_yds", line))
    sig, u = rng.normal(size=n), rng.normal(0, 0.3, n)
    ours = 60.0
    book_line = ours + s0 * (priced * sig + u)
    actual = ours + s0 * (info * sig + u + rng.normal(size=n))
    p_over = np.clip(0.5 + rng.normal(0, 0.03, n), 0.3, 0.7)
    lean = np.where(ours > book_line, "OVER", "UNDER")
    lh = pd.DataFrame({"season": 2026, "week": np.repeat(weeks, per_week), "player": [f"pl{i % 300}" for i in range(n)], "market": "rec yds", "line": book_line,
                       "actual": actual, "our proj": ours, "lean": lean, "book %": np.round(100 * np.where(lean == "OVER", p_over, 1 - p_over)),
                       "result": np.where((actual > book_line) == (lean == "OVER"), "win", "loss"),
                       "role_ratio": np.exp(S.FP_GAP_SD * sig + shift) if with_ratio else np.nan})
    return lh


def test_the_fp_gap_is_clipped_centred_within_the_week_and_scaled_by_the_frozen_sd():
    lh = _fp_history(shift=0.15)
    lh.loc[0, "role_ratio"] = 9.0                                            # a 9x disagreement: a placeholder, not a view
    lh.loc[1, "role_ratio"] = np.nan                                         # no FantasyPoints number for this player
    d = S.fp_gap_frame(lh)
    assert len(d) < len(lh) and d.fp_dev.abs().max() < 7                     # the NaN row is dropped and the 9x ratio clipped at +/-0.693 / 0.203 = 3.4 SD
    assert d.groupby("week").fp_dev.mean().abs().max() < 1e-9                # centred within each week, so a level shift from a new model version cancels
    assert d.fp_dev.std() == pytest.approx(1.0, abs=0.08) and (d.raw == 60.0).all()
    assert S.fp_gap_frame(lh.drop(columns="role_ratio")).empty and S.fp_gap_frame(lh.iloc[:0]).empty


def test_fp_gap_table_tells_a_projection_that_knows_more_from_one_the_book_already_priced():
    lh = _fp_history(info=0.3, priced=0.0)
    t = S.fp_gap_table(lh, reps=150)
    r = t.iloc[0]
    assert r.lines.startswith("weeks 2-5") and r.n > 1400 and r.coef[1] > 0 and r.pts[0] > 5                         # nobody priced it: beyond the price and our edge
    assert r.replication[0] == pytest.approx(0.3, abs=0.07) and abs(r.book_moves[0]) < 0.06 and r.result_vs_line[0] == pytest.approx(0.3, abs=0.07)
    assert t.iloc[1].lines.endswith("rec yds only (descriptive)") and t.iloc[1].n == r.n                             # the only market in the fixture
    assert t.iloc[-1].lines.startswith("forward") and t.iloc[-1].n == 0                                               # no forward week yet: the row is there, empty
    t = S.fp_gap_table(_fp_history(seed=9, info=0.3, priced=0.3, shift=-0.1), reps=150).iloc[0]                       # the book moved all of it, level shifted
    assert t.replication[0] == pytest.approx(0.3, abs=0.07) and t.book_moves[0] == pytest.approx(0.3, abs=0.07) and abs(t.result_vs_line[0]) < 0.08 and abs(t.coef[0]) < 0.12


def test_fp_gap_table_keeps_the_weeks_already_seen_apart_from_the_forward_weeks():
    lh = pd.concat([_fp_history(), _fp_history(weeks=(6, 7), per_week=300, seed=13, info=0.0)], ignore_index=True)
    t = S.fp_gap_table(lh, reps=100)
    hist, fwd = t.iloc[0], t.iloc[-1]
    assert hist.weeks == "2-5" and fwd.weeks == "6-7" and fwd.n > 500 and hist.n > 1400
    assert fwd.replication[0] == pytest.approx(0.0, abs=0.1) and hist.replication[0] == pytest.approx(0.3, abs=0.07)   # a fixed rule applied to a different truth
    assert S.fp_gap_table(_fp_history(with_ratio=False), reps=50).empty                                                # nothing with a number: no table


def test_the_report_shows_the_fantasypoints_section_and_the_h5_rule():
    txt = S.build_report(**{**_results(), "fp": S.fp_gap_table(_fp_history(), reps=60)})
    for needle in ("know more than our number", "beyond the price and our edge", "too few lines", "H5 rule", "FP_SHIFT_CAP", "FantasyPoints history"):
        assert needle in txt
    assert "(no graded lines with a FantasyPoints number)" in S.build_report(**_results())


# ── execution: where and on which side the same leans are placed ───────────────────────────────────────────
def _placement_props(n_each: int = 100) -> pd.DataFrame:
    """Two kinds of props, one lean each. OVER leans: DraftKings has the lower NUMBER (54.5), FanDuel the better PRICE (55.5 at -105). UNDER leans: DraftKings has the
    better price (54.5 at -110), FanDuel the higher number (55.5 at -115). Every result is 55: the best number wins each bet, the best price loses each."""
    rows = []
    for i in range(2 * n_each):
        over = i < n_each
        rows.append({"season": 2025, "week": 5 + i % 14, "nk": f"p{i}", "market": "rec yds", "point_dk": 54.5, "point_fd": 55.5, "over_price_dk": -110, "under_price_dk": -110,
                     "over_price_fd": -105, "under_price_fd": -115, "actual": 55.0, "raw": 60.0 if over else 50.0})
    P = pd.DataFrame(rows)
    P["ref"] = (P.point_dk + P.point_fd) / 2
    return P


def test_execution_table_prices_the_same_leans_placed_at_the_best_price_and_at_the_best_number():
    e = S.execution_table(_placement_props(), reps=60)
    pol = e["policies"].set_index(["policy", "market"])
    a, b = pol.loc[("A  best price (the app before)", "all yardage")], pol.loc[("B  best number (the app now)", "all yardage")]
    assert a.bets == b.bets == 200 and a.roi[0] == pytest.approx(-100.0) and a.win == 0.0                                   # the best price lands on the number that loses
    assert b.roi[0] == pytest.approx(100 * (0.5 * 100 / 110 + 0.5 * 100 / 115)) and b.win == 100.0                           # the best number wins both kinds
    assert pol.loc[("B  UNDER leans only", "all yardage")].roi[0] == pytest.approx(100 * 100 / 115) and pol.loc[("B  UNDER leans only", "all yardage")].bets == 100
    assert pol.loc[("A  UNDER leans only", "rec yds")].roi[0] == pytest.approx(-100.0) and pol.loc[("B  best number (the app now)", "rec yds")]["over share"] == pytest.approx(50.0)
    assert ("A  best price (the app before)", "rush yds") not in pol.index and e["props"] == 200 and e["over_lean"] == 50.0   # a market with no props is left out
    n, mean, lo, hi = e["paired"]["every lean"]
    assert n == 200 and mean == pytest.approx(100 * (0.5 * (100 / 110 + 1) + 0.5 * (100 / 115 + 1)))                       # B minus A, per bet
    bl = e["blind"].set_index(["bet", "placed", "market"])
    assert bl.loc[("always UNDER", "best number", "all yardage"), "roi"][0] == pytest.approx(100 * 100 / 115)             # FanDuel's 55.5 beats a 55 result
    assert bl.loc[("always OVER", "best number", "all yardage"), "roi"][0] == pytest.approx(100 * 100 / 110)               # DraftKings' 54.5 does too
    assert bl.loc[("always OVER", "best price", "all yardage"), "roi"][0] == pytest.approx(-100.0)                         # -105 at 55.5 loses a 55


def test_execution_frame_attaches_our_projection_and_the_consensus_number():
    L, res = _two_books([55.0] * 12)
    wr = pd.DataFrame({"season": 2025, "week": 5, "nk": [f"guy{i}" for i in range(10)], "b_rec_yds": 60.0, "b_rec": 4.0})
    rb = pd.DataFrame({"season": 2025, "week": 5, "nk": ["guy10", "guy11"], "b_rec_yds": 20.0, "b_rec": 2.0, "b_rush_yds": 70.0})
    P = S.execution_frame(L, res, wr, rb)
    assert len(P) == 12 and (P.ref == 55.0).all() and set(P.raw) == {60.0, 20.0} and set(P[P.nk.isin(["guy10", "guy11"])].raw) == {20.0}      # rec yds -> b_rec_yds, from the RB table for the RBs
    assert S.execution_frame(L, res, wr.iloc[:0], rb.iloc[:0]).empty


def test_the_report_shows_the_execution_section():
    txt = S.build_report(**{**_results(), "execution": S.execution_table(_placement_props(), reps=60)})
    for needle in ("What execution alone is worth", "B  best number (the app now)", "always UNDER", "Taking the best number instead of the best price"):
        assert needle in txt
    assert "What execution alone is worth" in S.build_report(**_results())
