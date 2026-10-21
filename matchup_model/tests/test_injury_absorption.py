"""matchup_model.opp.injury_absorption on a synthetic league whose true absorption is known: the observations, the new / ongoing split, kappa, and the report."""
import numpy as np
import pandas as pd
import pytest

from dfs.names import normalize_name
from matchup_model.opp import injuries as I
from matchup_model.opp import injury_absorption as IA

KAPPA_NEW, KAPPA_ONGOING = 0.30, 0.0
# (name, position, trailing target share). W1 is the one who gets hurt; DEPTH is inactive with no designation the week W1 is out.
ROTATION = [("W1", "WR", 0.30), ("W2", "WR", 0.20), ("W3", "WR", 0.15), ("T1", "TE", 0.15), ("DEPTH", "WR", 0.10)]
SEASONS = (2022, 2023, 2024)
TEAMS = [f"T{i:02d}" for i in range(8)]
OUT_WEEK, LATER_WEEK = 11, 12


def _league():
    """Every team plays weeks 1-12 of three seasons (the trailing window of six games never crosses a season). W1 plays weeks 1-10 and is Out in week 11 (new news) and
    week 12 (ongoing). In week 11 the others' shares are their trailing share times (1 + KAPPA_NEW x the proportional boost); in week 12 they equal the new trailing share
    (the absence is already in it), so the fits must come back as exactly KAPPA_NEW and KAPPA_ONGOING."""
    base, inj, snaps = [], [], []
    boost = 0.30 / (0.20 + 0.15 + 0.15 + 0.10)                         # W1's share over the other four, all 'available' in the model's rule
    for season in SEASONS:
        for team in TEAMS:
            hist = {name: [] for name, _, _ in ROTATION}
            for week in range(1, LATER_WEEK + 1):
                for name, pos, share in ROTATION:
                    if name == "W1":
                        played, actual = week < OUT_WEEK, share
                    else:
                        played = name != "DEPTH" or week < OUT_WEEK        # DEPTH is inactive (no designation, no snaps) from week 11
                        actual = {OUT_WEEK: share * (1.0 + KAPPA_NEW * boost), LATER_WEEK: I._ewma(np.array(hist[name]))}.get(week, share)
                    if played:
                        hist[name].append(actual)
                        base.append({"season": season, "week": week, "team": team, "player_id": f"{team}-{name}", "player_display_name": name, "position": pos,
                                     "target_share": actual, "carry_share": 0.0})
                        snaps.append({"season": season, "week": week, "team": team, "name_key": normalize_name(name), "offense_snaps": 40})
                if week >= OUT_WEEK:
                    inj.append({"season": season, "week": week, "team": team, "gsis_id": f"{team}-W1", "report_status": "out"})
    return pd.DataFrame(base), pd.DataFrame(inj), pd.DataFrame(snaps)


@pytest.fixture(scope="module")
def league():
    return _league()


@pytest.fixture(scope="module")
def obs(league):
    base, inj, snaps = league
    return IA.observations("rec", base, inj, snaps)


def test_a_new_absence_gives_one_row_per_teammate_who_played(obs):
    wk = obs[obs.week == OUT_WEEK]
    assert len(wk) == len(SEASONS) * len(TEAMS) * 3                       # W2, W3, T1: DEPTH had no snaps, W1 is the Out player
    assert set(wk.pid.str.split("-").str[1]) == {"W2", "W3", "T1"}
    assert wk.V.round(6).eq(0.30).all() and wk.V_new.round(6).eq(0.30).all() and wk.V_old.abs().lt(1e-9).all()
    assert wk.V_WR.round(6).eq(0.30).all() and wk.V_TE.eq(0.0).all()        # the Out player's position
    assert wk.pos.isin(["WR", "TE"]).all() and (wk.pos == "TE").sum() == len(SEASONS) * len(TEAMS)


def test_B_is_the_old_proportional_rule(obs):
    wk = obs[(obs.week == OUT_WEEK) & obs.pid.str.endswith("W2")].iloc[0]
    # vacated .30 over the available .60 (DEPTH counts: the model cannot know he is inactive): the old rule scaled every teammate's share by 1.5
    assert wk.B == pytest.approx(wk.s * 1.5)
    assert wk.a == pytest.approx(wk.s * (1.0 + KAPPA_NEW * 0.5))


def test_the_second_week_out_is_an_ongoing_absence(obs):
    later = obs[obs.week == LATER_WEEK]
    assert len(later) and later.V_new.abs().lt(1e-9).all() and later.V_old.round(6).eq(0.30).all()
    assert set(IA.new_absences(obs).week) == {OUT_WEEK}
    assert set(IA.ongoing_absences(obs).week) == {LATER_WEEK}


def test_a_small_vacated_share_is_not_an_observation(league):
    base, inj, snaps = league
    small = base.assign(target_share=np.where(base.player_display_name.eq("W1"), 0.06, base.target_share))
    assert IA.observations("rec", small, inj, snaps).empty              # .06 vacated < MIN_V
    assert not IA.observations("rec", small, inj, snaps, min_v=0.05).empty


def test_a_questionable_teammate_is_not_vacated_and_does_not_count_as_out(league):
    base, inj, snaps = league
    q = pd.concat([inj, pd.DataFrame([{"season": 2022, "week": OUT_WEEK, "team": "T00", "gsis_id": "T00-W3", "report_status": "questionable"}])], ignore_index=True)
    o = IA.observations("rec", base, q, snaps)
    row = o[(o.season == 2022) & (o.week == OUT_WEEK) & (o.team == "T00")]
    assert set(row.pid.str.split("-").str[1]) == {"W2", "T1"}            # W3 carries a designation: not a 'healthy teammate'
    assert row.V.round(6).eq(0.30).all()                                 # his 'vacated' share (30% of .15) is not an Out player's
    assert row.B.iloc[0] / row.s.iloc[0] == pytest.approx(1 + (0.30 + 0.15 * 0.3) / (0.20 + 0.15 * 0.7 + 0.15 + 0.10))


def test_fit_kappa_recovers_the_true_fraction_leaving_one_season_out(obs):
    new = IA.fit_kappa(IA.new_absences(obs))
    assert new["kappa"] == pytest.approx(KAPPA_NEW, abs=1e-6)
    assert set(new["kappa_by_season"]) == set(SEASONS) and all(v == pytest.approx(KAPPA_NEW, abs=0.01) for v in new["kappa_by_season"].values())
    assert new["team_weeks"] == len(SEASONS) * len(TEAMS) and new["n"] == new["team_weeks"] * 3
    assert new["rmse"]["K"] < 1e-6 < new["rmse"]["A"] and new["rmse"]["K"] < new["rmse"]["B"]
    assert new["gain_of_vacated"] == pytest.approx(0.50 * KAPPA_NEW * 0.5 / 0.30)       # the three recipients hold .50 of the share and gain kappa x 50% of it
    assert new["net_gain_of_vacated"] == pytest.approx(new["gain_of_vacated"]) and new["beta"] == 1.0      # no usual fall-back in this data
    assert new["gain_by_pos"]["WR"] + new["gain_by_pos"]["TE"] == pytest.approx(new["net_gain_of_vacated"])
    lo, hi = new["kappa_ci"]
    assert lo <= KAPPA_NEW + 1e-9 and hi >= KAPPA_NEW - 1e-9 and new["kappa_raw"] == pytest.approx(new["kappa"])      # exact data: a degenerate interval around the truth
    ongoing = IA.fit_kappa(IA.ongoing_absences(obs))
    assert ongoing["kappa"] == pytest.approx(KAPPA_ONGOING, abs=1e-6) and ongoing["rmse"]["B"] > ongoing["rmse"]["K"]


def test_the_model_rule_is_wrong_by_the_amount_the_fit_says(obs):
    new = IA.fit_kappa(IA.new_absences(obs))
    # the old rule overshoots (kappa .30 of the boost is real), and is further off than doing nothing
    assert new["rmse"]["B"] > new["rmse"]["A"] > new["rmse"]["K"]


def test_the_usual_fall_back_of_a_share_is_not_counted_against_the_boost(obs):
    new = IA.new_absences(obs)
    # recipients' actual share is 1.15 x trailing, the old rule says 1.5 x. If a teammate usually gets only 0.9 of his trailing share, the injury is worth 1.15 - 0.9 = .25
    # of it against a boost of 0.9 x .5 = .45: kappa .556 instead of the .30 that comparing with the plain trailing share gives.
    fit = IA.fit_kappa(new.assign(beta=0.9))
    assert fit["beta"] == pytest.approx(0.9) and fit["kappa"] == pytest.approx(0.25 / 0.45) and fit["kappa_raw"] == pytest.approx(0.30)
    assert fit["net_gain_of_vacated"] > fit["gain_of_vacated"]                     # beyond the fall-back the teammates gained more than the plain comparison shows
    assert fit["rmse"]["K"] < 1e-6                                                 # exactly the damped boost on top of the fall-back
    assert fit["rmse"]["A"] > 0 and fit["rmse"]["B"] > fit["rmse"]["K"]


def test_leakage_is_learned_on_the_other_seasons_only():
    clean = pd.DataFrame({"season": [2022] * 4 + [2023] * 4 + [2024] * 4, "s": 0.2, "a": [0.16] * 4 + [0.20] * 4 + [0.18] * 4})
    got = IA.leakage(clean, [2022, 2023, 2024])
    assert got[2022] == pytest.approx(0.95) and got[2023] == pytest.approx(0.85) and got[2024] == pytest.approx(0.90)
    assert IA.leakage(clean.iloc[:0], [2022]) == {2022: 1.0}                      # no clean weeks: no correction


def test_clean_weeks_are_team_weeks_without_an_out_or_doubtful_player(league):
    base, inj, snaps = league
    clean = IA.clean_weeks("rec", base, inj, snaps)
    assert not clean.empty and clean.week.max() == OUT_WEEK - 1 and set(clean.pos) == {"WR", "TE"}      # W1 is Out from week 11
    assert clean[(clean.season == 2022) & (clean.week == 1)].empty                # nothing before the first game to trail
    q = pd.concat([inj, pd.DataFrame([{"season": 2022, "week": 4, "team": "T00", "gsis_id": "T00-W2", "report_status": "questionable"},
                                      {"season": 2022, "week": 5, "team": "T00", "gsis_id": "T00-W2", "report_status": "doubtful"}])], ignore_index=True)
    c2 = IA.clean_weeks("rec", base, q, snaps)
    both = lambda w: len(c2[(c2.season == 2022) & (c2.week == w) & (c2.team == "T00")])      # noqa: E731
    assert both(4) > 0 and both(5) == 0                                           # Questionable stays clean, Doubtful does not


def test_run_fits_the_whole_pipeline_on_the_league(league):
    base, inj, snaps = league
    fits, own = IA.run(base, inj, snaps)
    new = fits[("rec", "new")]
    assert new["team_weeks"] == len(SEASONS) * len(TEAMS) and fits[("rec", "ongoing")]["team_weeks"] == len(SEASONS) * len(TEAMS)
    assert new["kappa_raw"] == pytest.approx(KAPPA_NEW, abs=1e-6)                 # against the plain trailing share
    assert KAPPA_NEW <= new["kappa"] < 0.4 and new["beta"] < 1.0                  # the first weeks of a season trail the previous one, so a little fall-back shows up
    assert fits[("rush", "new")] is None                                          # nobody carries in this league: no fit rather than a made-up one
    assert own.empty or set(own.status) <= {"questionable", "doubtful"}


def test_questionable_players_who_played_are_compared_with_their_trailing_share(league):
    base, inj, snaps = league
    q = pd.concat([inj, pd.DataFrame([
        {"season": 2022, "week": OUT_WEEK, "team": "T00", "gsis_id": "T00-W2", "report_status": "questionable"},
        {"season": 2022, "week": OUT_WEEK, "team": "T00", "gsis_id": "T00-DEPTH", "report_status": "doubtful"}])], ignore_index=True)
    own = IA.hurt_who_played("rec", base, q, snaps)
    assert set(own.status) == {"questionable", "doubtful"}
    w2, depth = own[own.status == "questionable"].iloc[0], own[own.status == "doubtful"].iloc[0]
    assert bool(w2.played) and w2.s == pytest.approx(0.20) and w2.a > w2.s     # played, and absorbed a little of W1's usage
    assert not bool(depth.played) and depth.a == 0.0


def _fits():
    fake = {"n": 100, "team_weeks": 40, "mean_vacated": 0.21, "beta": 0.91, "kappa": 0.46, "kappa_ci": (0.35, 0.58), "kappa_raw": 0.14, "kappa_by_season": {2022: 0.17, 2023: 0.2},
            "rmse": {"A": 8.18, "B": 8.85, "K": 8.1}, "gain_of_vacated": 0.16, "net_gain_of_vacated": 0.41, "gain_by_pos": {"WR": 0.39, "TE": -0.04, "RB": 0.06}}
    return {("rec", "new"): fake, ("rec", "ongoing"): dict(fake, kappa=0.39), ("rush", "new"): dict(fake, kappa=0.75), ("rush", "ongoing"): None}


def test_the_report_states_the_shipped_scales_and_survives_a_missing_fit():
    shipped = {"rec": {"new": 0.45, "ongoing": 0.40}, "rush": {"new": 0.75, "ongoing": 0.45}}
    own = pd.DataFrame({"unit": ["rec"] * 3, "status": ["questionable"] * 3, "season": 2024, "played": [True, True, False], "s": [0.2, 0.1, 0.1], "a": [0.18, 0.09, 0.0]})
    txt = IA.build_report(_fits(), own, shipped)
    for needle in ("## Receiving", "## Rushing", "## What was shipped", "## Not changed in v4", "Receiving 0.45 for a new absence and 0.40 for an ongoing one",
                   "rushing 0.75 / 0.45", "NEW (played the team's last game)", "ONGOING (already missed it)", "WR +0.39, TE -0.04, RB +0.06",
                   "kappa (95% CI)", "0.46 (+0.35 to +0.58)", "kappa raw", "| 0.91 |", "re-learned"):
        assert needle in txt
    assert "0.90" in txt                                                 # 0.27 / 0.30 of trailing share when he played, against the 0.70 multiplier
    assert "| ongoing | 0 |" in txt                                       # the missing (rush, ongoing) fit still gets a row
    assert "Not changed in v4" not in IA.build_report(_fits(), pd.DataFrame(), shipped)


def test_the_report_file_is_where_the_docs_say_it_is():
    assert IA.REPORT.name == "injury_absorption_report.md" and IA.REPORT.parent.name == "matchup_model"
    assert I._UNIT_COL == {"rec": "target_share", "rush": "carry_share"}
