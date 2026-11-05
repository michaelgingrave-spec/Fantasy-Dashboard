"""The game simulator and the parlay pricer: identities that must hold on every path, means that must be preserved, and the pricing arithmetic."""
from __future__ import annotations

import numpy as np
import pytest

from matchup_model.sim import engine as E
from matchup_model.sim import sgp as S
from matchup_model.sim import validate as V


def _team(abbr: str, implied: float, qb_name: str, wr_names: tuple[str, ...], rb_name: str) -> E.TeamSpec:
    players = [E.PlayerSpec(qb_name, "QB", pass_att=33, pass_yds=250, pass_td=1.7, ints=0.7, rush_yds=14, rush_td=0.1),
               E.PlayerSpec(rb_name, "RB", rush_att=16, rush_yds=70, rush_td=0.5, rec=2.5, rec_yds=18, rec_td=0.1)]
    for i, nm in enumerate(wr_names):
        players.append(E.PlayerSpec(nm, "WR", tgt=9 - 2 * i, rec=6 - 1.4 * i, rec_yds=80 - 18 * i, rec_td=0.45 - 0.1 * i))
    players.append(E.PlayerSpec(f"{abbr} TE", "TE", tgt=5, rec=3.6, rec_yds=40, rec_td=0.3))
    return E.TeamSpec(abbr, implied, players)


@pytest.fixture(scope="module")
def game():
    home = _team("HOM", 25.0, "Home QB", ("Home WR1", "Home WR2", "Home WR3"), "Home RB")
    away = _team("AWY", 20.0, "Away QB", ("Away WR1", "Away WR2", "Away WR3"), "Away RB")
    return home, away, E.simulate(home, away, n=30000, seed=11)


def test_same_seed_same_paths():
    home = _team("HOM", 25.0, "Home QB", ("Home WR1", "Home WR2"), "Home RB")
    away = _team("AWY", 20.0, "Away QB", ("Away WR1", "Away WR2"), "Away RB")
    a = E.simulate(home, away, n=2000, seed=3)
    b = E.simulate(home, away, n=2000, seed=3)
    c = E.simulate(home, away, n=2000, seed=4)
    assert np.array_equal(a.team["HOM"]["points"], b.team["HOM"]["points"])
    assert not np.array_equal(a.team["HOM"]["points"], c.team["HOM"]["points"])


def test_team_points_average_the_implied_totals(game):
    _, _, r = game
    assert abs(r.team["HOM"]["points"].mean() - 25.0) < 0.35
    assert abs(r.team["AWY"]["points"].mean() - 20.0) < 0.35
    assert abs(r.team["HOM"]["margin"].mean() - 5.0) < 0.5


def test_points_are_made_of_touchdowns_and_field_goals(game):
    _, _, r = game
    for t in ("HOM", "AWY"):
        d = r.team[t]
        extra = d["points"] - 7 * d["td"] - 3 * d["fg"]
        assert set(np.unique(extra)) <= {0, 7}                       # only a defensive / special-teams touchdown on top
        assert np.array_equal(d["pass_td"] + d["rush_td"], d["td"]) or (d["pass_td"] + d["rush_td"] <= d["td"]).all()


def test_unpinned_quarterback_line_is_the_sum_of_his_receivers():
    home = _team("HOM", 25.0, "Home QB", ("Home WR1", "Home WR2", "Home WR3"), "Home RB")
    away = _team("AWY", 20.0, "Away QB", ("Away WR1", "Away WR2", "Away WR3"), "Away RB")
    r = E.simulate(home, away, n=6000, seed=11, params=E.SimParams().with_(qb_pin=False))
    recv = [p.name for p in home.players if p.pos in ("WR", "TE", "RB")]
    rec_yds = sum(r.stats[n]["rec_yds"] for n in recv)
    rec = sum(r.stats[n]["rec"] for n in recv)
    qb = r.stats["Home QB"]
    assert (qb["pass_yds"] >= rec_yds - 1e-6).all()                  # the rest is the unlisted depth receivers
    assert (qb["comp"] >= rec).all()
    assert (qb["pass_yds"] - rec_yds).mean() < 0.12 * qb["pass_yds"].mean()
    assert (qb["pass_td"] >= sum(r.stats[n]["rec_td"] for n in recv)).all()


def test_pinned_quarterback_is_set_to_his_own_projection_and_stays_inside_the_team(game):
    home, _, r = game
    qb, d = r.stats["Home QB"], r.team["HOM"]
    assert abs(qb["pass_yds"].mean() / 250 - 1) < 0.01                # the projected 250 yards, not the receivers' total
    assert abs(qb["pass_att"].mean() / 33 - 1) < 0.01
    assert (qb["pass_yds"] <= d["pass_yds"] + 1e-6).all() and (qb["pass_att"] <= d["pass_att"] + 1).all()
    assert (qb["comp"] <= qb["pass_att"]).all() and (qb["pass_td"] <= qb["comp"]).all() and (qb["int"] <= qb["pass_att"]).all()
    assert np.corrcoef(qb["pass_yds"], r.stats["Home WR1"]["rec_yds"])[0, 1] > 0.25         # still moves with his receivers


def test_receivers_cannot_score_without_catching_and_targets_cover_catches(game):
    _, _, r = game
    for name, s in r.stats.items():
        assert (s["rec"] <= s["tgt"]).all()
        assert (s["rec_td"] <= s["rec"]).all()
        assert (s["rec_yds"][s["rec"] == 0] == 0).all()


def test_every_players_mean_is_his_projection(game):
    home, _, r = game
    for p in home.players:
        if p.pos == "WR":
            assert abs(r.stats[p.name]["tgt"].mean() - p.tgt) < 0.04 * p.tgt
            assert abs(r.stats[p.name]["rec"].mean() - p.rec) < 0.05 * p.rec + 0.05
            assert abs(r.stats[p.name]["rec_yds"].mean() - p.rec_yds) < 0.06 * p.rec_yds
    rb = [p for p in home.players if p.pos == "RB"][0]
    assert abs(r.stats[rb.name]["rush_att"].mean() - rb.rush_att) < 0.04 * rb.rush_att
    assert abs(r.stats[rb.name]["rush_yds"].mean() - rb.rush_yds) < 0.06 * rb.rush_yds


def test_a_backup_quarterback_projected_for_a_few_passes_throws_the_teams_passes_if_he_is_in_the_game():
    """His projection is weighted by his chance to play (14 yards, 3 attempts); with the starter removed he is the quarterback, and pinning him to it would leave the team with no passing game."""
    full = _team("HOM", 25.0, "Home QB", ("Home WR1", "Home WR2", "Home WR3"), "Home RB")
    players = [E.PlayerSpec("Backup QB", "QB", pass_att=3.4, pass_yds=14.0, pass_td=0.1, ints=0.1) if p.pos == "QB" else p for p in full.players]
    away = _team("AWY", 20.0, "Away QB", ("Away WR1", "Away WR2", "Away WR3"), "Away RB")
    r = E.simulate(E.TeamSpec("HOM", 25.0, players), away, n=8000, seed=5)
    assert r.stats["Backup QB"]["pass_yds"].mean() > 180 and r.stats["Backup QB"]["pass_att"].mean() > 28
    assert abs(r.stats["Backup QB"]["pass_yds"].mean() / r.team["HOM"]["pass_yds"].mean() - 1) < 0.01       # the team's passing line, as an unpinned quarterback gets it


def test_a_thin_list_leaves_the_quarterbacks_volume_and_the_receivers_projections_alone():
    """With most receivers left out of the simulation (a late scratch), the team still throws what its quarterback is projected for; the targets nobody listed catches go to the unlisted slot."""
    full = _team("HOM", 25.0, "Home QB", ("Home WR1", "Home WR2", "Home WR3"), "Home RB")
    thin = E.TeamSpec("HOM", 25.0, [p for p in full.players if p.name not in ("Home WR2", "Home WR3")])
    away = _team("AWY", 20.0, "Away QB", ("Away WR1", "Away WR2", "Away WR3"), "Away RB")
    r = E.simulate(thin, away, n=8000, seed=5)
    assert abs(r.stats["Home QB"]["pass_att"].mean() / 33 - 1) < 0.02
    assert r.team["HOM"]["pass_att"].mean() >= 33 * E.SimParams().pass_scale - 0.5
    wr1 = next(p for p in thin.players if p.name == "Home WR1")
    assert abs(r.stats["Home WR1"]["tgt"].mean() - wr1.tgt) < 0.06 * wr1.tgt            # the listed teammate does not inherit the targets


def test_yardage_is_right_skewed(game):
    _, _, r = game
    x = r.stats["Home WR2"]["rec_yds"]
    assert np.median(x) < x.mean()


def test_a_trailing_team_throws_more_and_runs_less(game):
    _, _, r = game
    d = r.team["AWY"]
    losing = d["margin"] < -10
    winning = d["margin"] > 10
    share = d["pass_att"] / (d["pass_att"] + d["rush_att"])             # more passing yards also mean more points, so the raw count of passes hardly moves with the margin; the mix does
    assert share[losing].mean() > share[winning].mean() + 0.02
    assert d["rush_att"][losing].mean() < d["rush_att"][winning].mean()


def test_passing_yards_and_touchdowns_move_together(game):
    _, _, r = game
    q = r.stats["Home QB"]
    assert np.corrcoef(q["pass_yds"], q["pass_td"])[0, 1] > 0.15
    assert np.corrcoef(q["pass_yds"], r.stats["Home WR1"]["rec_yds"])[0, 1] > 0.25


# ------------------------------------------------------------------------------------------------------------------------------------------------ pricing
def test_odds_ratio_fit_is_independent_when_the_odds_ratio_is_one():
    assert V.odds_ratio_fit(0.25, 0.25, 0.25, 0.25, 0.4, 0.7) == pytest.approx(0.28)


def test_odds_ratio_fit_keeps_the_margins_and_the_direction_of_association():
    q = V.odds_ratio_fit(0.40, 0.10, 0.10, 0.40, 0.3, 0.6)          # positive association in the simulator
    assert q > 0.3 * 0.6
    assert q < min(0.3, 0.6)
    assert V.odds_ratio_fit(0.05, 0.45, 0.45, 0.05, 0.5, 0.5) < 0.25  # negative association


def test_a_leg_the_simulator_never_hits_is_priced_as_independent_not_as_impossible(game):
    """A line far above what the simulator can produce (a depth receiver projected for 1 yard against a 15-yard line) has no paths to estimate an association from: the half-path continuity
    correction prices it like independence instead of as a pair that can never (or always) hit together."""
    _, _, r = game
    legs = [S.Leg("rec_yds", "over", 400.5, player="Home WR3", p=0.50), S.Leg("rec_yds", "over", 60.5, player="Away WR1", p=0.50)]
    assert S.leg_prob(r, legs[0]) == 0.0
    out = S.price(r, legs, mode="market")
    assert out["joint"] == pytest.approx(0.25, abs=0.03)
    few = (0.0, 3 / 8000, 0.5, 0.5 - 3 / 8000)                       # three paths in 8000 have the leg: none of them also has the other leg
    assert V.odds_ratio_fit(*few, 0.5, 0.5) < 0.01                   # an odds ratio built from three paths says the pair can never hit together ...
    assert 0.10 < V.odds_ratio_fit(*few, 0.5, 0.5, n=8000) < 0.25    # ... the half-path correction leaves a mild negative association


def test_ipf_matches_the_requested_margins():
    ev = [np.random.default_rng(1).random(5000) < 0.4, np.random.default_rng(2).random(5000) < 0.5]
    ev[1] = ev[1] | ev[0]                                            # make them associated
    tab = S.cell_table(ev, np.full(5000, 1 / 5000))
    fit = S.ipf_joint(tab, [0.30, 0.55])
    assert fit[1, :].sum() == pytest.approx(0.30, abs=1e-6)
    assert fit[:, 1].sum() == pytest.approx(0.55, abs=1e-6)
    assert fit[1, 1] > 0.30 * 0.55


def test_price_of_independent_legs_is_the_product_and_fair_odds_invert(game):
    home, away, r = game
    legs = [S.Leg("rec_yds", "over", 60.5, player="Home WR1", p=0.5), S.Leg("rec_yds", "over", 45.5, player="Away WR1", p=0.5)]
    out = S.price(r, legs, mode="market")
    assert out["independent"] == pytest.approx(0.25)
    assert abs(out["lift"] - 1.0) < 0.2                              # opposing receivers are close to independent
    assert out["fair_decimal"] == pytest.approx(1 / out["joint"])


def test_teammates_of_the_same_offense_are_priced_as_positively_correlated(game):
    _, _, r = game
    legs = [S.Leg("pass_yds", "over", 250.5, player="Home QB", p=0.5), S.Leg("rec_yds", "over", 70.5, player="Home WR1", p=0.5)]
    out = S.price(r, legs, mode="market")
    assert out["lift"] > 1.25
    assert out["joint"] > 0.3
    assert out["corr"][0, 1] > 0.2


def test_spread_total_and_moneyline_legs(game):
    _, _, r = game
    cover = S.leg_prob(r, S.Leg("spread", "over", -3.5, team="HOM"))
    win = S.leg_prob(r, S.Leg("moneyline", "over", team="HOM"))
    assert win > 0.5 > S.leg_prob(r, S.Leg("moneyline", "over", team="AWY"))
    assert cover < win
    assert 0.2 < S.leg_prob(r, S.Leg("total", "over", 45.5)) < 0.7


def test_prices_and_expected_value_arithmetic():
    assert S.to_american(0.5) == pytest.approx(-100.0) or S.to_american(0.5) == pytest.approx(100.0)
    assert S.to_american(0.25) == pytest.approx(300.0)
    assert S.to_american(0.75) == pytest.approx(-300.0)
    assert S.no_vig(-110, -110) == pytest.approx(0.5)
    assert S.ev_at(0.25, 300) == pytest.approx(0.0)
    assert S.ev_at(0.30, 300) == pytest.approx(0.2)


def test_market_mode_pins_each_leg_to_the_book_and_own_mode_does_not(game):
    _, _, r = game
    legs = [S.Leg("pass_yds", "over", 250.5, player="Home QB", p=0.40), S.Leg("rec_yds", "over", 70.5, player="Home WR1", p=0.45)]
    mk = S.price(r, legs, mode="market")
    own = S.price(r, legs, mode="own")
    assert mk["leg_prob"] == [0.40, 0.45]
    assert own["leg_prob"] == pytest.approx(own["sim_leg_prob"])


def test_conditional_is_the_priced_joint_over_the_given_legs_chance_and_respects_the_pins(game):
    _, _, r = game
    qb = S.Leg("pass_yds", "over", 250.5, player="Home QB", p=0.40)
    wr = S.Leg("rec_yds", "over", 70.5, player="Home WR1", p=0.45)
    rb = S.Leg("rush_yds", "over", 70.5, player="Away RB")                                   # no price: the simulator's own number
    out = S.conditional(r, [qb], [wr, rb])
    for leg, o in zip((wr, rb), out):
        priced = S.price(r, [qb, leg], mode="market")
        assert o["joint"] == pytest.approx(priced["joint"], abs=1e-9)                        # the same table and the same pinning as the parlay price
        assert o["when_given"] == pytest.approx(o["joint"] / 0.40) and o["lift"] == pytest.approx(o["when_given"] / o["alone"])
    assert out[0]["alone"] == 0.45 and out[1]["alone"] == pytest.approx(S.leg_prob(r, rb))
    assert out[0]["lift"] > 1.2                                                              # a quarterback and his wideout hit together
    two = S.conditional(r, [qb, wr], [S.Leg("any_td", "over", player="HOM TE")])[0]         # given more than one leg: the denominator is their priced joint
    assert two["joint"] / two["when_given"] == pytest.approx(S.price(r, [qb, wr], mode="market")["joint"], abs=1e-9)
    with pytest.raises(ValueError):
        S.conditional(r, [], [wr])


def test_compact_shrinks_the_arrays_without_changing_the_numbers():
    home = _team("HOM", 25.0, "Home QB", ("Home WR1", "Home WR2"), "Home RB")
    away = _team("AWY", 20.0, "Away QB", ("Away WR1", "Away WR2"), "Away RB")
    full = E.simulate(home, away, n=3000, seed=2)
    c = E.simulate(home, away, n=3000, seed=2).compact()
    assert c.stats["Home WR1"]["rec_yds"].dtype == np.float32 and c.stats["Home WR1"]["rec"].dtype == np.int16
    assert abs(float(c.stats["Home WR1"]["rec_yds"].mean()) - float(full.stats["Home WR1"]["rec_yds"].mean())) < 0.01
    assert S.leg_prob(c, S.Leg("rec_yds", "over", 60.5, player="Home WR1")) == pytest.approx(S.leg_prob(full, S.Leg("rec_yds", "over", 60.5, player="Home WR1")), abs=1e-3)
    assert np.array_equal(c.team["HOM"]["margin"], full.team["HOM"]["margin"])
