"""Same-game parlay pricing from simulated paths.

A leg is one yes / no event in the game (a player's stat over or under a number, a team's margin or points). `joint_prob` is the share of paths where every leg
hits. Two ways to use it:

  * `own`     -- the simulator's own probabilities, end to end;
  * `market`  -- each leg is first pinned to the book's no-vig probability (iterative proportional fitting of the 2^k table of leg outcomes), so the single legs are
                 the market's and only the ASSOCIATION between them comes from the simulator. This is the parlay the books are pricing, with the correlation they should
                 be charging for.

`lift` is the joint probability over the product of the legs' probabilities (1.0 = independent). The fair price of the parlay is 1 / joint probability.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from matchup_model.sim.engine import SimResult

PLAYER_MARKETS = {"pass_yds": "pass_yds", "pass_td": "pass_td", "pass_att": "pass_att", "comp": "comp", "int": "int", "rush_att": "rush_att", "rush_yds": "rush_yds",
                  "rush_td": "rush_td", "tgt": "tgt", "rec": "rec", "rec_yds": "rec_yds", "rec_td": "rec_td", "any_td": "any_td"}
TEAM_MARKETS = ("spread", "moneyline", "team_total", "total")


@dataclass(frozen=True)
class Leg:
    market: str                       # a key of PLAYER_MARKETS, or 'spread' / 'moneyline' / 'team_total' / 'total'
    side: str                         # 'over' | 'under' (anytime TD: 'over' = yes, 'under' = no; spread / moneyline: 'over' = the team covers / wins)
    line: float = 0.0                 # the number; for 'spread' it is the team's own handicap (-10.5 for a 10.5-point favourite)
    player: str | None = None
    team: str | None = None
    p: float | None = None            # the book's no-vig probability that this leg hits (for the market-pinned price)

    def label(self) -> str:
        who = self.player or self.team or "game"
        if self.market == "any_td":
            return f"{who} anytime TD" + ("" if self.side == "over" else " (no)")
        if self.market == "moneyline":
            return f"{who} to win"
        if self.market == "spread":
            return f"{who} {self.line:+g}"
        word = "over" if self.side == "over" else "under"
        return f"{who} {word} {self.line:g} {self.market.replace('_', ' ')}"


def on_side(leg: Leg, x):
    """Whether the number x (a player's stat, a team's margin or points, the game total; a scalar or an array) is on the leg's side of its line. A number exactly on the
    line counts as a miss for both sides, the conservative treatment of a push; anytime TD is x > 0 (yes) or x == 0 (no); spread and moneyline read x as the team's margin."""
    over = leg.side == "over"
    if leg.market == "any_td":
        return (x > 0.5) if over else (x < 0.5)
    if leg.market == "spread":
        return (x + leg.line > 0) if over else (x + leg.line < 0)
    if leg.market == "moneyline":
        return (x > 0) if over else (x < 0)
    return (x > leg.line) if over else (x < leg.line)


def leg_events(res: SimResult, leg: Leg) -> np.ndarray:
    """Boolean (n,) array: the leg hits on each simulated path (see `on_side` for a stat exactly on the number)."""
    m = leg.market
    if m in PLAYER_MARKETS:
        if leg.player not in res.stats:
            raise KeyError(f"{leg.player} is not in the simulated game")
        return on_side(leg, res.stats[leg.player][PLAYER_MARKETS[m]].astype(float))
    if m in ("spread", "moneyline"):
        return on_side(leg, res.team[leg.team]["margin"])
    if m == "team_total":
        return on_side(leg, res.team[leg.team]["points"])
    if m == "total":
        return on_side(leg, sum(v["points"] for v in res.team.values()))
    raise ValueError(f"unknown market {m!r}")


def _weights(res: SimResult) -> np.ndarray:
    return res.weights if res.weights is not None else np.full(res.n, 1.0 / res.n)


def leg_prob(res: SimResult, leg: Leg) -> float:
    return float((leg_events(res, leg) * _weights(res)).sum())


def cell_table(events: list[np.ndarray], w: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """The 2^k table of how the k legs came out (index 1 = hit), as probabilities, from the simulated paths."""
    k = len(events)
    idx = np.zeros(len(w), dtype=np.int64)
    for j, e in enumerate(events):
        idx |= e.astype(np.int64) << j
    flat = np.bincount(idx, weights=w, minlength=2 ** k) + eps
    return (flat / flat.sum()).reshape((2,) * k, order="F")     # axis j <-> leg j


def ipf_joint(table: np.ndarray, targets: list[float], iters: int = 500, tol: float = 1e-10) -> np.ndarray:
    """Fit the table's single-leg margins to `targets` (P(leg j hits)) while keeping its odds ratios (iterative proportional fitting)."""
    t = table.copy()
    k = t.ndim
    for _ in range(iters):
        worst = 0.0
        for j in range(k):
            axes = tuple(a for a in range(k) if a != j)
            m1 = float(np.take(t, 1, axis=j).sum())
            p = float(np.clip(targets[j], 1e-6, 1 - 1e-6))
            worst = max(worst, abs(m1 - p))
            f1, f0 = p / max(m1, 1e-12), (1 - p) / max(1 - m1, 1e-12)
            sl1 = [slice(None)] * k
            sl0 = [slice(None)] * k
            sl1[j], sl0[j] = 1, 0
            t[tuple(sl1)] *= f1
            t[tuple(sl0)] *= f0
        if worst < tol:
            break
    return t / t.sum()


def odds_ratio_fit(p11: float, p10: float, p01: float, p00: float, a: float, b: float, n: float | None = None) -> float:
    """P(both) for a 2x2 table whose odds ratio is kept but whose margins are set to P(A)=a, P(B)=b (closed form of iterative proportional fitting).

    With `n` (the number of simulated paths behind the table) every cell gets half a path first (the usual continuity correction): a leg the simulator thinks (almost) impossible has
    only a handful of paths in two of its cells, and an odds ratio built from three paths is noise that would otherwise price the pair as if it could never (or always) hit together."""
    if n:
        p11, p10, p01, p00 = ((x * n + 0.5) / (n + 2.0) for x in (p11, p10, p01, p00))
    p11, p10, p01, p00 = (max(x, 1e-9) for x in (p11, p10, p01, p00))
    th = (p11 * p00) / (p10 * p01)
    if abs(th - 1) < 1e-9:
        return a * b
    A = th - 1
    B = -(1 + A * (a + b))
    C = th * a * b
    disc = max(B * B - 4 * A * C, 0.0)
    x = (-B - np.sqrt(disc)) / (2 * A)
    return float(min(max(x, max(0.0, a + b - 1) + 1e-12), min(a, b) - 1e-12))


def price(res: SimResult, legs: list[Leg], mode: str = "market") -> dict:
    """The parlay: joint probability, the product of the single legs, the lift, the fair decimal and American odds, each leg's probability and the pairwise correlations.

    mode 'market' pins every leg that has `p` to the book's number (legs without one keep the simulator's); mode 'own' uses the simulator throughout."""
    if not legs:
        raise ValueError("a parlay needs at least one leg")
    w = _weights(res)
    ev = [leg_events(res, l) for l in legs]
    own = [float((e * w).sum()) for e in ev]
    if mode == "market":
        targets = [l.p if l.p is not None else o for l, o in zip(legs, own)]
    else:
        targets = own
    if len(legs) == 1:
        joint = targets[0] if mode == "market" else own[0]
        table = None
    else:
        table = cell_table(ev, w, eps=0.5 * float((w ** 2).sum()))      # half a path per cell (continuity correction, see odds_ratio_fit)
        fitted = ipf_joint(table, targets) if mode == "market" else table
        joint = float(fitted[(1,) * len(legs)])
    indep = float(np.prod(targets))
    own_joint = float((np.logical_and.reduce(ev) * w).sum())
    corr = np.eye(len(legs))
    for i, j in itertools.combinations(range(len(legs)), 2):
        a, b = ev[i].astype(float), ev[j].astype(float)
        vi, vj = own[i] * (1 - own[i]), own[j] * (1 - own[j])
        cov = float((w * a * b).sum()) - own[i] * own[j]
        corr[i, j] = corr[j, i] = cov / np.sqrt(vi * vj) if vi * vj > 0 else 0.0
    n_eff = 1.0 / float((w ** 2).sum())
    se = float(np.sqrt(max(joint * (1 - joint), 0.0) / n_eff))
    return {"legs": [l.label() for l in legs], "leg_prob": targets, "sim_leg_prob": own, "joint": joint, "se": se, "independent": indep, "lift": joint / indep if indep > 0 else float("nan"),
            "sim_joint": own_joint, "fair_decimal": 1.0 / joint if joint > 0 else float("inf"), "fair_american": to_american(joint), "corr": corr}


def conditional(res: SimResult, given: list[Leg], candidates: list[Leg]) -> list[dict]:
    """How often each candidate leg hits when every leg in `given` hits, in the market-pinned world of `price` (a leg with `p` is pinned to it, the others keep the simulator's number).

    One dict per candidate: `alone` (its probability), `joint` (the given legs and this one all hit), `when_given` (joint over the chance the given legs all hit) and `lift` (when_given
    over alone: above 1 the candidate tends to hit with the given legs). The given legs' table is built once; each candidate adds one more leg to it."""
    if not given:
        raise ValueError("conditional needs at least one given leg")
    w = _weights(res)
    eps = 0.5 * float((w ** 2).sum())
    ev = [leg_events(res, l) for l in given]
    t = [l.p if l.p is not None else float((e * w).sum()) for l, e in zip(given, ev)]
    base = float(ipf_joint(cell_table(ev, w, eps), t)[(1,) * len(ev)]) if len(ev) > 1 else t[0]
    out = []
    for c in candidates:
        e = leg_events(res, c)
        alone = c.p if c.p is not None else float((e * w).sum())
        joint = float(ipf_joint(cell_table(ev + [e], w, eps), t + [alone])[(1,) * (len(ev) + 1)])
        out.append({"alone": alone, "joint": joint, "when_given": joint / max(base, 1e-12), "lift": joint / max(base * alone, 1e-12)})
    return out


def to_american(p: float) -> float:
    if p <= 0:
        return float("inf")
    if p >= 1:
        return float("-inf")
    return -100 * p / (1 - p) if p > 0.5 else 100 * (1 - p) / p


def to_prob_american(price: float) -> float:
    return 100.0 / (price + 100.0) if price > 0 else -price / (-price + 100.0)


def no_vig(over_price: float, under_price: float) -> float:
    """No-vig probability of the over from its two American prices."""
    po, pu = to_prob_american(over_price), to_prob_american(under_price)
    return po / (po + pu)


def ev_at(joint: float, offered_american: float) -> float:
    """Expected profit per unit staked at the offered parlay price, if `joint` is the true probability."""
    dec = 1 + (offered_american / 100 if offered_american > 0 else 100 / -offered_american)
    return joint * dec - 1.0
