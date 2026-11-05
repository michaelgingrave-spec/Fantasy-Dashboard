"""The game simulator: one game, many draws, every player's box score and both teams' scores.

It is a stochastic wrapper around the projections, not a second projection: each player's EXPECTED line comes from the model (`PlayerSpec`), and the simulator adds what a
set of means cannot say -- how outcomes are spread (skewed, not normal) and how they move together. The dependence comes from the structure of a football game:

  * a shared TEMPO factor moves both teams' play counts together (a shootout or a slog), and a SCRIPT factor (one team leading, the other trailing) shifts each team's
    pass / run mix: trailing teams throw more and run less;
  * a team's targets are split among its receivers (a Dirichlet-multinomial, so teammates compete for a fixed pie), each target is caught or not, and each catch gains yards
    (gamma, with a per-game random effect for the player and a team-efficiency factor shared by all of its players);
  * the quarterback's line is the SUM of his receivers' lines (completions, yards, touchdowns), so QB and receiver outcomes agree by construction;
  * the team's yards then drive its scoring: touchdowns and field goals come from a latent that mixes the team's total yards, the script factor and luck, so a big
    passing day is a high-scoring day, and the margin follows;
  * touchdowns are divided into passing and rushing (by how the yards split) and handed to players by their touchdown share and the volume they got in that path.

Every player's MEAN is his projection (`vol_pass` / `vol_rush` are 1.0; an earlier fit scaled everyone down to the team volume actually seen, which put each role player 6-9% under
his own projection and made the simulator's overs run 4-7 points too rare). The projections over-allocate a little in total (the receivers' targets add up to about 12% more than the
quarterback's projected attempts, where the team throws only 3.6% more), so the team's volume here is the receivers' total, at least what the quarterback's projection implies, and the
quarterback's own line is then set to HIS projection (`qb_pin`): he throws his projected share of the team's passes, and his yards are the team's yards scaled to his projection. A
quarterback projected for under 60% of the team's passing is a backup whose projection is weighted by his chance to play: if he is in the simulation at all he throws the team's passes.

The noise parameters live in `SimParams`; `calibrate.py` checks them against past box scores and `validate.py` against lines the simulator was not fitted to.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

RB_CATCH = 0.75                       # receptions per target for a back whose projection carries catches but not targets (model.RB_CATCH)
QB_YPC = 5.8                          # yards per carry for a quarterback whose projection carries rushing yards but not carries
POS_CATCH = {"WR": 0.64, "TE": 0.70, "RB": 0.78, "QB": 0.60}
POS_YPC = {"WR": 13.5, "TE": 10.5, "RB": 7.5, "QB": 10.0}
POS_RUSH_YPC = {"RB": 4.4, "QB": QB_YPC, "WR": 6.5, "TE": 4.0}
STATS = ("pass_att", "comp", "pass_yds", "pass_td", "int", "rush_att", "rush_yds", "rush_td", "tgt", "rec", "rec_yds", "rec_td")


@dataclass
class PlayerSpec:
    """One player's EXPECTED box score (the model's projection); anything not projected is 0."""
    name: str
    pos: str
    tgt: float = 0.0
    rec: float = 0.0
    rec_yds: float = 0.0
    rec_td: float = 0.0
    rush_att: float = 0.0
    rush_yds: float = 0.0
    rush_td: float = 0.0
    pass_att: float = 0.0
    pass_yds: float = 0.0
    pass_td: float = 0.0
    ints: float = 0.0


@dataclass
class TeamSpec:
    abbr: str
    implied: float                    # the market's implied points for this team
    players: list[PlayerSpec] = field(default_factory=list)


@dataclass
class SimParams:
    # scoring: touchdowns and field goals as rounded latent normals, so points land on real scores and the margin / total spread matches history
    fg_mean: float = 1.35
    fg_slope: float = 0.030           # fewer expected field goals for a high implied total
    misc_pts: float = 0.35            # defensive / special-teams scores and safeties, in points per team
    sd_td: float = 1.278
    sd_fg: float = 1.00
    fg_td_corr: float = 0.30          # a team that scores touchdowns kicks fewer field goals
    a_yards: float = 0.63             # weight of the team's standardised total yards in the scoring latent
    a_script: float = 0.3            # weight of the script factor (leading teams score more)
    def_td_p: float = 0.05
    # volume
    vol_pass: float = 1.0             # team pass volume as a multiple of the sum of its players' projected targets: 1.0 keeps every player's projected mean (see the module note)
    vol_rush: float = 1.0
    pass_scale: float = 1.0355        # team pass attempts per attempt the starting quarterback is projected for (2023-24 box scores: relievers throw the rest); the team's volume is at least this
    qb_pin: bool = True               # the quarterback's line is set to his own projection (False = the sum of his receivers' lines, as before)
    qb_pin_min: float = 0.6           # ... but only when that projection is at least this share of the team's passing volume: a smaller one is a backup's, weighted by his chance to play, and if he is in the game he throws the team's passes
    pass_G: float = 0.075             # tempo factor (shared by both teams), log pass attempts per SD
    rush_G: float = 0.03
    pass_script: float = 0.09        # log pass attempts per SD of the script factor (trailing -> more passes)
    rush_script: float = 0.189        # log rush attempts per SD (leading -> more runs)
    pass_sd: float = 0.156
    rush_sd: float = 0.2366
    # allocation
    kappa_tgt: float = 60           # Dirichlet concentration of target shares (small = volatile roles)
    kappa_car: float = 26.67
    # efficiency
    cr_eff: float = 0.15              # catch-rate logit shift per SD of the team's efficiency factor
    cr_sd: float = 0.10
    rec_shape: float = 1.7            # gamma shape of one catch's yards
    ypc_eff: float = 0.0765             # log yards-per-catch shift per SD of efficiency
    ypc_sd: float = 0.22              # per-player-game random effect on yards per catch
    rush_shape: float = 1.231
    ypc_r_eff: float = 0.07
    ypc_r_sd: float = 0.338
    rush_rho: float = 0             # correlation of rushing efficiency with passing efficiency
    # touchdowns and interceptions
    td_pass_link: float = 0.98        # touchdowns move to the air as passing yards outrun rushing yards (logit per log-ratio)
    int_eff: float = 0.25
    int_script: float = 0.12

    def with_(self, **kw) -> "SimParams":
        return replace(self, **kw)


@dataclass
class SimResult:
    n: int
    stats: dict                       # player name -> stat -> (n,) array
    team: dict                        # team abbr -> {'points','td','fg','pass_att','rush_att','margin',...} -> (n,) arrays
    pos: dict                         # player name -> position
    team_of: dict                     # player name -> team abbr
    weights: np.ndarray | None = None

    def get(self, name: str, stat: str) -> np.ndarray:
        return self.stats[name][stat]

    def compact(self) -> "SimResult":
        """Shrink every array (float32 yards, int16 counts) so a cached game takes a few tens of MB instead of a few hundred; values are unchanged to display precision."""
        for group in (*self.stats.values(), *self.team.values()):
            for k, v in list(group.items()):
                group[k] = v.astype(np.float32) if v.dtype.kind == "f" else v.astype(np.int16)
        return self


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _multinomial(total: np.ndarray, w: np.ndarray, rng) -> np.ndarray:
    """Counts (n, K) summing to `total` (n,) with per-row probabilities `w` (n, K): sequential conditional binomials, vectorised over rows."""
    n, K = w.shape
    out = np.zeros((n, K), dtype=np.int64)
    rem = total.astype(np.int64).copy()
    mass = np.ones(n)
    for k in range(K - 1):
        p = np.clip(w[:, k] / np.maximum(mass, 1e-12), 0.0, 1.0)
        c = rng.binomial(rem, p)
        out[:, k] = c
        rem = rem - c
        mass = mass - w[:, k]
    out[:, K - 1] = rem
    return out


def _alloc_capped(total: np.ndarray, w: np.ndarray, cap: np.ndarray, rng, rounds: int = 4) -> np.ndarray:
    """Like `_multinomial` but nobody gets more than `cap` (a receiver cannot score more touchdowns than he has catches): the excess is re-dealt to those with room left."""
    out = _multinomial(total, w, rng)
    for _ in range(rounds):
        excess = np.maximum(out - cap, 0).sum(1)
        if not excess.any():
            break
        out = np.minimum(out, cap)
        w2 = w * (cap - out > 0)
        s2 = w2.sum(1)
        ok = s2 > 0
        add = np.zeros_like(out)
        if ok.any():
            add[ok] = _multinomial(excess[ok], w2[ok] / s2[ok][:, None], rng)
        out = out + add
    return np.minimum(out, cap)


def _dirichlet_share(s: np.ndarray, kappa: float, n: int, rng) -> np.ndarray:
    """(n, K) share vectors with mean `s` (sums to 1) and concentration `kappa`."""
    g = rng.gamma(np.maximum(kappa * s, 1e-3)[None, :].repeat(n, 0), 1.0)
    return g / g.sum(1, keepdims=True)


def _score(implied: float, z: np.ndarray, rng, P: SimParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Touchdowns, field goals and points for one team across n paths from its scoring latent `z` (standard normal). The touchdown mean is solved on the same draws so
    that E[points] equals the implied total."""
    n = len(z)
    mu_fg = float(np.clip(P.fg_mean - P.fg_slope * (implied - 22.0), 0.7, 2.2))
    zf = -P.fg_td_corr * z + math.sqrt(1 - P.fg_td_corr ** 2) * rng.standard_normal(n)
    dtd = (rng.random(n) < P.def_td_p).astype(np.int64)
    fg = np.maximum(0, np.rint(mu_fg + P.sd_fg * zf)).astype(np.int64)
    mu_td = max((implied - 3 * mu_fg - P.misc_pts) / 7.0, 0.3)
    for _ in range(5):
        td = np.maximum(0, np.rint(mu_td + P.sd_td * z)).astype(np.int64)
        pts = 7 * td + 3 * fg + 7 * dtd
        mu_td += (implied - pts.mean()) / 7.0
    td = np.maximum(0, np.rint(mu_td + P.sd_td * z)).astype(np.int64)
    return td, fg, 7 * td + 3 * fg + 7 * dtd


def starting_qb(team: TeamSpec) -> PlayerSpec | None:
    """The quarterback who is projected to throw the most; he is given the team's passing line (None when the team lists no quarterback)."""
    qbs = [p for p in team.players if p.pos == "QB"]
    return max(qbs, key=lambda p: (p.pass_att, p.pass_yds)) if qbs else None


def _prep_team(team: TeamSpec, P: SimParams | None = None):
    """Receivers, rushers, the LISTED players' usage (sums of the projections, so every player's expected usage is preserved) and the unlisted 'others' slot, which takes whatever volume is
    left to reach a minimum team volume and, for passes, the volume the quarterback's projection implies beyond the listed receivers (a thin list or a late scratch leaves targets that nobody
    listed catches; they are not handed to the listed players)."""
    P = P or SimParams()
    rec = []
    for p in team.players:
        tgt = p.tgt if p.tgt > 0 else (p.rec / RB_CATCH if p.rec > 0 else 0.0)
        if tgt > 0 or p.rec > 0:
            tgt = max(tgt, p.rec / 0.95)
            rec.append((p, tgt))
    s_tgt = sum(t for _, t in rec)
    total = max(s_tgt + max(0.05 * s_tgt, 0.8), 18.0 + 0.9)
    qb = starting_qb(team)
    if P.qb_pin and qb is not None and qb.pass_att > 0:
        total = max(total, qb.pass_att * P.pass_scale / P.vol_pass)
    rush = []
    for p in team.players:
        att = p.rush_att if p.rush_att > 0 else (p.rush_yds / (QB_YPC if p.pos == "QB" else POS_RUSH_YPC.get(p.pos, 4.4)) if p.rush_yds > 0 else 0.0)
        if att > 0:
            rush.append((p, att))
    s_att = sum(a for _, a in rush)
    total_r = max(s_att + max(0.05 * s_att, 0.5), 12.0 + 0.6)
    return rec, s_tgt, total - s_tgt, rush, s_att, total_r - s_att


def _standardise(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x, dtype=float)


def simulate(home: TeamSpec, away: TeamSpec, n: int = 20000, seed: int = 0, params: SimParams | None = None) -> SimResult:
    """`n` draws of the whole game. Returns every listed player's box score (stat -> (n,) array) and both teams' points, touchdowns, field goals and attempts."""
    P = params or SimParams()
    rng = np.random.default_rng(seed)
    teams = {home.abbr: home, away.abbr: away}
    G = rng.standard_normal(n)
    s_home = rng.standard_normal(n)
    script = {home.abbr: s_home, away.abbr: -s_home}               # > 0 = leading
    eff = {t: rng.standard_normal(n) for t in teams}
    rush_eff = {t: P.rush_rho * eff[t] + math.sqrt(1 - P.rush_rho ** 2) * rng.standard_normal(n) for t in teams}

    # ---- phase 1: volume and yards for each team
    ph: dict = {}
    for t, spec in teams.items():
        rec_list, s_tgt, others_tgt, rush_list, s_att, others_att = _prep_team(spec, P)
        base_pass, base_rush = (s_tgt + others_tgt) * P.vol_pass, (s_att + others_att) * P.vol_rush
        a_pass = base_pass * np.exp(P.pass_G * G - P.pass_script * script[t] + P.pass_sd * rng.standard_normal(n))
        a_rush = base_rush * np.exp(P.rush_G * G + P.rush_script * script[t] + P.rush_sd * rng.standard_normal(n))
        a_pass = np.maximum(10, np.rint(a_pass * (base_pass / a_pass.mean()))).astype(np.int64)
        a_rush = np.maximum(8, np.rint(a_rush * (base_rush / a_rush.mean()))).astype(np.int64)

        K = len(rec_list) + 1
        shares = np.array([tg for _, tg in rec_list] + [others_tgt], float)
        shares = shares / shares.sum()
        targets = _multinomial(a_pass, _dirichlet_share(shares, P.kappa_tgt, n, rng), rng)
        rec_ct = np.zeros((n, K), dtype=np.int64)
        rec_yd = np.zeros((n, K))
        for k in range(K):
            if k < len(rec_list):
                p, tg = rec_list[k]
                cr = float(np.clip(p.rec / tg if (tg > 0 and p.rec > 0) else POS_CATCH.get(p.pos, 0.64), 0.30, 0.92))
                ypc = p.rec_yds / p.rec if p.rec > 0 and p.rec_yds > 0 else POS_YPC.get(p.pos, 11.0)
            else:
                cr, ypc = 0.64, 10.5
            pc = _expit(_logit(cr) + P.cr_eff * eff[t] + P.cr_sd * rng.standard_normal(n))
            rec_ct[:, k] = rng.binomial(targets[:, k], pc)
            scale = ypc * np.exp(P.ypc_eff * eff[t] + P.ypc_sd * rng.standard_normal(n) - 0.5 * (P.ypc_sd ** 2 + P.ypc_eff ** 2)) / P.rec_shape
            rec_yd[:, k] = np.where(rec_ct[:, k] > 0, rng.gamma(np.maximum(P.rec_shape * rec_ct[:, k], 1e-9), scale), 0.0)

        R = len(rush_list) + 1
        rshare = np.array([a for _, a in rush_list] + [others_att], float)
        rshare = rshare / rshare.sum()
        carries = _multinomial(a_rush, _dirichlet_share(rshare, P.kappa_car, n, rng), rng)
        rush_yd = np.zeros((n, R))
        for k in range(R):
            if k < len(rush_list):
                p, att = rush_list[k]
                ypc_r = p.rush_yds / att if att > 0 and p.rush_yds > 0 else POS_RUSH_YPC.get(p.pos, 4.4)
            else:
                ypc_r = 5.0
            scale = ypc_r * np.exp(P.ypc_r_eff * rush_eff[t] + P.ypc_r_sd * rng.standard_normal(n) - 0.5 * (P.ypc_r_sd ** 2 + P.ypc_r_eff ** 2)) / P.rush_shape
            rush_yd[:, k] = np.where(carries[:, k] > 0, rng.gamma(np.maximum(P.rush_shape * carries[:, k], 1e-9), scale), 0.0)
        ph[t] = dict(rec_list=rec_list, rush_list=rush_list, shares=shares, rshare=rshare, base_pass=base_pass, base_rush=base_rush, a_pass=a_pass, a_rush=a_rush,
                     targets=targets, rec_ct=rec_ct, rec_yd=rec_yd, carries=carries, rush_yd=rush_yd, pass_yds=rec_yd.sum(1), rush_yds=rush_yd.sum(1))

    # ---- phase 2: scoring from the yards and the script, then the touchdowns
    score = {}
    for t, spec in teams.items():
        d = ph[t]
        y_hat = _standardise(d["pass_yds"] + d["rush_yds"])
        resid = math.sqrt(max(1 - P.a_yards ** 2 - P.a_script ** 2, 0.05))
        z = P.a_yards * y_hat + P.a_script * script[t] + resid * rng.standard_normal(n)
        score[t] = _score(spec.implied, z / z.std(), rng, P)
    pts = {t: score[t][2] for t in teams}

    stats: dict = {}
    team_out: dict = {}
    pos: dict = {}
    team_of: dict = {}
    opp = {home.abbr: away.abbr, away.abbr: home.abbr}
    for t, spec in teams.items():
        d = ph[t]
        td_t, fg_t, pts_t = score[t]
        margin = pts[t] - pts[opp[t]]
        rec_list, rush_list = d["rec_list"], d["rush_list"]
        a_pass, a_rush, carries, rec_ct = d["a_pass"], d["a_rush"], d["carries"], d["rec_ct"]
        n_rec = len(rec_list)

        rec_td_w = np.array([max(p.rec_td, 0.005) for p, _ in rec_list] + [0.04 * max(sum(p.rec_td for p, _ in rec_list), 0.5)], float)
        rush_td_w = np.array([max(p.rush_td, 0.003) for p, _ in rush_list] + [0.03 * max(sum(p.rush_td for p, _ in rush_list), 0.4)], float)
        pi0 = float(np.clip(rec_td_w.sum() / (rec_td_w.sum() + rush_td_w.sum()), 0.35, 0.80))
        ratio = np.log(np.maximum(d["pass_yds"], 1.0) / max(d["pass_yds"].mean(), 1.0)) - np.log(np.maximum(d["rush_yds"], 1.0) / max(d["rush_yds"].mean(), 1.0))
        pi = _expit(_logit(pi0) + P.td_pass_link * ratio)
        td_pass = rng.binomial(td_t, pi)
        td_rush = td_t - td_pass
        rec_exp = np.maximum(d["shares"] * d["base_pass"] * 0.64, 0.3)
        wr = rec_td_w[None, :] * rec_ct / rec_exp[None, :]
        wr_sum = wr.sum(1)
        td_pass = np.where(wr_sum > 0, td_pass, 0)
        rec_td = _alloc_capped(td_pass, wr / np.maximum(wr_sum, 1e-12)[:, None], rec_ct, rng)
        td_pass = rec_td.sum(1)
        car_exp = np.maximum(d["rshare"] * d["base_rush"], 0.3)
        wc = rush_td_w[None, :] * carries / car_exp[None, :]
        wc_sum = wc.sum(1)
        td_rush = np.where(wc_sum > 0, td_rush, 0)
        rush_td = _alloc_capped(td_rush, wc / np.maximum(wc_sum, 1e-12)[:, None], carries, rng)
        td_rush = rush_td.sum(1)

        comp = rec_ct.sum(1)
        qb = starting_qb(spec)
        p_int = 0.025
        if qb is not None and qb.pass_att > 5 and qb.ints > 0:
            p_int = float(np.clip(qb.ints / qb.pass_att, 0.005, 0.06))
        ints = rng.binomial(a_pass, np.clip(p_int * np.exp(-P.int_eff * eff[t] + P.int_script * script[t]), 0.002, 0.2))

        for k, (p, _) in enumerate(rec_list):
            s = stats.setdefault(p.name, {})
            s["tgt"], s["rec"], s["rec_yds"], s["rec_td"] = d["targets"][:, k], rec_ct[:, k], d["rec_yd"][:, k], rec_td[:, k]
        for k, (p, _) in enumerate(rush_list):
            s = stats.setdefault(p.name, {})
            s["rush_att"], s["rush_yds"], s["rush_td"] = carries[:, k], d["rush_yd"][:, k], rush_td[:, k]
        zero, zero_i = np.zeros(n), np.zeros(n, dtype=np.int64)
        for p in spec.players:
            s = stats.setdefault(p.name, {})
            pos[p.name], team_of[p.name] = p.pos, t
            for key in ("tgt", "rec", "rec_td", "rush_att", "rush_td", "pass_att", "comp", "pass_td", "int"):
                s.setdefault(key, zero_i)
            for key in ("rec_yds", "rush_yds", "pass_yds"):
                s.setdefault(key, zero)
        if qb is not None:
            s = stats[qb.name]
            att_q, comp_q, int_q, yds_q = a_pass, comp, ints, d["pass_yds"]
            if P.qb_pin and qb.pass_att >= P.qb_pin_min * d["base_pass"]:  # the starter throws his projected share of the team's passes (relievers throw the rest), for as many yards as he is projected for
                u = rng.random(n)                                       # one draw per path: a smaller count never exceeds a larger one, so completions stay within attempts
                k_att = min(qb.pass_att / d["base_pass"], 1.0)
                att_q, comp_q, int_q = (np.floor(x * k_att + u).astype(np.int64) for x in (a_pass, comp, ints))
                comp_q = np.maximum(comp_q, td_pass)                    # every passing touchdown is a completion by the starter
                att_q = np.maximum(att_q, comp_q)
                if qb.pass_yds > 0:
                    yds_q = d["pass_yds"] * min(qb.pass_yds / max(float(d["pass_yds"].mean()), 1e-9), 1.0)
            s["pass_att"], s["comp"], s["pass_yds"], s["pass_td"], s["int"] = att_q, comp_q, yds_q, td_pass, int_q
        team_out[t] = {"points": pts_t, "td": td_t, "fg": fg_t, "pass_att": a_pass, "rush_att": a_rush, "margin": margin, "pass_yds": d["pass_yds"], "pass_td": td_pass,
                       "rush_td": td_rush, "rush_yds": d["rush_yds"], "comp": comp}

    for s in stats.values():                                        # an anytime-touchdown flag for every player
        s["any_td"] = ((s["rush_td"] + s["rec_td"]) > 0).astype(np.int64)
    return SimResult(n=n, stats=stats, team=team_out, pos=pos, team_of=team_of)
