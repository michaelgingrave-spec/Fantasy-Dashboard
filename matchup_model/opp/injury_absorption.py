"""How much of a hurt teammate's vacated usage do the healthy teammates really absorb?  (the evidence behind model.INJ_REDIST)

    python -m matchup_model.opp.injury_absorption        # writes matchup_model/injury_absorption_report.md

Unit 'rec' = share of the team's targets among WR/TE/RB, unit 'rush' = share of the team's carries among RBs. For every team-week with an OUT designation that
vacates >= MIN_V of the unit's trailing (as-of, EWMA of the last 6 team games) share, every healthy teammate (no designation) who actually took offensive snaps
that week is one observation: his actual share against his trailing share and against what the model's proportional rule predicted (B). kappa is the fraction of that
rule's boost that is real, fit by least squares on the other seasons and scored on the held-out one, separately for NEW absences (the Out player appeared in the
team's last game: it is news) and ONGOING ones (he had already missed it, so his teammates' trailing shares already contain part of it).

What the actual share is compared with. A rotation player's trailing share overstates what he gets next week even when nobody is hurt: on team-weeks with no Out or
Doubtful player on the report, the teammates' actual share is only about 0.9 of the trailing one (regression to the mean, depth players taking snaps). The model's scale
constants (REC_SCALE, RB_SCALE) exist to absorb exactly that, so the no-injury expectation is beta x s, with beta learned on those clean weeks of the other seasons, and
kappa is fit on what is left:   a - beta s = kappa x beta (B - s).   A first version of this analysis compared with the raw trailing share (beta = 1) and read about 0.14
for receiving instead of ~0.45: the usual fall-back was counted against the boost. `kappa_raw` keeps that reading so the difference stays visible.

Teammates who did not play are left out. Players who were released, on IR or inactive with no designation still carry a trailing share and an actual of zero, which
fakes NEGATIVE absorption (a first version of this analysis found teammates losing usage when a starter was out for exactly that reason).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dfs.names import normalize_name
from matchup_model.opp import data as D
from matchup_model.opp import injuries as I

REPORT = Path(__file__).resolve().parents[1] / "injury_absorption_report.md"
MIN_V = 0.12                 # the Out players must vacate at least this much of the unit's trailing share
POS = ("WR", "TE", "RB")
FIRST_SEASON = 2022
BOOT = 400                   # team-week bootstrap draws for the kappa interval


def _rotation(pw: pd.DataFrame, col: str, team: str, season: int, week: int):
    """(share, position, name key, played-last-game set) for the team's rotation entering (season, week): the same trailing window and cut-offs as injuries.py."""
    prior = pw[(pw.season < season) | ((pw.season == season) & (pw.week < week))]
    if prior.empty:
        return None
    gw = prior[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).tail(I._TRAIL)
    recent = prior.merge(gw, on=["season", "week"])
    last = gw.iloc[-1]
    in_last = set(recent[(recent.season == last.season) & (recent.week == last.week)].player_id)
    sh, pos, names = {}, {}, {}
    for pid, g in recent.groupby("player_id"):
        if len(g) < 2 and float(g[col].mean()) < 0.10:
            continue
        s = I._ewma(g[col].to_numpy())
        if s >= I._MIN_SHARE:
            sh[pid], pos[pid], names[pid] = s, g["position"].iloc[-1], normalize_name(g["player_display_name"].iloc[-1])
    return (sh, pos, names, in_last) if sh else None


def _context(unit: str, base: pd.DataFrame, snaps: pd.DataFrame):
    col, positions = I._UNIT_COL[unit], I._UNIT_POS[unit]
    pw = base[base.position.isin(positions)]
    by_team = {t: g for t, g in pw.groupby("team")}
    actual = {(s, w, t, p): float(v) for s, w, t, p, v in zip(pw.season, pw.week, pw.team, pw.player_id, pw[col])}
    on = snaps[snaps.offense_snaps > 0]
    played = set(zip(on.season, on.week, on.team.map(D.canon_team), on.name_key))
    return col, by_team, actual, played


def observations(unit: str, base: pd.DataFrame, inj: pd.DataFrame, snaps: pd.DataFrame, min_v: float = MIN_V) -> pd.DataFrame:
    """One row per healthy teammate who played in a team-week where an Out player vacates >= min_v of the unit's trailing share: his trailing share `s`, actual share
    `a`, the model's proportional prediction `B` (injuries.py's rule at scale 1.0), the vacated share split into `V_new` / `V_old`, and `V_<pos>` by the Out
    players' positions. `inj` needs lower-case `report_status`; `snaps` needs season, week, team, name_key, offense_snaps."""
    col, by_team, actual, played = _context(unit, base, snaps)
    rows = []
    keys = inj[inj.report_status.eq("out")][["season", "week", "team"]].drop_duplicates()
    for season, week, team in keys.itertuples(index=False):
        team = D.canon_team(team)
        if team not in by_team:
            continue
        rot = _rotation(by_team[team], col, team, season, week)
        if rot is None:
            continue
        sh, pos, names, in_last = rot
        rep = inj[(inj.season == season) & (inj.week == week) & (inj.team.map(D.canon_team) == team)]
        status = dict(zip(rep.gsis_id, rep.report_status))
        out = {p for p in sh if status.get(p) == "out"}
        V = sum(sh[p] for p in out)
        healthy = {p for p in sh if status.get(p, "") == "" and (season, week, team, names[p]) in played}
        if V < min_v or not healthy:
            continue
        V_new = sum(sh[p] for p in out if p in in_last)
        key = max(out, key=lambda p: sh[p])                              # the Out player who vacates the most: the one a with / without split is taken against
        avail = {p: I.play_prob(status.get(p, "")) for p in sh}
        vacated = sum(sh[p] * (1.0 - avail[p]) for p in sh)
        avail_share = sum(sh[p] * avail[p] for p in sh)
        vac_pos = {q: sum(sh[p] for p in out if pos[p] == q) for q in POS}
        for p in healthy:
            B = min(avail[p] * sh[p] * (1.0 + vacated / avail_share), I._MULT_CAP * sh[p]) if avail_share > 1e-9 else sh[p]
            rows.append({"unit": unit, "season": season, "week": week, "team": team, "pid": p, "pos": pos[p], "s": sh[p], "a": actual.get((season, week, team, p), 0.0),
                         "V": V, "V_new": V_new, "V_old": V - V_new, "B": B, "key": key, "name": names[p], "key_name": names[key], **{f"V_{q}": vac_pos[q] for q in POS}})
    return pd.DataFrame(rows)


def clean_weeks(unit: str, base: pd.DataFrame, inj: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """Teammates who played in team-weeks where nobody on the team's report is Out or Doubtful (Questionable allowed): their trailing share `s` and actual share `a`
    with no absence to explain a difference. The usual fall-back of a share from its trailing value is measured here."""
    col, by_team, actual, played = _context(unit, base, snaps)
    hurt = inj[inj.report_status.isin(["out", "doubtful"])]
    hurt = set(zip(hurt.season, hurt.week, hurt.team.map(D.canon_team)))
    rows = []
    games = base[base.position.isin(I._UNIT_POS[unit])][["season", "week", "team"]].drop_duplicates()
    for season, week, team in games.itertuples(index=False):
        if (season, week, team) in hurt or team not in by_team:
            continue
        rot = _rotation(by_team[team], col, team, season, week)
        if rot is None:
            continue
        sh, pos, names, _ = rot
        for p in sh:
            if (season, week, team, names[p]) in played:
                rows.append({"unit": unit, "season": season, "week": week, "team": team, "pos": pos[p], "s": sh[p], "a": actual.get((season, week, team, p), 0.0)})
    return pd.DataFrame(rows)


def leakage(clean: pd.DataFrame, seasons) -> dict[int, float]:
    """beta per season = actual / trailing share on the clean weeks of the OTHER seasons (the usual fall-back, learned without looking at the season it is applied to)."""
    out = {}
    for s_ in seasons:
        tr = clean[clean.season != s_] if len(clean) else clean
        if tr.empty:
            out[int(s_)] = 1.0
            continue
        out[int(s_)] = float(tr.a.sum() / tr.s.sum()) if tr.s.sum() > 0 else 1.0
    return out


def new_absences(obs: pd.DataFrame) -> pd.DataFrame:
    return obs[obs.V_new >= MIN_V]


def ongoing_absences(obs: pd.DataFrame) -> pd.DataFrame:
    return obs[(obs.V_new < 0.03) & (obs.V_old >= MIN_V)]


def _k(d: pd.DataFrame) -> float:
    den = float((d.adj ** 2).sum())
    return float((d.adj * d.net).sum() / den) if den > 0 else 0.0


def fit_kappa(obs: pd.DataFrame) -> dict:
    """kappa = least-squares share of the model's boost (after the usual fall-back) that is real:  a - beta s = kappa x beta (B - s).  Fit on the other seasons and scored
    on the held-out one (leave-one-season-out); the interval is a team-week bootstrap of the pooled fit. `obs` may carry a per-row `beta` (default 1)."""
    beta = obs["beta"] if "beta" in obs else pd.Series(1.0, index=obs.index)
    d = obs.assign(beta=beta, adj=beta * (obs.B - obs.s), net=obs.a - beta * obs.s, base=beta * obs.s)
    res = {"A": [], "B": [], "K": []}
    ys, kappas = [], {}
    for s_ in sorted(d.season.unique()):
        tr, te = d[d.season != s_], d[d.season == s_]
        k = _k(tr)
        kappas[int(s_)] = round(k, 2)
        res["A"].append(te.base.to_numpy()); res["B"].append((te.beta * te.B).to_numpy()); res["K"].append((te.base + k * te.adj).to_numpy()); ys.append(te.a.to_numpy())
    y = np.concatenate(ys)
    rmse = {k: 100 * float(np.sqrt(np.mean((y - np.concatenate(v)) ** 2))) for k, v in res.items()}
    groups = [g for _, g in d.groupby(["season", "week", "team"])]
    rng = np.random.default_rng(7)
    boot = [_k(pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])) for _ in range(BOOT)]
    g = d.assign(gain=d.a - d.s).groupby(["season", "week", "team"]).agg(gain=("gain", "sum"), V=("V", "first"))
    net = d.assign(g=d.net)
    V = float(g.V.sum())
    raw = d.assign(adj=d.B - d.s, net=d.a - d.s)
    return {"n": len(d), "team_weeks": int(len(g)), "mean_vacated": float(g.V.mean()), "beta": float(d.beta.mean()),
            "kappa": _k(d), "kappa_ci": (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))), "kappa_raw": _k(raw),
            "kappa_by_season": kappas, "rmse": rmse,
            "gain_of_vacated": float(g.gain.sum() / V), "net_gain_of_vacated": float(net.g.sum() / V),
            "gain_by_pos": {q: float(net[net.pos == q].g.sum() / V) for q in POS}}


def hurt_who_played(unit: str, base: pd.DataFrame, inj: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """Players designated Questionable / Doubtful who still took snaps: his actual share over his trailing share. The model scales his OWN projection by P(plays)
    (0.70 / 0.25), but a prop line only exists if he is expected to play (a bet on a player who does not is void), so the number to compare is the share GIVEN that he plays."""
    col, by_team, actual, played = _context(unit, base, snaps)
    rows = []
    keys = inj[inj.report_status.isin(["questionable", "doubtful"])][["season", "week", "team"]].drop_duplicates()
    for season, week, team in keys.itertuples(index=False):
        team = D.canon_team(team)
        if team not in by_team:
            continue
        rot = _rotation(by_team[team], col, team, season, week)
        if rot is None:
            continue
        sh, pos, names, _ = rot
        rep = inj[(inj.season == season) & (inj.week == week) & (inj.team.map(D.canon_team) == team)]
        for pid, st in zip(rep.gsis_id, rep.report_status):
            if pid in sh and st in ("questionable", "doubtful"):
                rows.append({"unit": unit, "status": st, "season": season, "played": (season, week, team, names[pid]) in played, "s": sh[pid],
                             "a": actual.get((season, week, team, pid), 0.0)})
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame) -> list[str]:
    out = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    return out + ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]


def build_report(fits: dict, own: pd.DataFrame, shipped: dict) -> str:
    L = ["# How much of a vacated share do the healthy teammates really absorb?", "",
         "Evidence behind `model.INJ_REDIST` (model v4). 2022-26 nflverse box scores, snap counts and injury reports; teammates who played only; leave-one-season-out. "
         "`injuries.py` used to hand the whole vacated share to the healthy teammates in proportion to their own share (a scale of 1.0). The unit is points of the "
         "team's target share (rec) or carry share (rush); RMSE of the teammates' shares.", "",
         "The no-injury expectation is `beta x` the trailing share, with `beta` the usual fall-back of an actual share from its trailing value on weeks with nobody Out or "
         "Doubtful on the team's report (learned on the other seasons); kappa is the fraction of the model's boost, `beta (B - s)`, that shows up beyond that. "
         "`kappa raw` is the same fit against the plain trailing share (beta = 1), which counts the usual fall-back against the boost and reads too low.", ""]
    for unit, label in (("rec", "Receiving (target share, WR/TE/RB)"), ("rush", "Rushing (carry share, RB)")):
        L += [f"## {label}", ""]
        rows = []
        for kind in ("new", "ongoing"):
            f = fits[(unit, kind)]
            if f is None:
                rows.append({"absence": kind, "team-weeks": 0}); continue
            lo, hi = f["kappa_ci"]
            rows.append({"absence": "NEW (played the team's last game)" if kind == "new" else "ONGOING (already missed it)", "team-weeks": f["team_weeks"], "teammates": f["n"],
                         "mean vacated": f"{f['mean_vacated']:.2f}", "beta": f"{f['beta']:.2f}", "teammates' gain beyond beta / vacated": f"{f['net_gain_of_vacated']:+.2f}",
                         "kappa (95% CI)": f"{f['kappa']:.2f} ({lo:+.2f} to {hi:+.2f})", "kappa raw": f"{f['kappa_raw']:.2f}",
                         "kappa by held-out season": ", ".join(f"{s} {k:+.2f}" for s, k in f["kappa_by_season"].items()),
                         "RMSE no boost": f"{f['rmse']['A']:.2f}", "RMSE the old full boost": f"{f['rmse']['B']:.2f}", "RMSE damped by kappa": f"{f['rmse']['K']:.2f}"})
        L += _md(pd.DataFrame(rows).fillna("")) + [""]
        f = fits[(unit, "new")]
        if f:
            L += ["Who picked up a NEW absence's vacated share beyond the usual fall-back, by position of the recipient: " + ", ".join(f"{q} {v:+.2f}" for q, v in f["gain_by_pos"].items()) +
                  " (the rest went to depth players outside the rotation).", ""]
    r, u = shipped["rec"], shipped["rush"]
    L += ["## What was shipped", "", f"`model.INJ_REDIST` = `{shipped}`: the kappa fitted above, rounded to two places. Receiving {r['new']:.2f} for a new absence and {r['ongoing']:.2f} "
          f"for an ongoing one, rushing {u['new']:.2f} / {u['ongoing']:.2f}. An ongoing absence is already partly in the teammates' trailing shares, so it earns less than a new "
          "one. The scale constants (REC_SCALE, RB_SCALE) were re-learned under it, because the old full boost had inflated the average involved player's projection and the "
          "old scales had absorbed that. Re-run this module after more weeks and compare.", ""]
    if own is not None and len(own):
        L += ["## Not changed in v4: the hurt player's OWN projection", "",
              "The model multiplies his own usage by P(plays) (Questionable 0.70, Doubtful 0.25). A prop line only exists if he is expected to play, and a bet on a "
              "player who does not is void, so the comparison that matters is his share GIVEN that he plays:", ""]
        rows = []
        for (unit, status), g in own.groupby(["unit", "status"]):
            p = g[g.played]
            rows.append({"unit": unit, "designation": status, "players": len(g), "played": f"{100 * g.played.mean():.0f}%", "actual / trailing share when he played":
                         f"{p.a.sum() / p.s.sum():.2f}" if len(p) and p.s.sum() > 0 else "", "the model's own multiplier": "0.70" if status == "questionable" else "0.25"})
        L += _md(pd.DataFrame(rows)) + ["", "Against the usual fall-back (beta, about 0.9) a ratio near 0.86-0.88 is about what a healthy player gets, so the model's 0.70 under-projects Questionable "
                                        "players who play (a candidate for the next version; not changed here because it also feeds the DFS screens, where P(plays) matters).", ""]
    return "\n".join(L) + "\n"


def run(base: pd.DataFrame | None = None, inj: pd.DataFrame | None = None, snaps: pd.DataFrame | None = None) -> tuple[dict, pd.DataFrame]:
    base = I._shares_base() if base is None else base
    inj = D.injuries() if inj is None else inj
    inj = inj.assign(report_status=inj.report_status.fillna("").str.lower())
    snaps = D.snaps() if snaps is None else snaps
    fits, own = {}, []
    for unit in ("rec", "rush"):
        obs = observations(unit, base, inj, snaps)
        if obs.empty:
            fits[(unit, "new")] = fits[(unit, "ongoing")] = None
            continue
        obs = obs[obs.season >= FIRST_SEASON]
        clean = clean_weeks(unit, base, inj, snaps)
        clean = clean[clean.season >= FIRST_SEASON] if len(clean) else clean
        obs = obs.assign(beta=obs.season.map(leakage(clean, sorted(obs.season.unique()))))
        for kind, sel in (("new", new_absences(obs)), ("ongoing", ongoing_absences(obs))):
            fits[(unit, kind)] = fit_kappa(sel) if sel[["season", "week", "team"]].drop_duplicates().shape[0] >= 20 else None
        own.append(hurt_who_played(unit, base, inj, snaps))
    own = pd.concat(own, ignore_index=True) if own else pd.DataFrame()
    return fits, (own[own.season >= FIRST_SEASON] if len(own) else own)


def main() -> None:
    from matchup_model.opp import model as M
    fits, own = run()
    txt = build_report(fits, own, M.INJ_REDIST)
    REPORT.write_text(txt, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(txt + f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
