"""Anytime-TD probability calibration — is our expected-TD model any good at forecasting
whether a player scores at all?

Anytime-TD props price as a Yes/No moneyline (no point value), so this isn't a line-vs-
projection edge check like edge_calib.py — it's a straight probability calibration: convert
our model's expected-TD-count (already computed inside opp_line's rec_td/rush_td) into
P(>=1 TD) via a Poisson approximation, and check whether that forecast matches reality.

    python -m matchup_model.opp.td_calib
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from matchup_model.opp import model as M

TEST_SEASONS = [2023, 2024, 2025]
TEST_WEEKS = range(4, 19)
GROUPS = {"WR/TE": ["WR", "TE"], "RB": ["RB"]}
BUCKETS = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.30),
           (0.30, 0.45), (0.45, 1.01)]

ROWS_CSV = Path(__file__).resolve().parents[1] / "opp_td_calibration.csv"
REPORT = Path(__file__).resolve().parents[1] / "opp_td_calibration.md"


def _baseline_rate(pw: pd.DataFrame, season: int, pos_group: list[str]) -> float:
    """Flat league TD-per-game rate for the position group, from seasons strictly before
    the test season (no leakage) — the "just guess the average" comparator."""
    d = pw[(pw.season < season) & (pw.position.isin(pos_group))]
    d = d[(d.targets + d.carries) >= 2]
    if d.empty:
        return 0.08
    scored = ((d["receiving_tds"] + d["rushing_tds"]) >= 1).mean()
    return float(scored)


def run() -> pd.DataFrame:
    pw = M._pw_with_team()
    rows, t0 = [], time.time()
    for season in TEST_SEASONS:
        base = {g: _baseline_rate(pw, season, pos) for g, pos in GROUPS.items()}
        sl = pw[(pw.season == season) & (pw.week.isin(TEST_WEEKS)) &
               (pw.position.isin(["WR", "TE", "RB"]))]
        sl = sl[(sl.targets + sl.carries) >= 2]
        for _, r in sl.iterrows():
            pid, pos, wk = r["player_id"], r["position"], int(r["week"])
            group = "RB" if pos == "RB" else "WR/TE"
            out = M.opp_line(pid, pos, season, wk, by="id", injury_adj=True)
            if out.get("dk_fp") is None:
                continue
            line = out["line"]
            lam = float(line.get("rec_td", 0.0)) + float(line.get("rush_td", 0.0))
            p_model = 1.0 - np.exp(-lam)
            actual = 1 if (r["receiving_tds"] + r["rushing_tds"]) >= 1 else 0
            # segmentation axes: scoring environment, favorite/dog, and this game's actual
            # role share (not trailing) — cheap to grab since vegas_row is cached inside
            # opp_line's own team_volume call
            vr = M.vegas_row(str(r["team"]), season, wk)
            share = float(r["carry_share"]) if pos == "RB" else float(r["target_share"])
            rows.append(dict(season=season, week=wk, player=r["player_display_name"],
                             pos=pos, group=group, lam=round(lam, 3),
                             p_model=round(p_model, 4), p_base=round(base[group], 4),
                             actual=actual,
                             team_total=vr.get("team_total", np.nan),
                             spread=vr.get("spread", np.nan),
                             share=round(share, 3) if np.isfinite(share) else np.nan))
        print(f"  {season}: {len(rows)} rows ({time.time()-t0:.0f}s)")
    return pd.DataFrame(rows)


def _brier(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def _calib_table(df: pd.DataFrame) -> list[str]:
    L = ["| pred range | n | mean predicted | actual rate |", "|---|--:|--:|--:|"]
    for lo, hi in BUCKETS:
        s = df[(df.p_model >= lo) & (df.p_model < hi)]
        if len(s) < 20:
            continue
        L.append(f"| {lo:.0%}-{hi:.0%} | {len(s)} | {100*s.p_model.mean():.1f}% | "
                 f"{100*s.actual.mean():.1f}% |")
    return L


def report(df: pd.DataFrame) -> str:
    L = ["# Anytime-TD calibration — expected-TD-count -> P(>=1 TD)", ""]
    L.append(f"- {len(df)} player-games, {TEST_SEASONS} wk {TEST_WEEKS.start}-{TEST_WEEKS.stop-1}.")
    L.append("- `lam` = our model's expected TD count for the game (rec_td+rush_td from "
             "opp_line, already blends trailing rate with the Vegas-implied team total). "
             "`p_model` = 1-e^-lam. `p_base` = flat league rate for the position group "
             "(seasons strictly before the test season) — the \"just guess the average\" "
             "comparator, no player-specific info at all.")
    L.append("")
    for group in GROUPS:
        g = df[df.group == group]
        if g.empty:
            continue
        b_model = _brier(g.p_model, g.actual)
        b_base = _brier(g.p_base, g.actual)
        L.append(f"## {group}  (n={len(g)})")
        L.append("")
        L.append(f"Brier score (lower = better): **model {b_model:.4f}** vs "
                 f"**flat-average baseline {b_base:.4f}**  "
                 f"({'model wins' if b_model < b_base else 'baseline wins'}, "
                 f"{100*(b_base-b_model)/b_base:+.1f}% {'improvement' if b_model<b_base else 'worse'})")
        L.append(f"Actual score rate this sample: {100*g.actual.mean():.1f}%  ·  "
                 f"mean model p: {100*g.p_model.mean():.1f}%  ·  mean baseline p: "
                 f"{100*g.p_base.mean():.1f}%")
        L.append("")
        L.append("### Calibration — does the predicted probability match reality?")
        L.append("")
        L += _calib_table(g)
        L.append("")
        # weekly top-12 by p_model: what fraction actually scored, vs random-12 baseline
        tops, rnd = [], []
        rng = np.random.default_rng(0)
        for (s, w), gg in g.groupby(["season", "week"]):
            if len(gg) < 15:
                continue
            k = 12
            tops.append(gg.nlargest(k, "p_model")["actual"].mean())
            rnd.append(gg.sample(k, random_state=rng.integers(1e6))["actual"].mean())
        if tops:
            L.append(f"Weekly top-12 by our model: **{100*np.mean(tops):.1f}%** actually scored "
                     f"vs **{100*np.mean(rnd):.1f}%** for a random 12 (n={len(tops)} weeks)")
        L.append("")
    return "\n".join(L)


def _tercile_col(s: pd.Series) -> pd.Series:
    try:
        return pd.qcut(s, 3, labels=("low", "mid", "high"), duplicates="drop")
    except ValueError:
        return pd.Series(["mid"] * len(s), index=s.index)


def segment_report(df: pd.DataFrame) -> str:
    """Split the same Brier comparison into terciles of each axis — does the model's edge
    over the flat baseline spread evenly, or concentrate in specific situations?"""
    L = ["", "## Where does the model's edge concentrate?", "",
         "Same Brier-score check as above, cut into terciles of each axis. `dev` = "
         "|model p - baseline p|, i.e. how far the model strays from 'just guess the "
         "position average' — the cut that matters most for spotting a bettable edge, "
         "since a near-average lam prices about like the market already does.", ""]
    axes = [("team_total", "Vegas implied team total (scoring environment)"),
            ("spread", "spread (positive = this team favored)"),
            ("share", "this game's actual role share (target share WR/TE, carry share RB)"),
            ("dev", "|model p - baseline p| (model's disagreement with a flat rate)")]
    for group in GROUPS:
        g = df[df.group == group].copy()
        if g.empty:
            continue
        g["dev"] = (g.p_model - g.p_base).abs()
        L.append(f"### {group}")
        L.append("")
        for col, desc in axes:
            gg = g.dropna(subset=[col])
            if len(gg) < 60:
                continue
            gg = gg.assign(_t=_tercile_col(gg[col]))
            L.append(f"**By {desc}:**")
            L.append("")
            L.append("| tercile | n | range | Brier model | Brier baseline | improvement | actual rate |")
            L.append("|---|--:|---|--:|--:|--:|--:|")
            for t in ("low", "mid", "high"):
                s = gg[gg._t == t]
                if s.empty:
                    continue
                bm, bb = _brier(s.p_model, s.actual), _brier(s.p_base, s.actual)
                imp = 100 * (bb - bm) / bb if bb else 0.0
                L.append(f"| {t} | {len(s)} | {s[col].min():.2f} to {s[col].max():.2f} | "
                         f"{bm:.4f} | {bb:.4f} | {imp:+.1f}% | {100*s.actual.mean():.1f}% |")
            L.append("")
    return "\n".join(L)


def main():
    df = run()
    if df.empty:
        print("no rows"); return
    df.to_csv(ROWS_CSV, index=False)
    txt = report(df) + segment_report(df)
    REPORT.write_text(txt, encoding="utf-8")
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n" + txt + f"\n\nwrote {REPORT}")


if __name__ == "__main__":
    main()
