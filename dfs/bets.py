"""A personal prop-bet log + self-calibration.

Append bets as you place them (from the Prop Edges screen or the Bet Log form), grade
them later from nflverse box scores, then look at hit rate / ROI **bucketed by the edge
you had** — the point is to learn which edge range actually cashes.

Storage: data/dfs/bets/bet_log.csv (committed, so it syncs to the deployed app and across
machines). Grading and analysis need the nflverse cache (matchup_model/opp/data).
"""
from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from dfs.config import DATA
from dfs.names import normalize_name

LOG_PATH = DATA / "bets" / "bet_log.csv"

COLUMNS = ["bet_id", "logged_at", "season", "week", "event", "player", "market", "side",
           "line", "odds", "book", "stake", "our_proj", "edge_toward", "edge_pct_toward",
           "z", "conf", "ev_pct", "close_line", "result", "actual", "payout", "graded_at", "note"]

# tier order for display — must match dfs.props.conf_label's tier names. NOT re-derived
# from raw |z| here: conf_label applies a per-market offset (rush yds/att need more z; pass
# att never qualifies), so bucketing by raw z would disagree with what the app actually
# shows/filters on for those markets. Always bucket by the stored `conf` column instead.
CONF_ORDER = ["—", "lean", "solid", "strong", "high"]

# our market label -> nflverse player_weeks column
_STAT_COL = {"rec yds": "receiving_yards", "receptions": "receptions", "rush yds": "rushing_yards",
             "rush att": "carries", "pass yds": "passing_yards", "pass TD": "passing_tds",
             "pass att": "attempts"}


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def load() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return _empty()
    df = pd.read_csv(LOG_PATH, dtype={"season": "Int64", "week": "Int64"})
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    for c in ("result", "side", "market", "book", "note", "graded_at", "conf"):
        df[c] = df[c].fillna("").astype(str).replace("nan", "")
    return df[COLUMNS]


def _save(df: pd.DataFrame) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(LOG_PATH, index=False)


def add_bet(*, season: int, week: int, event: str, player: str, market: str, side: str,
            line: float, odds: int, book: str = "", stake: float = 1.0,
            our_proj: float | None = None, ev_pct: float | None = None,
            close_line: float | None = None, note: str = "") -> str:
    """Append one bet. `side` is 'OVER' or 'UNDER'. Returns the new bet_id."""
    side = side.upper().strip()
    edge_toward = z = conf = None
    if our_proj is not None:
        edge_toward = (our_proj - line) if side == "OVER" else (line - our_proj)
        try:
            from dfs.props import _NICE, _sigma, conf_label
            comp = next((k for k, v in _NICE.items() if v == market), market)
            z = edge_toward / _sigma(comp, line)
            conf = conf_label(z, comp)
        except Exception:  # noqa: BLE001
            z = conf = None
    row = {
        "bet_id": uuid.uuid4().hex[:8], "logged_at": date.today().isoformat(),
        "season": int(season), "week": int(week), "event": event, "player": player,
        "market": market, "side": side, "line": float(line), "odds": int(odds),
        "book": book, "stake": float(stake),
        "our_proj": None if our_proj is None else round(float(our_proj), 2),
        "edge_toward": None if edge_toward is None else round(edge_toward, 2),
        "edge_pct_toward": None if (edge_toward is None or not line) else round(100 * edge_toward / line, 1),
        "z": None if z is None else round(z, 2),
        "conf": conf,
        "ev_pct": None if ev_pct is None else round(float(ev_pct), 1),
        "close_line": close_line, "result": "", "actual": None, "payout": None,
        "graded_at": "", "note": note,
    }
    df = pd.concat([load(), pd.DataFrame([row])], ignore_index=True)
    _save(df)
    return row["bet_id"]


def delete_bet(bet_id: str) -> None:
    df = load()
    _save(df[df["bet_id"] != bet_id])


def _payout(odds: float, result: str, stake: float) -> float:
    if result == "win":
        return stake * (odds / 100.0 if odds > 0 else 100.0 / abs(odds))
    if result == "loss":
        return -stake
    return 0.0


def grade(force: bool = False) -> tuple[pd.DataFrame, int]:
    """Fill result/actual/payout for bets whose game has box-score data. Returns
    (updated_log, n_graded)."""
    df = load()
    if df.empty:
        return df, 0
    try:
        from matchup_model.opp import data as D
        pw = D.player_weeks()
    except Exception:
        return df, 0
    pw = pw.assign(nk=pw["player_display_name"].map(normalize_name))
    n = 0
    for i, r in df.iterrows():
        res = "" if pd.isna(r.get("result")) else str(r.get("result")).strip()
        if not force and res in ("win", "loss", "push"):
            continue
        col = _STAT_COL.get(str(r["market"]))
        if col is None or pd.isna(r["season"]) or pd.isna(r["week"]):
            continue
        hit = pw[(pw.nk == normalize_name(str(r["player"]))) &
                 (pw.season == int(r["season"])) & (pw.week == int(r["week"]))]
        if hit.empty:
            continue
        actual = float(hit[col].sum())
        line = float(r["line"])
        side = str(r["side"]).upper()
        result = "push" if actual == line else (
            "win" if ((actual > line) == (side == "OVER")) else "loss")
        df.at[i, "actual"] = round(actual, 1)
        df.at[i, "result"] = result
        df.at[i, "payout"] = round(_payout(float(r["odds"]), result, float(r["stake"])), 3)
        df.at[i, "graded_at"] = date.today().isoformat()
        n += 1
    if n:
        _save(df)
    return df, n


# ── analysis ────────────────────────────────────────────────────────────────


def _summ(g: pd.DataFrame) -> dict:
    played = g[g["result"].isin(["win", "loss", "push"])]
    dec = played[played["result"].isin(["win", "loss"])]
    staked = float(dec["stake"].sum())
    won = float(played["payout"].fillna(0).sum())
    clv = played.dropna(subset=["close_line"])
    clv_beat = None
    if len(clv):
        # beating the number: an OVER bettor wants the line to have RISEN by close (they
        # locked in the cheaper number before the market agreed with them); an UNDER
        # bettor wants it to have FALLEN.
        b = np.where(clv["side"].str.upper() == "OVER",
                     clv["line"] < clv["close_line"], clv["line"] > clv["close_line"])
        clv_beat = float(np.mean(b))
    return {
        "bets": len(g), "graded": len(played),
        "win%": round(100 * (dec["result"] == "win").mean(), 1) if len(dec) else None,
        "units": round(won, 2), "staked": round(staked, 2),
        "ROI%": round(100 * won / staked, 1) if staked else None,
        "CLV beat%": round(100 * clv_beat, 0) if clv_beat is not None else None,
    }


def by_edge_bucket(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Your logged bets bucketed by confidence tier (matches the `conf` column shown on
    Prop Edges — NOT a re-derivation from raw |z|, which would disagree for rush props)."""
    df = load() if df is None else df
    g = df[df["result"].isin(["win", "loss", "push"])].copy()
    if g.empty or "conf" not in g.columns:
        return pd.DataFrame()
    rows = []
    for tier in CONF_ORDER:
        sub = g[g["conf"] == tier]
        if sub.empty:
            continue
        rows.append({"conf": tier, **_summ(sub)})
    return pd.DataFrame(rows)


def by_key(df: pd.DataFrame | None = None, key: str = "market") -> pd.DataFrame:
    df = load() if df is None else df
    g = df[df["result"].isin(["win", "loss", "push"])]
    if g.empty:
        return pd.DataFrame()
    return pd.DataFrame([{key: k, **_summ(sub)} for k, sub in g.groupby(key)]).sort_values(
        "units", ascending=False)


def overall(df: pd.DataFrame | None = None) -> dict:
    df = load() if df is None else df
    return _summ(df)


# ── line history: grade EVERY pulled line (not just logged bets) vs the real book line ──
_PRICE = -110  # assumed juice for the notional ROI on ungraded/graded pulled lines


def _roi_at(price: int, wins: int, losses: int) -> float:
    dec = price / 100.0 if price > 0 else 100.0 / abs(price)
    n = wins + losses
    return (wins * dec - losses) / n if n else np.nan


def load_line_history() -> pd.DataFrame:
    from dfs.props import LINE_HISTORY_PATH
    if not LINE_HISTORY_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(LINE_HISTORY_PATH)
    df["result"] = df.get("result", "").fillna("").astype(str).replace("nan", "")
    return df


def grade_line_history(force: bool = False) -> tuple[pd.DataFrame, int]:
    """Fill actual/result for pulled lines whose game has box-score data, graded vs the
    real book line (`line` column). Returns (df, n_graded)."""
    from dfs.props import LINE_HISTORY_PATH
    df = load_line_history()
    if df.empty:
        return df, 0
    try:
        from matchup_model.opp import data as D
        pw = D.player_weeks().assign(nk=lambda d: d["player_display_name"].map(normalize_name))
    except Exception:
        return df, 0
    n = 0
    for i, r in df.iterrows():
        if not force and str(r.get("result") or "") in ("win", "loss", "push"):
            continue
        col = _STAT_COL.get(str(r["market"]))
        if col is None or pd.isna(r.get("season")) or pd.isna(r.get("week")):
            continue
        hit = pw[(pw.nk == normalize_name(str(r["player"]))) &
                 (pw.season == int(r["season"])) & (pw.week == int(r["week"]))]
        if hit.empty:
            continue
        actual, line, side = float(hit[col].sum()), float(r["line"]), str(r["lean"]).upper()
        df.at[i, "actual"] = round(actual, 1)
        df.at[i, "result"] = ("push" if actual == line else
                              "win" if ((actual > line) == (side == "OVER")) else "loss")
        n += 1
    if n:
        df.to_csv(LINE_HISTORY_PATH, index=False)
    return df, n


def line_history_buckets(df: pd.DataFrame | None = None, min_n: int = 0) -> pd.DataFrame:
    """Win% + notional ROI@-110 by confidence tier, graded vs the real book line. Buckets
    by the stored `conf` column (matches Prop Edges) — not re-derived from raw |z|, which
    disagrees with `conf` for rush props (per-market offset) and pass att (always excluded).
    Shows every tier that has at least one graded row by default (`min_n=0`) — early season
    that means small samples, but hiding a tier outright is worse than a visible small n."""
    df = load_line_history() if df is None else df
    if df.empty or "conf" not in df.columns:
        return pd.DataFrame()
    g = df[df["result"].isin(["win", "loss", "push"])].copy()
    if g.empty:
        return pd.DataFrame()
    rows = []
    for tier in CONF_ORDER:
        s = g[g["conf"] == tier]
        if s.empty:
            continue
        w, l = int((s.result == "win").sum()), int((s.result == "loss").sum())
        if w + l < min_n:
            continue
        rows.append({"conf": tier, "n": len(s),
                     "win%": round(100 * w / (w + l), 1) if (w + l) else None,
                     "ROI@-110%": round(100 * _roi_at(_PRICE, w, l), 1) if (w + l) else None})
    return pd.DataFrame(rows)


def line_history_by_market(df: pd.DataFrame | None = None, min_n: int = 1) -> pd.DataFrame:
    """Win% + notional ROI@-110 by market, graded vs the real book line — every prop we've
    pulled this season (line_history.csv only ever holds the current season), not just
    ones you bet. Sorted by ROI so the best-calibrated markets sort first."""
    df = load_line_history() if df is None else df
    if df.empty or "market" not in df.columns:
        return pd.DataFrame()
    g = df[df["result"].isin(["win", "loss", "push"])].copy()
    if g.empty:
        return pd.DataFrame()
    rows = []
    for mk, s in g.groupby("market"):
        w, l = int((s.result == "win").sum()), int((s.result == "loss").sum())
        if w + l < min_n:
            continue
        rows.append({"market": mk, "n": len(s), "win%": round(100 * w / (w + l), 1),
                     "ROI@-110%": round(100 * _roi_at(_PRICE, w, l), 1)})
    return pd.DataFrame(rows).sort_values("ROI@-110%", ascending=False).reset_index(drop=True)


def backtest_buckets() -> pd.DataFrame:
    """The 2023-25 trailing-form-proxy calibration (no real lines needed).
    Empty until `python -m matchup_model.opp.edge_calib` has been run."""
    try:
        from matchup_model.opp.edge_calib import bucket_table
        return bucket_table()
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
