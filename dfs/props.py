"""The Odds API player-prop lines, and the edge vs our trailing-usage projection.

The free Odds API tier returns player props. `/events` is free; each
`/events/{id}/odds` call costs 1 credit *per market*, so the screen gates pulls behind
an explicit button and a short-TTL cache. Our side of the edge is
`matchup_model.project_stats.projected_line()` (trailing usage x efficiency), which is
2025-season data until the Wednesday pulls feed 2026 game logs — so a player whose role
changed for 2026 shows a fake edge. `role_ratio` flags those.
"""
from __future__ import annotations

import json
import math
import statistics as _st
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import pandas as pd
import requests

from dfs.config import DATA, ODDS_API_KEY
from dfs.names import normalize_name

# every pull appends here so the edge calibration can use *real* book lines, not just
# hand-logged bets (see dfs.bets.line_history_buckets)
LINE_HISTORY_PATH = DATA / "props" / "line_history.csv"

# the most recent pull, saved so reloading the screen (or restarting the app) doesn't
# need a fresh — and billable — pull. Local cache, not synced (like data/dfs/dk/).
LAST_PULL_CSV = DATA / "props" / "last_pull.csv"
LAST_PULL_META = DATA / "props" / "last_pull_meta.json"
LAST_PULL_ATTD_CSV = DATA / "props" / "last_pull_attd.csv"   # optional 2nd frame, see save_last_pull

_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"

# Odds API market -> the key in project_stats' reconstructed line
MARKETS = {
    "player_reception_yds": "rec_yds",
    "player_receptions": "rec",
    "player_rush_yds": "rush_yds",
    "player_rush_attempts": "rush_att",
    "player_pass_yds": "pass_yds",
    "player_pass_tds": "pass_td",
    "player_pass_attempts": "pass_att",
}
CORE_MARKETS = ["player_reception_yds", "player_receptions", "player_rush_yds",
                "player_pass_yds", "player_pass_tds"]
# anytime-TD prices as a Yes/No moneyline (no line/point) — a different shape from every
# market above, so it's pulled and priced separately (see attd_edges_from_raw) rather than
# folded into MARKETS/edges_from_raw.
ATTD_MARKET = "player_anytime_td"
BOOKS = "draftkings,fanduel"      # only pull these two books
_NICE = {"rec_yds": "rec yds", "rec": "receptions", "rush_yds": "rush yds",
         "rush_att": "rush att", "pass_yds": "pass yds", "pass_td": "pass TD",
         "pass_att": "pass att"}

# when a player pulled from the odds feed isn't in this week's projections (missing file,
# or a name-normalization mismatch), guess their position from which market is being priced
# rather than always guessing WR — a QB with no projections row would otherwise get every
# pass_yds/pass_td row silently dropped (projected_line("QB"-shaped stat) returns no
# pass-market keys when built as a WR).
_FALLBACK_POS = {"pass_yds": "QB", "pass_td": "QB", "pass_att": "QB",
                 "rush_yds": "RB", "rush_att": "RB", "rec_yds": "WR", "rec": "WR"}

# Outcome standard deviation per market, fitted as sigma = a + b*line from 2023-25
# box scores (matchup_model/opp/edge_calib). Lets us express the edge in SD units (`z`),
# which IS comparable across markets — a "+33% edge" on a 1.5 reception line and a "+7%
# edge" on a 270 pass-yd line are very different bets; in z they're ~0.6 vs ~0.35 SD.
_SIGMA_LIN = {
    "pass_td": (0.98, 0.1233), "pass_att": (13.44, -0.1425), "pass_yds": (85.32, -0.0352),
    "rec_yds": (16.28, 0.3324), "rec": (1.21, 0.2303), "rush_att": (4.53, 0.0692),
    "rush_yds": (17.76, 0.2649),
}
# our normal-model hit probability is overconfident vs realised results — shrink toward .5
_P_SHRINK = 0.60


def _sigma(comp: str, line: float) -> float:
    a, b = _SIGMA_LIN.get(comp, (max(0.5, 0.4 * abs(line)), 0.0))
    return max(a + b * float(line), 0.5)


# categorical confidence from |z|, calibrated to the 2023-25 backtest buckets:
#   —      <0.15 SD   ~52% (coin flip)
#   lean   0.15-0.30  ~54%
#   solid  0.30-0.50  ~57%
#   strong 0.50-0.80  ~57%
#   high   0.80+      ~62%
# rush yds / rush att only turn +EV around z 0.60, so they carry an offset; pass att is a
# losing market at every z, so it never gets a label.
_CONF_TIERS = ((0.80, "high"), (0.50, "strong"), (0.30, "solid"), (0.15, "lean"))
_CONF_RANK = {"—": 0, "lean": 1, "solid": 2, "strong": 3, "high": 4}
_MKT_Z_OFFSET = {"rush_yds": 0.28, "rush_att": 0.28}
_MKT_SKIP = {"pass_att"}


def _adj_z(z: float, comp: str | None = None) -> float:
    """Effective |z| after the per-market adjustment (rush needs more raw z to count;
    pass_att never counts). Shared by conf_label's tier lookup AND the p(hit) probability
    below so the displayed tier and the EV math they sit next to always agree."""
    if comp in _MKT_SKIP or z is None or not (abs(z) == abs(z)):   # NaN-safe
        return 0.0
    return max(abs(float(z)) - _MKT_Z_OFFSET.get(comp, 0.0), 0.0)


def conf_label(z: float, comp: str | None = None) -> str:
    if z is None or not (abs(z) == abs(z)):   # NaN-safe
        return "—"
    az = _adj_z(z, comp)
    for thr, lbl in _CONF_TIERS:
        if az >= thr:
            return lbl
    return "—"


# Anytime-TD confidence: there's no line/point on a Yes/No market, so this ISN'T the
# z-score system above — it's tiers on |model p - flat league-baseline p| for the position
# group. Backtested in matchup_model/opp/td_calib.py (9213 player-games, 2023-25 walk-
# forward): the model beats the flat baseline overall (Brier score: WR/TE +4.9%, RB +9.9%),
# and that edge concentrates almost entirely in the top `dev` tercile (WR/TE +11.4% there
# vs +0.7% in the bottom third; RB +18.8% vs +1.1%). Thresholds below are hand-set off
# those tercile boundaries — v1, not exhaustively fit.
_ATTD_DEV_TIERS = {
    "WR/TE": ((0.22, "high"), (0.14, "strong"), (0.08, "solid"), (0.04, "lean")),
    "RB": ((0.25, "high"), (0.16, "strong"), (0.10, "solid"), (0.05, "lean")),
}


def _attd_group(pos: str | None) -> str | None:
    pos = (pos or "").upper()
    if pos in ("WR", "TE"):
        return "WR/TE"
    if pos == "RB":
        return "RB"
    return None                      # QB anytime-TD wasn't in the backtest — skip it


@lru_cache(maxsize=4)
def _attd_baseline(group: str) -> float:
    """Flat league anytime-TD rate for the position group, over every season on file — the
    same 'just guess the average' comparator td_calib.py backtests against, just run
    through 'now' (season=2100 -> every real season counts) instead of held out for a
    walk-forward test."""
    from matchup_model.opp import data as _D
    from matchup_model.opp.td_calib import GROUPS, _baseline_rate
    return _baseline_rate(_D.player_weeks(), 2100, GROUPS[group])


def attd_conf_label(dev: float, group: str) -> str:
    if dev is None or not (abs(dev) == abs(dev)) or group not in _ATTD_DEV_TIERS:
        return "—"
    adev = abs(float(dev))
    for thr, lbl in _ATTD_DEV_TIERS[group]:
        if adev >= thr:
            return lbl
    return "—"


class PropsError(RuntimeError):
    pass


def _get(path: str, **params):
    if not ODDS_API_KEY:
        raise PropsError("No ODDS_API_KEY. Add a free key from the-odds-api.com to .env.")
    params["apiKey"] = ODDS_API_KEY
    r = requests.get(f"{_BASE}{path}", params=params, timeout=25)
    rem = r.headers.get("x-requests-remaining")
    if r.status_code == 401:
        raise PropsError("Odds API rejected the key (401 unauthorized).")
    if r.status_code == 422:
        raise PropsError(f"Odds API 422 — market not on this plan or bad params: {r.text[:180]}")
    r.raise_for_status()
    return r.json(), rem


def list_events(days_ahead: int = 8) -> tuple[list[dict], str | None]:
    """Upcoming NFL games within `days_ahead` days. Free (0 credits)."""
    data, rem = _get("/events")
    now = datetime.now(timezone.utc)
    cut = now + timedelta(days=days_ahead)
    out = []
    for e in data if isinstance(data, list) else []:
        try:
            ts = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
        except Exception:
            continue
        if now - timedelta(hours=4) <= ts <= cut:
            local = ts.astimezone()
            out.append({"id": e["id"], "commence_time": e["commence_time"],
                        "away_team": e.get("away_team", ""), "home_team": e.get("home_team", ""),
                        "is_sunday": local.weekday() == 6,
                        "label": f"{e['away_team']} @ {e['home_team']}  ·  "
                                 f"{local.strftime('%a %m/%d %I:%M %p')}"})
    return out, rem


def bulk_prop_edges(events: list[dict], markets: list[str],
                    week: int) -> tuple[pd.DataFrame, str | None]:
    """Pull + build edge rows for several events in one go (e.g. every Sunday game).
    Costs `len(events) * len(markets)` credits total. A per-event fetch failure is
    skipped, not fatal, so one bad game doesn't lose the rest of the pull."""
    frames, rem, errors = [], None, []
    for ev in events:
        try:
            raw, rem = fetch_event_odds(ev["id"], markets)
        except PropsError as e:
            errors.append(f"{ev.get('label', ev.get('id'))}: {e}")
            continue
        d = edges_from_raw(raw, week)
        if not d.empty:
            d.insert(0, "game", ev.get("label", ev.get("id", "")))
        frames.append(d)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df.empty and "conf" in df.columns:
        df = (df.assign(_rank=df["conf"].map(_CONF_RANK).fillna(0))
                .sort_values(["_rank", "p edge"], ascending=False, na_position="last")
                .drop(columns="_rank").reset_index(drop=True))
    if errors:
        df.attrs["errors"] = errors
    return df, rem


def _amer_to_prob(price: float) -> float:
    return 100 / (price + 100) if price > 0 else (-price) / ((-price) + 100)


def _amer_to_dec(price: float) -> float:
    return 1 + price / 100 if price > 0 else 1 + 100 / (-price)


def _dec_to_amer(dec: float) -> int:
    return round((dec - 1) * 100) if dec >= 2.0 else round(-100 / (dec - 1))


def parlay_odds(prices: list[int]) -> dict:
    """Combined price for a same-slip parlay of independent legs (naive product of
    decimal odds — correct for legs on different games; optimistic for same-game legs,
    which are usually correlated one way or another)."""
    dec = 1.0
    for p in prices:
        dec *= _amer_to_dec(p)
    return {"decimal": round(dec, 3), "american": _dec_to_amer(dec), "implied_from_price": 1 / dec}


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / (2 ** 0.5)))


def fetch_event_odds(event_id: str, markets: list[str]) -> tuple[dict, str | None]:
    """Raw Odds API payload for one event, DraftKings + FanDuel only. 1 credit per market."""
    return _get(f"/events/{event_id}/odds", bookmakers=BOOKS, oddsFormat="american",
                markets=",".join(markets))


def event_prop_edges(event_id: str, markets: list[str], week: int) -> tuple[pd.DataFrame, str | None]:
    """Fetch one event's props and build the edge table. See `edges_from_raw`."""
    raw, rem = fetch_event_odds(event_id, markets)
    return edges_from_raw(raw, week), rem


def edges_from_raw(raw: dict, week: int) -> pd.DataFrame:
    """One row per (player, market): consensus line, our projection, edge, the side we
    lean, the book's no-vig probability for that side, and a rough EV%."""
    from dfs.projections import load_weekly_projections
    try:
        from matchup_model.opp.blend import blended_line as projected_line, current_season
        _season = current_season()
    except Exception:  # noqa: BLE001
        from matchup_model.project_stats import projected_line
        _season = None

    # (player, market) -> {"points": [...], "over": [(price, book)], "under": [(price, book)]}
    acc: dict = {}
    for bk in raw.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            mk = mkt["key"]
            if mk not in MARKETS:
                continue
            for o in mkt.get("outcomes", []):
                key = (o.get("description", ""), mk)
                d = acc.setdefault(key, {"points": [], "over": [], "under": []})
                if o.get("point") is not None:
                    d["points"].append(o["point"])
                side = "over" if o.get("name") == "Over" else "under" if o.get("name") == "Under" else None
                if side:
                    d[side].append((o["price"], bk["key"]))

    try:
        proj = load_weekly_projections(week)
        info = {normalize_name(n): (p, float(fp)) for n, p, fp in
                zip(proj["name"], proj["pos"], proj["proj"])}
    except Exception:
        info = {}

    line_cache: dict = {}
    rows = []
    for (player, mk), d in acc.items():
        if not d["points"]:
            continue
        comp = MARKETS[mk]
        line = float(_st.median(d["points"]))
        nk = normalize_name(player)
        pos, wk_proj = info.get(nk, (_FALLBACK_POS.get(comp, "WR"), None))
        if nk not in line_cache:
            line_cache[nk] = projected_line(nk, pos, as_of_season=_season, as_of_week=week)
        pl = line_cache[nk]
        our = pl.get("line", {}).get(comp) if pl.get("fp") is not None else None
        if our is None:
            continue
        our = float(our)
        edge = our - line
        lean = "OVER" if edge > 0 else "UNDER"

        # standardized edge: how many outcome-SDs our projection sits past the line.
        # THIS is the number to compare across markets (not `edge %`).
        sigma = _sigma(comp, line)
        z = edge / sigma
        # model P(bet hits), shrunk toward .5 (the raw normal model is overconfident).
        # Uses the same market-adjusted z as conf_label so the tier and the probability
        # never disagree (a rush prop needs the same "extra" edge for both).
        p_raw = _norm_cdf(_adj_z(z, comp))
        p_hit = 0.5 + _P_SHRINK * (p_raw - 0.5)

        # best price on the leaned side + no-vig prob from the median two-way
        prices = d["over"] if lean == "OVER" else d["under"]
        best_price, best_book = max(prices, key=lambda x: _amer_to_dec(x[0])) if prices else (None, "")
        p_over = _st.median([_amer_to_prob(p) for p, _ in d["over"]]) if d["over"] else None
        p_under = _st.median([_amer_to_prob(p) for p, _ in d["under"]]) if d["under"] else None
        novig = None
        if p_over is not None and p_under is not None and (p_over + p_under):
            novig = (p_over if lean == "OVER" else p_under) / (p_over + p_under)
        p_edge = (p_hit - novig) if novig is not None else None

        role_ratio = (wk_proj / pl["fp"]) if (wk_proj and pl.get("fp")) else None
        conf = conf_label(z, comp)
        rows.append({
            "player": player, "market": _NICE.get(comp, comp),
            "line": round(line, 1), "our proj": round(our, 1),
            "conf": conf,
            "z": round(z, 2),
            "edge": round(edge, 1),
            "p(hit)%": round(100 * p_hit),
            "p edge": round(100 * p_edge, 1) if p_edge is not None else None,
            "book %": round(100 * novig) if novig is not None else None,
            "edge %": round(100 * edge / line) if line else None,
            "lean": lean,
            "best": f"{best_book} {best_price:+d}" if best_price is not None else "",
            "role_ratio": round(role_ratio, 2) if role_ratio is not None else None,
            "_rank": _CONF_RANK.get(conf, 0),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["_rank", "p edge"], ascending=False,
                            na_position="last").drop(columns="_rank").reset_index(drop=True)
    return df


def attd_edges_from_raw(raw: dict, week: int) -> pd.DataFrame:
    """One row per player for the anytime-TD (Yes/No) market. It prices as a moneyline with
    no line/point, so it can't reuse edges_from_raw's line-vs-projection math:

    * `p(model)%` = 1 - e^-lam, lam = rec_td + rush_td from the same blended projection
      used everywhere else on this screen.
    * `dev` = model prob − a flat league-average rate for the position group — the
      *backtested* signal (see attd_conf_label / matchup_model/opp/td_calib.py). `conf`
      tiers off this, not off the book price.
    * `p edge` / `book %` = model prob vs the book's own de-vigged Yes probability — the
      actual market-mispricing check. `lean` follows this (falls back to `dev`'s sign only
      when a book doesn't post a No price to de-vig against).

    QBs are skipped — td_calib.py only validated WR/TE/RB. Returns the same `conf` /
    `role_ratio` column names as edges_from_raw so confident_only/role_stable apply as-is.
    """
    from dfs.projections import load_weekly_projections
    from matchup_model.opp.blend import blended_line, current_season
    season = current_season()

    # player -> {"yes": [(price, book)], "no": [(price, book)]}
    acc: dict = {}
    for bk in raw.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            if mkt["key"] != ATTD_MARKET:
                continue
            for o in mkt.get("outcomes", []):
                side = str(o.get("name", "")).strip().lower()
                if side not in ("yes", "no"):
                    continue
                d = acc.setdefault(o.get("description", ""), {"yes": [], "no": []})
                d[side].append((o["price"], bk["key"]))

    try:
        proj = load_weekly_projections(week)
        info = {normalize_name(n): (p, float(fp)) for n, p, fp in
                zip(proj["name"], proj["pos"], proj["proj"])}
    except Exception:
        info = {}

    rows = []
    for player, d in acc.items():
        if not d["yes"]:
            continue
        nk = normalize_name(player)
        pos, wk_proj = info.get(nk, (None, None))
        group = _attd_group(pos)
        if group is None:
            continue                  # unknown position, or a QB — not backtested, skip
        pl = blended_line(nk, pos, as_of_season=season, as_of_week=week)
        if pl.get("fp") is None:
            continue
        line = pl.get("line", {})
        lam = float(line.get("rec_td", 0.0)) + float(line.get("rush_td", 0.0))
        p_model = 1.0 - math.exp(-lam)
        p_base = _attd_baseline(group)
        dev = p_model - p_base
        conf = attd_conf_label(dev, group)

        best_price, best_book = max(d["yes"], key=lambda x: _amer_to_dec(x[0]))
        p_yes = _st.median([_amer_to_prob(p) for p, _ in d["yes"]])
        p_no = _st.median([_amer_to_prob(p) for p, _ in d["no"]]) if d["no"] else None
        novig = (p_yes / (p_yes + p_no)) if (p_no is not None and (p_yes + p_no)) else p_yes
        p_edge = (p_model - novig) if novig is not None else None
        lean = "YES" if (p_edge if p_edge is not None else dev) >= 0 else "NO"

        role_ratio = (wk_proj / pl["fp"]) if (wk_proj and pl.get("fp")) else None
        rows.append({
            "player": player, "pos": pos, "conf": conf,
            "p(model)%": round(100 * p_model), "p(base)%": round(100 * p_base),
            "dev": round(100 * dev, 1),
            "book %": round(100 * novig) if novig is not None else None,
            "p edge": round(100 * p_edge, 1) if p_edge is not None else None,
            "lean": lean,
            "best": f"{best_book} {best_price:+d}",
            "role_ratio": round(role_ratio, 2) if role_ratio is not None else None,
            "_rank": _CONF_RANK.get(conf, 0),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["_rank", "p edge"], ascending=False,
                            na_position="last").drop(columns="_rank").reset_index(drop=True)
    return df


def event_attd_edges(event_id: str, week: int) -> tuple[pd.DataFrame, str | None]:
    """Fetch one event's Anytime-TD props and build the edge table. See attd_edges_from_raw."""
    raw, rem = fetch_event_odds(event_id, [ATTD_MARKET])
    return attd_edges_from_raw(raw, week), rem


def bulk_attd_edges(events: list[dict], week: int) -> tuple[pd.DataFrame, str | None]:
    """Same idea as bulk_prop_edges, for the Anytime-TD market — pulled and shown separately
    since its pricing shape (Yes/No, no line) and confidence math are unrelated to the rest
    of the table. Costs `len(events)` credits total (1 market)."""
    frames, rem, errors = [], None, []
    for ev in events:
        try:
            raw, rem = fetch_event_odds(ev["id"], [ATTD_MARKET])
        except PropsError as e:
            errors.append(f"{ev.get('label', ev.get('id'))}: {e}")
            continue
        d = attd_edges_from_raw(raw, week)
        if not d.empty:
            d.insert(0, "game", ev.get("label", ev.get("id", "")))
        frames.append(d)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df.empty and "conf" in df.columns:
        df = (df.assign(_rank=df["conf"].map(_CONF_RANK).fillna(0))
                .sort_values(["_rank", "p edge"], ascending=False, na_position="last")
                .drop(columns="_rank").reset_index(drop=True))
    if errors:
        df.attrs["errors"] = errors
    return df, rem


def confident_only(df: pd.DataFrame, role_lo: float = 0.65, role_hi: float = 1.5,
                   ratio_hi: float = 1.9) -> pd.DataFrame:
    """Keep rows with a real edge:

    * `conf` is a labelled tier (drops `—` — under ~0.15 SD, and all pass-att / low-z rush),
    * `role_ratio` in [role_lo, role_hi] — the 2026 FP projection isn't wildly off the
      player's 2025 trailing output (role looks unchanged),
    * `our proj <= ratio_hi * line` — kills stale-role blowups (proj ~2x a sharp line).
    """
    if df.empty:
        return df
    keep = pd.Series(True, index=df.index)
    if "conf" in df.columns:
        keep &= df["conf"].isin(["lean", "solid", "strong", "high"])
    elif "z" in df.columns:
        keep &= df["z"].abs() >= 0.15
    if "role_ratio" in df.columns:
        keep &= df["role_ratio"].isna() | df["role_ratio"].between(role_lo, role_hi)
    if {"our proj", "line"}.issubset(df.columns):
        keep &= (df["our proj"] / df["line"].replace(0, pd.NA)) <= ratio_hi
    return df[keep].reset_index(drop=True)


def role_stable(df: pd.DataFrame, role_lo: float = 0.85, role_hi: float = 1.20,
                ratio_hi: float = 1.40) -> pd.DataFrame:
    """A tighter follow-on filter for what's actually worth betting, not just "not obvious
    garbage" (that's confident_only's job). Requires:

    * `role_ratio` in [role_lo, role_hi] — a CONFIRMED stable role. Unlike confident_only,
      a missing role_ratio does NOT pass here — "recommended" means we checked and it's
      fine, not that we couldn't check.
    * `our proj <= ratio_hi * line` — tighter than confident_only's 1.9x; the 1.4-1.9x zone
      is exactly where stale-2025-role inflation concentrates (rec yds/receptions props on
      players whose role changed for 2026), so it's excluded here even though confident_only
      lets it through.

    Meant to run on top of confident_only's output.
    """
    if df.empty:
        return df
    keep = pd.Series(True, index=df.index)
    if "role_ratio" in df.columns:
        keep &= df["role_ratio"].notna() & df["role_ratio"].between(role_lo, role_hi)
    else:
        keep &= False
    if {"our proj", "line"}.issubset(df.columns):
        keep &= (df["our proj"] / df["line"].replace(0, pd.NA)) <= ratio_hi
    return df[keep].reset_index(drop=True)


# back-compat alias
stable_only = confident_only


def snapshot_lines(edges: pd.DataFrame, season: int, week: int, event: str) -> int:
    """Append every (player, market) line + our projection from a pull to
    line_history.csv, so the calibration can later grade against the *real* book line
    (not just hand-logged bets). Deduped on (season, week, player, market), latest kept."""
    if edges is None or edges.empty:
        return 0
    keep = ["player", "market", "line", "our proj", "conf", "z", "edge", "p(hit)%", "p edge",
            "book %", "edge %", "lean", "best"]
    snap = edges[[c for c in keep if c in edges.columns]].copy()
    snap.insert(0, "season", int(season))
    snap.insert(1, "week", int(week))
    snap.insert(2, "event", event)
    snap["pulled_at"] = datetime.now().isoformat(timespec="seconds")
    snap["actual"] = pd.NA
    snap["result"] = ""
    LINE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LINE_HISTORY_PATH.exists():
        prev = pd.read_csv(LINE_HISTORY_PATH)
        snap = (pd.concat([prev, snap], ignore_index=True)
                .drop_duplicates(["season", "week", "player", "market"], keep="last"))
    snap.to_csv(LINE_HISTORY_PATH, index=False)
    return int(len(edges))


def save_last_pull(df: pd.DataFrame, game: str, rem: str | None, snap: int | None = None,
                   attd_df: pd.DataFrame | None = None) -> None:
    """Cache the full pull (every column, including role_ratio) to disk so reopening the
    screen — or restarting the app — restores it instead of needing a fresh pull.
    `attd_df`, if pulled, is cached the same way in its own file (a run that didn't pull
    Anytime TD clears any stale file from an earlier run, so a restore never shows props
    that weren't actually part of this pull)."""
    LAST_PULL_CSV.parent.mkdir(parents=True, exist_ok=True)
    (df if df is not None else pd.DataFrame()).to_csv(LAST_PULL_CSV, index=False)
    if attd_df is not None and not attd_df.empty:
        attd_df.to_csv(LAST_PULL_ATTD_CSV, index=False)
    else:
        LAST_PULL_ATTD_CSV.unlink(missing_ok=True)
    meta = {"game": game, "rem": rem, "snap": snap,
            "pulled_at": datetime.now().isoformat(timespec="seconds")}
    LAST_PULL_META.write_text(json.dumps(meta), encoding="utf-8")


def load_last_pull() -> dict | None:
    """The last-saved pull as a `pe_data`-shaped dict, or None if there isn't one / it's
    unreadable. `attd_df` is None when the last pull didn't include Anytime TD props."""
    if not LAST_PULL_CSV.exists() or not LAST_PULL_META.exists():
        return None
    try:
        df = pd.read_csv(LAST_PULL_CSV)
        meta = json.loads(LAST_PULL_META.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    attd_df = None
    if LAST_PULL_ATTD_CSV.exists():
        try:
            attd_df = pd.read_csv(LAST_PULL_ATTD_CSV)
        except Exception:  # noqa: BLE001
            attd_df = None
    return {"df": df, "rem": meta.get("rem"), "game": meta.get("game"),
            "snap": meta.get("snap"), "pulled_at": meta.get("pulled_at"), "restored": True,
            "attd_df": attd_df}


def clear_last_pull() -> None:
    for p in (LAST_PULL_CSV, LAST_PULL_META, LAST_PULL_ATTD_CSV):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
