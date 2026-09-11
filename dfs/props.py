"""The Odds API player-prop lines, and the edge vs our trailing-usage projection.

The free Odds API tier returns player props. `/events` is free; each
`/events/{id}/odds` call costs 1 credit *per market*, so the screen gates pulls behind
an explicit button and a short-TTL cache. Our side of the edge is
`matchup_model.project_stats.projected_line()` (trailing usage x efficiency), which is
2025-season data until the Wednesday pulls feed 2026 game logs — so a player whose role
changed for 2026 shows a fake edge. `role_ratio` flags those.
"""
from __future__ import annotations

import statistics as _st
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from dfs.config import DATA, ODDS_API_KEY
from dfs.names import normalize_name

# every pull appends here so the edge calibration can use *real* book lines, not just
# hand-logged bets (see dfs.bets.line_history_buckets)
LINE_HISTORY_PATH = DATA / "props" / "line_history.csv"

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
BOOKS = "draftkings,fanduel"      # only pull these two books
_NICE = {"rec_yds": "rec yds", "rec": "receptions", "rush_yds": "rush yds",
         "rush_att": "rush att", "pass_yds": "pass yds", "pass_td": "pass TD",
         "pass_att": "pass att"}

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


def conf_label(z: float, comp: str | None = None) -> str:
    if comp in _MKT_SKIP or z is None or not (abs(z) == abs(z)):   # NaN-safe
        return "—"
    az = abs(float(z)) - _MKT_Z_OFFSET.get(comp, 0.0)
    for thr, lbl in _CONF_TIERS:
        if az >= thr:
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
            out.append({"id": e["id"], "commence_time": e["commence_time"],
                        "label": f"{e['away_team']} @ {e['home_team']}  ·  "
                                 f"{ts.astimezone().strftime('%a %m/%d %I:%M %p')}"})
    return out, rem


def _amer_to_prob(price: float) -> float:
    return 100 / (price + 100) if price > 0 else (-price) / ((-price) + 100)


def _amer_to_dec(price: float) -> float:
    return 1 + price / 100 if price > 0 else 1 + 100 / (-price)


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + _erf(z / (2 ** 0.5)))


def _erf(x: float) -> float:
    # Abramowitz & Stegun 7.1.26
    s = 1 if x >= 0 else -1
    x = abs(x)
    t = 1 / (1 + 0.3275911 * x)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
              - 0.284496736) * t + 0.254829592) * t * (2.718281828 ** (-x * x))
    return s * y


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
        pos, wk_proj = info.get(nk, ("WR", None))
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
        # model P(bet hits), shrunk toward .5 (the raw normal model is overconfident)
        p_raw = _norm_cdf(abs(z))
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
