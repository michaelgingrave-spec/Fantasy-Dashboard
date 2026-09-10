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

from dfs.config import ODDS_API_KEY
from dfs.names import normalize_name

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
_NICE = {"rec_yds": "rec yds", "rec": "receptions", "rush_yds": "rush yds",
         "rush_att": "rush att", "pass_yds": "pass yds", "pass_td": "pass TD",
         "pass_att": "pass att"}

# std-dev of the stat as a fraction of the line — a ballpark only, for the ~EV column.
_SIGMA_FRAC = {"rec_yds": 0.62, "rush_yds": 0.60, "pass_yds": 0.22, "rec": 0.55,
               "rush_att": 0.35, "pass_att": 0.18, "pass_td": 0.85}


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
    """Raw Odds API payload for one event. Costs 1 credit per market."""
    return _get(f"/events/{event_id}/odds", regions="us", oddsFormat="american",
                markets=",".join(markets))


def event_prop_edges(event_id: str, markets: list[str], week: int) -> tuple[pd.DataFrame, str | None]:
    """Fetch one event's props and build the edge table. See `edges_from_raw`."""
    raw, rem = fetch_event_odds(event_id, markets)
    return edges_from_raw(raw, week), rem


def edges_from_raw(raw: dict, week: int) -> pd.DataFrame:
    """One row per (player, market): consensus line, our projection, edge, the side we
    lean, the book's no-vig probability for that side, and a rough EV%."""
    from dfs.projections import load_weekly_projections
    from matchup_model.project_stats import projected_line

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
            line_cache[nk] = projected_line(nk, pos)
        pl = line_cache[nk]
        our = pl.get("line", {}).get(comp) if pl.get("fp") is not None else None
        if our is None:
            continue
        our = float(our)
        edge = our - line
        lean = "OVER" if edge > 0 else "UNDER"

        # best price on the leaned side + no-vig prob from the median two-way
        prices = d["over"] if lean == "OVER" else d["under"]
        best_price, best_book = max(prices, key=lambda x: _amer_to_dec(x[0])) if prices else (None, "")
        p_over = _st.median([_amer_to_prob(p) for p, _ in d["over"]]) if d["over"] else None
        p_under = _st.median([_amer_to_prob(p) for p, _ in d["under"]]) if d["under"] else None
        novig = None
        if p_over is not None and p_under is not None and (p_over + p_under):
            novig = (p_over if lean == "OVER" else p_under) / (p_over + p_under)

        # rough EV using a normal around a *market-anchored* mean — trusting our raw
        # projection fully makes every disagreement look like +100% EV, which it isn't.
        anchor = 0.6 * line + 0.4 * our
        sigma = max(_SIGMA_FRAC.get(comp, 0.5) * max(line, 1e-6), 1e-6)
        p_our_over = 1 - _norm_cdf((line - anchor) / sigma)
        p_our = min(max(p_our_over if lean == "OVER" else 1 - p_our_over, 0.05), 0.95)
        ev = None
        if best_price is not None:
            dec = _amer_to_dec(best_price)
            ev = p_our * (dec - 1) - (1 - p_our)

        role_ratio = (wk_proj / pl["fp"]) if (wk_proj and pl.get("fp")) else None
        rows.append({
            "player": player, "market": _NICE.get(comp, comp),
            "line": round(line, 1), "our proj": round(our, 1),
            "edge": round(edge, 1), "edge %": round(100 * edge / line, 0) if line else None,
            "lean": lean, "book %": round(100 * novig, 0) if novig is not None else None,
            "~EV %": round(100 * ev, 1) if ev is not None else None,
            "best": f"{best_book} {best_price:+d}" if best_price is not None else "",
            "role_ratio": round(role_ratio, 2) if role_ratio is not None else None,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("edge", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return df


def confident_only(df: pd.DataFrame, role_lo: float = 0.65, role_hi: float = 1.5,
                   ratio_lo: float = 0.6, ratio_hi: float = 1.7) -> pd.DataFrame:
    """Keep only rows we'd actually trust for Week 1:

    * `role_ratio` in [role_lo, role_hi] — the 2026 weekly FP projection isn't wildly
      off the player's 2025 trailing output (role looks unchanged), and
    * `our proj / line` in [ratio_lo, ratio_hi] — our number isn't so far from a sharp
      market line that the model is the likelier thing to be wrong.
    """
    if df.empty:
        return df
    keep = pd.Series(True, index=df.index)
    if "role_ratio" in df.columns:
        keep &= df["role_ratio"].isna() | df["role_ratio"].between(role_lo, role_hi)
    if {"our proj", "line"}.issubset(df.columns):
        r = df["our proj"] / df["line"].replace(0, pd.NA)
        keep &= r.between(ratio_lo, ratio_hi)
    return df[keep].reset_index(drop=True)


# back-compat alias
stable_only = confident_only
