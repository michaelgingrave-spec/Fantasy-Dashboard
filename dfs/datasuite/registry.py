"""Read whatever FantasyPoints Data Suite exports land in data/datasuite/.

The exact table schemas are not known ahead of time, so every file goes through a
schema-tolerant adapter:
  * find the header row (skip a leading group-label row if present)
  * detect the player-name, team and (optional) position columns
  * keep every other numeric column as a feature, prefixed ``ds__``

Filename convention (best effort, not required):
  {category}_{table}_{scope}.csv   e.g.
  receiving_advanced_player.csv, rushing_advanced_player.csv,
  defense_advanced_receiving_team.csv, passing_pressure_player.csv
`scope` in {player, team, offense, defense}; category in
{passing, rushing, receiving, offense, defense, misc}.
An optional data/datasuite/manifest.json can override: {filename: {table_type, scope, week, as_of}}.
"""
from __future__ import annotations

import csv as _csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from dfs.names import DK_TEAMS, name_team_key, norm_team, normalize_name

_NAME_HINTS = ("name", "player", "playername")
_TEAM_HINTS = ("team", "tm", "teamabbrev", "teamabbreviation")
_POS_HINTS = ("pos", "position")
_CATEGORIES = ("passing", "rushing", "receiving", "offense", "defense", "misc")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class DataSuiteFile:
    path: Path
    category: str
    scope: str          # player | team | offense | defense
    as_of: datetime | None
    raw: pd.DataFrame

    @property
    def table_type(self) -> str:
        return f"{self.category}_{self.scope}"


def _slug(s: str) -> str:
    return _SLUG_RE.sub("_", str(s).strip().lower()).strip("_")


def _load_manifest(folder: Path) -> dict:
    p = folder / "manifest.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    obj = v
                    break
        return pd.DataFrame(obj)
    # CSV: sniff whether row 0 is a real header or a group-label banner row. Do it with the
    # csv module — pandas' C parser chokes when the banner row has fewer fields than the header.
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        sample = []
        for i, row in enumerate(_csv.reader(fh)):
            sample.append(row)
            if i >= 2:
                break
    header_row = 0
    if len(sample) >= 2:
        r0 = sum(1 for x in sample[0] if str(x).strip())
        r1 = sum(1 for x in sample[1] if str(x).strip())
        if r1 > r0:  # banner/group-label row on top
            header_row = 1
    return pd.read_csv(
        path, header=header_row, dtype=str, keep_default_na=False,
        engine="python", on_bad_lines="skip",
    )


def _detect_col(cols: list[str], hints: tuple[str, ...]) -> str | None:
    norm = {c: _slug(c) for c in cols}
    for c, n in norm.items():
        if n in hints:
            return c
    for c, n in norm.items():
        if any(h in n for h in hints):
            return c
    return None


def _looks_like_team_col(series: pd.Series) -> bool:
    vals = [norm_team(v) for v in series.dropna().astype(str).head(40) if str(v).strip()]
    if not vals:
        return False
    hit = sum(1 for v in vals if v in DK_TEAMS)
    return hit / len(vals) > 0.6


def _looks_like_name_col(series: pd.Series) -> bool:
    vals = [str(v) for v in series.dropna().astype(str).head(40) if str(v).strip()]
    if not vals:
        return False
    spaced = sum(1 for v in vals if " " in v.strip() and any(ch.isalpha() for ch in v))
    return spaced / len(vals) > 0.5


def _infer_scope(name: str, cols: list[str]) -> str:
    low = name.lower()
    if "defense" in low or low.endswith("_def") or "_def_" in low:
        return "defense"
    if "offense" in low or low.endswith("_off") or "_off_" in low:
        return "offense"
    if "team" in low:
        return "team"
    return "player"


def _infer_category(name: str) -> str:
    low = name.lower()
    for c in _CATEGORIES:
        if c in low:
            return c
    return "misc"


def discover(folder: Path) -> list[DataSuiteFile]:
    folder = Path(folder)
    if not folder.exists():
        return []
    manifest = _load_manifest(folder)
    out: list[DataSuiteFile] = []
    for path in sorted(folder.glob("*")):
        if path.name == "manifest.json" or path.suffix.lower() not in (".csv", ".json"):
            continue
        try:
            raw = _read_any(path)
        except Exception:
            continue
        if raw.empty or raw.shape[1] < 3 or len(raw) < 3:
            continue
        meta = manifest.get(path.name, {})
        category = meta.get("category") or _infer_category(path.stem)
        scope = meta.get("scope") or _infer_scope(path.stem, list(raw.columns))
        as_of = None
        if meta.get("as_of"):
            try:
                as_of = datetime.fromisoformat(str(meta["as_of"]))
            except ValueError:
                as_of = None
        if as_of is None:
            as_of = datetime.fromtimestamp(path.stat().st_mtime)
        out.append(DataSuiteFile(path, category, scope, as_of, raw))
    return out


def load_table(f: DataSuiteFile) -> pd.DataFrame:
    """Normalize one file to a keyed feature frame.

    player scope -> columns: nt_key, name, team, pos?, ds__<feat>...
    team/defense/offense scope -> columns: team_key, ds__<feat>...
    Returns an empty frame if no usable key column is found.
    """
    df = f.raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)

    team_col = _detect_col(cols, _TEAM_HINTS) or next(
        (c for c in cols if _looks_like_team_col(df[c])), None
    )
    prefix = f"{f.category}_"

    if f.scope in ("team", "defense", "offense"):
        if not team_col:
            return pd.DataFrame()
        keyed = pd.DataFrame({"team_key": df[team_col].map(norm_team)})
        feats = _numeric_features(df, drop={team_col}, prefix=prefix)
        return pd.concat([keyed, feats], axis=1).groupby("team_key", as_index=False).mean(numeric_only=True)

    name_col = _detect_col(cols, _NAME_HINTS) or next(
        (c for c in cols if _looks_like_name_col(df[c])), None
    )
    if not name_col:
        return pd.DataFrame()
    pos_col = _detect_col(cols, _POS_HINTS)
    team_series = df[team_col].map(norm_team) if team_col else pd.Series([""] * len(df))
    keyed = pd.DataFrame(
        {
            "nt_key": [name_team_key(n, t) for n, t in zip(df[name_col], team_series)],
            "name": df[name_col].astype(str).str.replace("\n", " ", regex=False).str.strip(),
            "team": team_series,
            "pos": (df[pos_col].astype(str).str.upper().str.strip() if pos_col else ""),
        }
    )
    feats = _numeric_features(df, drop={name_col, team_col, pos_col}, prefix=prefix)
    tbl = pd.concat([keyed, feats], axis=1)
    tbl = tbl[tbl["nt_key"].str.strip().astype(bool)]
    return tbl.drop_duplicates(subset="nt_key", keep="first").reset_index(drop=True)


def _numeric_features(df: pd.DataFrame, drop: set, prefix: str) -> pd.DataFrame:
    keep = {}
    for c in df.columns:
        if c in drop or c is None:
            continue
        s = pd.to_numeric(
            df[c].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False),
            errors="coerce",
        )
        if s.notna().mean() < 0.5:  # mostly non-numeric -> not a feature
            continue
        keep[f"ds__{prefix}{_slug(c)}"] = s
    return pd.DataFrame(keep, index=df.index)
