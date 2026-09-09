"""Central configuration: paths, DraftKings NFL Classic rules, tunable knobs, .env loading."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# dfs/ lives inside the Fantasy Model repo; all DFS data is under data/dfs/.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dfs"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:  # python-dotenv missing or unreadable .env — env vars still work
    pass

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASUITE_DIR = DATA / "datasuite"
DK_CACHE_DIR = DATA / "dk"                 # live-pull cache (git-ignored)
DK_SNAPSHOT_DIR = DATA / "dk_snapshot"     # weekly committed snapshot (tracked) — see dfs/refresh_dk.py
ODDS_CACHE_DIR = DATA / ".cache"
OUTPUT_DIR = DATA / "outputs"

for _d in (DATA, DATASUITE_DIR, DK_CACHE_DIR, DK_SNAPSHOT_DIR, ODDS_CACHE_DIR, OUTPUT_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except OSError:  # read-only FS on some hosts — the app still runs
        pass


def projection_path(week: int) -> Path:
    """User drops the FantasyPoints weekly export here as 'projections.week{N}.csv'."""
    return DATA / f"projections.week{week}.csv"


# ── Environment ──────────────────────────────────────────────────────────────
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
DK_SPORT = os.getenv("DK_SPORT", "NFL").strip() or "NFL"
DK_DRAFT_GROUP_ID = os.getenv("DK_DRAFT_GROUP_ID", "").strip()

# ── DraftKings NFL Classic contest rules ─────────────────────────────────────
SALARY_CAP = 50_000
ROSTER_SIZE = 9
# slot -> (min, max) count of that base position across the 9 roster spots.
# FLEX is the extra RB/WR/TE: exactly one of RB/WR/TE goes one above its minimum.
POSITION_LIMITS = {
    "QB": (1, 1),
    "RB": (2, 3),
    "WR": (3, 4),
    "TE": (1, 2),
    "DST": (1, 1),
}
FLEX_POSITIONS = ("RB", "WR", "TE")
FLEX_TOTAL = 7  # RB + WR + TE always sum to exactly 7 (2+3+1 + 1 flex)
DK_DST_LABELS = ("DST", "D/ST", "DEF")

# ── Optimizer defaults (all overridable in the UI) ───────────────────────────
@dataclass
class OptimizerConfig:
    n_lineups: int = 1
    lean: float = 0.0                 # 0 = median projection, 1 = ceiling
    variance: float = 0.0            # 0..1 — random projection jitter per lineup; higher = more
                                     # varied player mixes, and re-running gives a fresh set
    seed: int | None = None          # set for reproducible variance runs
    min_salary: int = 49_000          # force lineups to spend at least this much
    max_from_team: int = 4            # cap skill players from a single NFL team
    min_uniques: int = 2             # min roster differences between generated lineups
    max_exposure: float = 0.6        # cap any player's share across generated lineups
    stack_qb: bool = True             # require QB + >= stack_min same-team pass catcher(s)
    stack_min: int = 1
    bring_back: bool = False          # also require >= 1 opposing WR/TE
    avoid_dst_vs_qb: bool = True      # don't roster a DST facing a rostered QB
    apply_matchup_boost: bool = False # tilt objective toward Edge Finder spots (Phase 2)
    boost_weight: float = 0.05
    locks: list[str] = field(default_factory=list)     # dk_id strings
    excludes: list[str] = field(default_factory=list)  # dk_id strings


# ── Scoring priors: per-position coefficient of variation (stdev / mean) ──────
# Rough industry values; used only to build a ceiling/floor around the point projection.
POSITION_CV = {"QB": 0.30, "RB": 0.40, "WR": 0.45, "TE": 0.50, "DST": 0.60}

# ── Matchup Edge model ──────────────────────────────────────────────────────
EDGE_ODDS_TTL_HOURS = 6
DK_CACHE_TTL_MIN = 30
