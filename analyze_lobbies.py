"""
analyze_lobbies.py
------------------
Drop exported CSV pairs from the Fantasy Dashboard into the lobby_exports/ folder:
    lobby_exports/draft_analysis.csv          + league_standings.csv
    lobby_exports/draft_analysis (1).csv      + league_standings (1).csv
    ...

Run:
    python analyze_lobbies.py

Outputs land in outputs/lobbies/:
    summary.csv              - one row per lobby (winner slot, proj pts, your rank)
    top_players.csv          - player frequency + context across all winning teams
    position_by_round.csv    - how often each position was taken each round (winners)
    phase_breakdown.csv      - early/mid/late round position split
    qb_timing.csv            - when winning teams drafted their first QB
    te_timing.csv            - when winning teams drafted their first TE
    team_stacks.csv          - NFL teams stacked on winning rosters
    value_capture.csv        - avg ADP value gained/lost by round
    report.txt               - human-readable narrative summary
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
INPUT_DIR    = ROOT / "lobby_exports"
OUTPUT_DIR   = ROOT / "outputs" / "lobbies"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── File discovery ─────────────────────────────────────────────────────────────
def _suffix(name: str) -> str:
    """Extract numeric suffix from filename, e.g. 'draft_analysis (3).csv' -> '3', base -> '0'."""
    m = re.search(r"\((\d+)\)", name)
    return m.group(1) if m else "0"


def load_pairs(input_dir: Path) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    da_files = {_suffix(f.name): f for f in input_dir.glob("draft_analysis*.csv")}
    ls_files = {_suffix(f.name): f for f in input_dir.glob("league_standings*.csv")}

    matched = sorted(set(da_files) & set(ls_files), key=lambda x: int(x))
    unmatched_da = set(da_files) - set(ls_files)
    unmatched_ls = set(ls_files) - set(da_files)

    if unmatched_da:
        print(f"  WARNING: draft_analysis files with no matching standings: {sorted(unmatched_da)}")
    if unmatched_ls:
        print(f"  WARNING: league_standings files with no matching draft_analysis: {sorted(unmatched_ls)}")

    pairs = []
    for key in matched:
        try:
            da = pd.read_csv(da_files[key])
            ls = pd.read_csv(ls_files[key])
            pairs.append((int(key), da, ls))
        except Exception as e:
            print(f"  ERROR loading pair {key}: {e}")

    return pairs


# ── Build winning-team dataset ─────────────────────────────────────────────────
def build_winners(pairs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (all_winner_picks, per_lobby_summary)."""
    winner_picks = []
    lobby_rows   = []

    for key, da, ls in pairs:
        # Identify winner
        rank1 = ls[ls["Rank"] == 1]
        if rank1.empty:
            print(f"  WARNING: lobby {key} has no rank-1 row in standings — skipping")
            continue

        winner = rank1.iloc[0]
        win_slot  = int(winner["Slot"])
        win_pts   = float(winner["Proj Pts"])

        # My rank in this lobby
        my_rows = ls[ls.get("My Team", pd.Series(dtype=str)) == "Yes"] if "My Team" in ls.columns else pd.DataFrame()
        my_rank = int(my_rows.iloc[0]["Rank"]) if not my_rows.empty else None

        # Winner's roster
        roster = da[da["Slot"] == win_slot].copy()
        roster["Lobby"]     = key
        roster["Win_Slot"]  = win_slot
        roster["Win_Pts"]   = win_pts
        winner_picks.append(roster)

        lobby_rows.append({
            "Lobby":     key,
            "Win Slot":  win_slot,
            "Win Pts":   round(win_pts, 1),
            "My Rank":   my_rank,
            "Num Teams": len(ls),
            "Num Picks": len(da[da["Slot"] == win_slot]),
        })

    all_picks = pd.concat(winner_picks, ignore_index=True) if winner_picks else pd.DataFrame()
    summary   = pd.DataFrame(lobby_rows)
    return all_picks, summary


# ── Analysis helpers ───────────────────────────────────────────────────────────
def top_players(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    df["ADP_num"] = pd.to_numeric(df["ADP"], errors="coerce")
    return (
        df.groupby("Player")
          .agg(
              POS        = ("POS",     "first"),
              NFL_Team   = ("NFL Team","first"),
              Lobbies    = ("Lobby",   "nunique"),
              Avg_Round  = ("Round",   "mean"),
              Avg_ADP    = ("ADP_num", "mean"),
              Avg_ProjG  = ("Proj/G",  "mean"),
          )
          .sort_values("Lobbies", ascending=False)
          .head(top_n)
          .round(1)
          .reset_index()
    )


def position_by_round(df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        df.groupby(["Round", "POS"])
          .size()
          .unstack(fill_value=0)
    )
    for col in ["QB", "RB", "WR", "TE"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["QB", "RB", "WR", "TE"]]
    pivot["Total"] = pivot.sum(axis=1)
    return pivot.reset_index()


def phase_breakdown(df: pd.DataFrame, n_lobbies: int) -> pd.DataFrame:
    df = df.copy()
    df["Phase"] = pd.cut(df["Round"], bins=[0, 5, 10, 20],
                         labels=["Early (R1-5)", "Mid (R6-10)", "Late (R11-20)"])
    raw = (df.groupby(["Phase", "POS"]).size().unstack(fill_value=0))
    for col in ["QB", "RB", "WR", "TE"]:
        if col not in raw.columns:
            raw[col] = 0
    avg = (raw[["QB", "RB", "WR", "TE"]] / n_lobbies).round(2)
    return avg.reset_index()


def timing_stat(df: pd.DataFrame, pos: str) -> pd.DataFrame:
    first = (df[df["POS"] == pos]
               .groupby("Lobby")["Round"]
               .min()
               .value_counts()
               .sort_index()
               .reset_index())
    first.columns = [f"First_{pos}_Round", "Count"]
    return first


def team_stacks(df: pd.DataFrame, min_players: int = 2) -> pd.DataFrame:
    counts = (df.groupby(["Lobby", "NFL Team"])
                .size()
                .reset_index(name="n"))
    stacks = counts[counts["n"] >= min_players]
    return (stacks["NFL Team"]
              .value_counts()
              .reset_index()
              .rename(columns={"count": "Winning_Lobbies"}))


def value_capture(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ADP_num"] = pd.to_numeric(df["ADP"], errors="coerce")
    df["Value"]   = df["ADP_num"] - df["Pick"]
    return (df.groupby("Round")["Value"]
              .mean()
              .round(1)
              .reset_index()
              .rename(columns={"Value": "Avg_Value_vs_ADP"}))


# ── Report writer ──────────────────────────────────────────────────────────────
def write_report(path: Path, all_picks: pd.DataFrame, summary: pd.DataFrame,
                 players: pd.DataFrame, pos_rnd: pd.DataFrame,
                 phase: pd.DataFrame, qb_time: pd.DataFrame,
                 te_time: pd.DataFrame, stacks: pd.DataFrame) -> None:

    n = len(summary)
    lines = []
    a = lines.append

    a("=" * 65)
    a(f"  BEST BALL DRAFT ANALYTICS — {n} LOBBIES ANALYZED")
    a("=" * 65)
    a("")

    # Slot distribution
    slot_vc = summary["Win Slot"].value_counts().sort_index()
    avg_slot = summary["Win Slot"].mean()
    a("WINNING DRAFT SLOT DISTRIBUTION")
    a("-" * 40)
    for slot, cnt in slot_vc.items():
        bar = "█" * cnt
        a(f"  Slot {slot:>2}: {bar}  ({cnt})")
    a(f"  Average winning slot: {avg_slot:.1f}")
    a("")

    # Round 1 & 2
    r1 = all_picks[all_picks["Round"] == 1]["POS"].value_counts()
    r2 = all_picks[all_picks["Round"] == 2]["POS"].value_counts()
    a("POSITION CHOICES — ROUNDS 1 & 2  (across all winning teams)")
    a("-" * 40)
    a(f"  Round 1: " + "  ".join(f"{p}={v}" for p, v in r1.items()))
    a(f"  Round 2: " + "  ".join(f"{p}={v}" for p, v in r2.items()))
    a("")

    # QB / TE timing
    qb_avg = all_picks[all_picks["POS"] == "QB"].groupby("Lobby")["Round"].min().mean()
    te_avg = all_picks[all_picks["POS"] == "TE"].groupby("Lobby")["Round"].min().mean()
    qb_mode = all_picks[all_picks["POS"] == "QB"].groupby("Lobby")["Round"].min().mode().iloc[0]
    a("QB & TE DRAFT TIMING")
    a("-" * 40)
    a(f"  First QB — avg round {qb_avg:.1f}, most common round {qb_mode}")
    a(f"  First TE — avg round {te_avg:.1f}")
    a("")

    # Roster composition
    comp = all_picks.groupby("POS").size() / n
    a("AVG WINNING ROSTER COMPOSITION")
    a("-" * 40)
    for pos in ["QB", "RB", "WR", "TE"]:
        a(f"  {pos}: {comp.get(pos, 0):.1f}")
    a("")

    # Top players
    a(f"TOP 20 PLAYERS ON WINNING TEAMS")
    a("-" * 40)
    a(f"  {'Player':<26} {'POS':<4} {'Lobbies':>7} {'AvgRnd':>7} {'ADP':>6} {'Proj/G':>7}")
    a(f"  {'-'*26} {'-'*4} {'-'*7} {'-'*7} {'-'*6} {'-'*7}")
    for _, row in players.head(20).iterrows():
        a(f"  {str(row['Player']):<26} {str(row['POS']):<4} "
          f"{int(row['Lobbies']):>7} {row['Avg_Round']:>7} "
          f"{str(row['Avg_ADP']):>6} {str(row['Avg_ProjG']):>7}")
    a("")

    # Stacks
    a("NFL TEAM STACKS ON WINNING ROSTERS  (2+ players from same team)")
    a("-" * 40)
    for _, row in stacks.head(12).iterrows():
        a(f"  {row['NFL Team']:<6} {int(row['Winning_Lobbies'])} lobbies")
    a("")

    # Value capture
    all_picks2 = all_picks.copy()
    all_picks2["ADP_num"] = pd.to_numeric(all_picks2["ADP"], errors="coerce")
    all_picks2["Value"]   = all_picks2["ADP_num"] - all_picks2["Pick"]
    pct_fell = (all_picks2["Value"] > 0).mean() * 100
    avg_val  = all_picks2["Value"].mean()
    a("VALUE CAPTURE")
    a("-" * 40)
    a(f"  {pct_fell:.0f}% of winning-team picks were players who 'fell' past their ADP")
    a(f"  Average value per pick vs ADP: {avg_val:+.1f}")
    a("")
    a("=" * 65)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"Looking for CSV pairs in: {INPUT_DIR}")
    if not INPUT_DIR.exists():
        print(f"  ERROR: folder not found — create it and drop your CSVs in.")
        sys.exit(1)

    pairs = load_pairs(INPUT_DIR)
    if not pairs:
        print("  No matched CSV pairs found. Drop draft_analysis + league_standings files into lobby_exports/")
        sys.exit(1)

    print(f"  Found {len(pairs)} matched pairs\n")

    all_picks, summary = build_winners(pairs)
    n = len(summary)

    if all_picks.empty:
        print("  No winning rosters could be built — check your CSV files.")
        sys.exit(1)

    # Run analyses
    players  = top_players(all_picks, top_n=40)
    pos_rnd  = position_by_round(all_picks)
    phase    = phase_breakdown(all_picks, n)
    qb_time  = timing_stat(all_picks, "QB")
    te_time  = timing_stat(all_picks, "TE")
    stacks   = team_stacks(all_picks)
    val_cap  = value_capture(all_picks)

    # Save CSVs
    summary.to_csv(OUTPUT_DIR / "summary.csv",              index=False)
    players.to_csv(OUTPUT_DIR / "top_players.csv",          index=False)
    pos_rnd.to_csv(OUTPUT_DIR / "position_by_round.csv",    index=False)
    phase.to_csv(  OUTPUT_DIR / "phase_breakdown.csv",      index=False)
    qb_time.to_csv(OUTPUT_DIR / "qb_timing.csv",            index=False)
    te_time.to_csv(OUTPUT_DIR / "te_timing.csv",            index=False)
    stacks.to_csv( OUTPUT_DIR / "team_stacks.csv",          index=False)
    val_cap.to_csv(OUTPUT_DIR / "value_capture.csv",        index=False)

    # Write text report
    report_path = OUTPUT_DIR / "report.txt"
    write_report(report_path, all_picks, summary, players, pos_rnd,
                 phase, qb_time, te_time, stacks)

    # Print report to console (replace block chars that Windows console can't render)
    report_text = report_path.read_text(encoding="utf-8").replace("█", "#")
    print(report_text)
    print(f"\nOutputs written to: {OUTPUT_DIR}")
    print("  summary.csv              per-lobby winner info")
    print("  top_players.csv          player frequency + context")
    print("  position_by_round.csv    position drafting by round")
    print("  phase_breakdown.csv      early/mid/late split")
    print("  qb_timing.csv            first QB round distribution")
    print("  te_timing.csv            first TE round distribution")
    print("  team_stacks.csv          NFL team stack frequency")
    print("  value_capture.csv        avg ADP value by round")
    print("  report.txt               full narrative summary")


if __name__ == "__main__":
    main()
