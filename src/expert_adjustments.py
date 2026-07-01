"""
expert_adjustments.py
Apply analyst-driven projection adjustments on top of the statistical model.

Three layers:
  1. SCHEME TRANSFERS    — coordinator moves: compute the offensive efficiency
     delta between the OC's prior team and the new team, then scale each
     player by how well their profile fits the new scheme.
  2. PLAYER TEAM CHANGES — individual players moving teams: compare the scheme
     efficiency of their OLD team vs their NEW team, with positive = upgrade.
     NOTE: ratio is inverted vs. scheme transfers — for player moves we compute
     (new_team_YPRR / old_team_YPRR), so > 1 means the player is moving up.
  3. MANUAL OVERRIDES    — your football takes: explicit % bumps/cuts with
     free-text reasons logged in the output.

All adjustments compound: (1 + scheme_adj) * (1 + manual_adj) - 1.

Usage:
  From main.py:
    from expert_adjustments import apply_all_adjustments
    projections = apply_all_adjustments(projections, season_stats)
"""

import pandas as pd
import numpy as np


# ==============================================================================
#   TEAM UPDATES FOR 2026
#   Players who changed teams entering 2026. The projection model assigns each
#   player their 2025 team; this dict overrides the Team column in the output.
#   Format: (player_name_exact, POS) -> new_team_abbrev
# ==============================================================================

TEAM_UPDATES_2026 = {
    # ── Wide Receivers ────────────────────────────────────────────────────────
    ("DJ Moore",              "WR"): "BUF",
    ("Michael Pittman",       "WR"): "PIT",
    ("Mike Evans",            "WR"): "SF",
    ("A.J. Brown",            "WR"): "NE",
    ("Jaylen Waddle",         "WR"): "DEN",   # MIA -> DEN (did NOT follow McDaniel)
    ("Wan'Dale Robinson",     "WR"): "TEN",
    ("Adonai Mitchell",       "WR"): "NYJ",
    ("Dontayvion Wicks",      "WR"): "PHI",
    ("Darnell Mooney",        "WR"): "NYG",
    ("Marquise Brown",        "WR"): "PHI",
    ("Jauan Jennings",        "WR"): "MIN",
    ("Romeo Doubs",           "WR"): "NE",
    ("Jalen Nailor",          "WR"): "LV",
    ("Olamide Zaccheaus",     "WR"): "ATL",
    ("Jahan Dotson",          "WR"): "ATL",
    ("Kendrick Bourne",       "WR"): "ARZ",
    ("Dyami Brown",           "WR"): "WAS",
    ("Calvin Austin",         "WR"): "NYG",
    ("Van Jefferson",         "WR"): "WAS",
    ("Tutu Atwell",           "WR"): "MIA",
    ("Christian Kirk",        "WR"): "SF",
    ("Nick Westbrook-Ikhine", "WR"): "IND",
    ("Jalen Tolbert",         "WR"): "MIA",
    ("Elijah Moore",          "WR"): "PHI",
    ("Kalif Raymond",         "WR"): "CHI",
    ("Devin Duvernay",        "WR"): "ARZ",
    ("Skyy Moore",            "WR"): "GB",
    # ── Running Backs ─────────────────────────────────────────────────────────
    ("Rico Dowdle",           "RB"): "PIT",
    ("Kenneth Walker",        "RB"): "KC",
    ("David Montgomery",      "RB"): "HST",
    ("Travis Etienne",        "RB"): "NO",
    ("Kenneth Gainwell",      "RB"): "TB",
    # De'Von Achane stayed on MIA — did NOT follow McDaniel to LAC
    ("De'Von Achane",         "RB"): "MIA",
    ("Rachaad White",         "RB"): "WAS",
    ("Isiah Pacheco",         "RB"): "DET",
    ("Emanuel Wilson",        "RB"): "SEA",
    ("Brian Robinson",        "RB"): "ATL",
    ("Emari Demercado",       "RB"): "KC",
    ("Keaton Mitchell",       "RB"): "LAC",   # BLT -> LAC (gets McDaniel scheme boost)
    ("Jerome Ford",           "RB"): "WAS",
    ("Michael Carter",        "RB"): "TEN",
    ("Chris Rodriguez",       "RB"): "JAX",
    # ── Tight Ends ────────────────────────────────────────────────────────────
    ("Isaiah Likely",         "TE"): "NYG",
    ("Chig Okonkwo",          "TE"): "WAS",   # TEN -> WAS (no longer gets Daboll/TEN boost)
    ("David Njoku",           "TE"): "LAC",   # CLV -> LAC (gets McDaniel scheme boost)
    ("Daniel Bellinger",      "TE"): "TEN",   # NYG -> TEN (gets Daboll scheme)
    ("Tyler Conklin",         "TE"): "DET",
    ("Noah Fant",             "TE"): "NO",
    ("Austin Hooper",         "TE"): "ATL",
}


# ==============================================================================
#   MANUAL OVERRIDES
#   Format: ('Player Name', 'POS', 'Team abbrev', adjustment_pct, 'Reason')
#   Team abbrev should be the player's 2026 team.
#   adjustment_pct: +0.20 = 20% bump, -0.15 = 15% cut
# ==============================================================================

MANUAL_OVERRIDES = [
    # ── QB change: Tua to ATL — upgrade over Penix ───────────────────────────
    ("Drake London",   "WR", "ATL", +0.10, "Tua to ATL — accurate field-stretching QB boosts outside WR"),
    ("Bijan Robinson", "RB", "ATL", +0.05, "Tua to ATL — better passing game opens RPO and run lanes"),
    ("Kyle Pitts",     "TE", "ATL", +0.08, "Tua to ATL — seam-threat TE finally gets an accurate QB"),

    # ── Role expansion / new starter bumps ───────────────────────────────────
    # Model only sees Montgomery as a DET committee back behind Gibbs.
    # In HST he's the clear feature back — model can't see role expansion.
    ("David Montgomery", "RB", "HST", +0.70, "New feature back in HST — clear WR1 role; model undersells from DET committee; FP projects 218 FP"),

    # ── Rookie/sophomore TE adjustments — comp pool undersells elite rookies ──
    # Tyler Warren: TE1-caliber season in 2025 (188 FP, 17 games), comp pool
    # dragged down by injury busts (Dulcich 6 FP, Musgrave 12 FP). FP TE3.
    ("Tyler Warren",     "TE", "IND", +0.25, "TE1-caliber talent — comp pool undersells role/volume; FP TE3"),
    # Colston Loveland: elite athletic profile, CHI offense has tools around him.
    # Same comp pool problem as Warren. FP TE4.
    ("Colston Loveland", "TE", "CHI", +0.57, "Elite rookie TE profile — comp pool skewed by injury busts; FP TE2 at 185 FP"),

    # ── Established starters the model undersells ─────────────────────────────
    # Brock Bowers: FP TE1 at 214 FP — elite sophomore target hog; model underrates volume.
    ("Brock Bowers",     "TE",  "LV", +0.28, "FP TE1 — elite sophomore TE; model underrates target share and volume"),
    # Jake Ferguson: DAL TE1 in a high-volume offense. Model undersells his role.
    ("Jake Ferguson",    "TE", "DAL", +0.20, "Established TE1 in high-volume DAL offense — FP TE9"),
    # Isaiah Likely: moved to NYG — model still has him as BLT backup; NYG starter role.
    ("Isaiah Likely",    "TE", "NYG", +0.73, "Moved to NYG — clear TE1 starter; model still sees BAL backup role; FP projects 136 FP"),
    # Blake Corum: clear LAR starter in 2026, comp system drags him down.
    ("Blake Corum",      "RB", "LA",  +0.20, "New featured starter in LAR — comp pool misses role expansion; FP RB37"),
    # Bhayshul Tuten: JAX featured back, model undersells his expanded role.
    ("Bhayshul Tuten",   "RB", "JAX", +0.18, "Featured back in JAX — model undersells role expansion; FP RB24"),

    # ── Elite WRs penalized for injury-shortened 2025 seasons ────────────────
    # CeeDee Lamb: 14-game 2025, model discounts for missed games. Elite talent.
    ("CeeDee Lamb",      "WR", "DAL", +0.18, "Elite WR1 — model penalizes injury-shortened 2025 (14 games); FP WR5"),
    # Justin Jefferson: 10-game 2025. Same effect — XGBoost sees limited data.
    ("Justin Jefferson", "WR", "MIN", +0.15, "Elite WR1 — injury-shortened 2025 (10 games) suppresses projection; FP WR6"),
    # Ladd McConkey: already gets McDaniel scheme boost but model still undershoots.
    ("Ladd McConkey",    "WR", "LAC", +0.10, "McDaniel scheme WR1 — breakout trajectory; FP WR17"),

    # ── Age/injury fade cuts ──────────────────────────────────────────────────
    # George Kittle: model sees great FPG from 11-game 2025; FP skeptical at age 32.
    ("George Kittle",    "TE", "SF",  -0.25, "Age/injury risk at 32 — FP very skeptical (TE33); model over-extrapolates elite FPG"),
    ("Keenan Allen",     "WR", "LAC", -0.48, "Aging WR at 33 in LAC — FP projects 99 FP vs our 179; limited role expected"),
    ("Alvin Kamara",     "RB",  "NO", -0.55, "Age/role concerns at 30 in NO — FP projects 70 FP vs our 154"),

    # ── Backup / depth cuts — model sees 2025 stats but they lost starting roles ──
    ("Michael Carter",        "RB", "TEN", -0.85, "Depth RB in TEN — not a starter; FP projects 3 FP"),
    ("Demarcus Robinson",     "WR",  "SF", -0.85, "Depth WR in SF — non-factor; FP projects 18 FP"),
    ("Bam Knight",            "RB", "ARZ", -0.85, "Depth RB behind Jeremiyah Love in ARZ; FP projects 2 FP"),
    ("Sean Tucker",           "RB",  "TB", -0.75, "Backup RB behind Bucky Irving in TB; FP projects 29 FP"),
    ("Tyler Badie",           "RB", "DEN", -0.85, "Depth RB in DEN — non-factor; FP projects 11 FP"),
    ("Devin Singletary",      "RB", "NYG", -0.75, "Backup behind Skattebo in NYG; FP projects 40 FP"),
    ("Dyami Brown",           "WR", "WAS", -0.85, "Depth WR in WAS — non-factor; FP projects 6 FP"),
    ("AJ Dillon",             "RB", "CAR", -0.85, "Fringe RB in CAR — non-factor; FP projects 5 FP"),
    ("Jeremy McNichols",      "RB", "WAS", -0.85, "Depth RB in WAS — non-factor; FP projects 8 FP"),
    ("Dont'e Thornton",       "WR",  "LV", -0.70, "Depth WR in LV — limited role; FP projects 26 FP"),
    ("Devin Neal",            "RB",  "NO", -0.72, "Depth RB in NO — limited role; FP projects 24 FP"),
    ("Casey Washington",      "WR", "ATL", -0.82, "Depth WR in ATL — non-factor; FP projects 13 FP"),
    ("Kaleb Johnson",         "RB", "PIT", -0.78, "Depth RB in PIT — limited role; FP projects 18 FP"),
    ("Isaiah Hodgins",        "WR", "NYG", -0.85, "Depth WR in NYG — non-factor; FP projects 5 FP"),
    ("Kyle Williams",         "WR",  "NE", -0.80, "Depth WR in NE — non-factor; FP projects 14 FP"),
    ("Jerome Ford",           "RB", "WAS", -0.70, "Depth RB in WAS — limited role; FP projects 22 FP"),
    ("Zavier Scott",          "RB", "MIN", -0.65, "Depth RB in MIN — limited role; FP projects 29 FP"),
    ("Tyler Allgeier",        "RB", "ATL", -0.60, "Backup behind Jeremiyah Love in ARZ; FP projects 64 FP"),
    ("Jordan Mason",          "RB", "MIN", -0.58, "Backup RB in MIN — not the starter; FP projects 71 FP"),
    ("Jacory Croskey-Merritt","RB", "WAS", -0.62, "Depth RB in WAS — FP projects 59 FP vs our 163"),
    ("Emanuel Wilson",        "RB", "SEA", -0.58, "Backup RB in SEA — FP projects 58 FP vs our 146"),
    ("Emari Demercado",       "RB",  "KC", -0.75, "Depth RB in KC committee; FP projects 32 FP"),
    ("Troy Franklin",         "WR", "DEN", -0.50, "Sophomore WR in crowded DEN offense; FP projects 83 FP"),

    # ── Role expansion / clear starter upgrades ───────────────────────────────
    ("Jayden Reed",           "WR",  "GB", +0.90, "Clear GB WR1 — massive role expansion from 2025; FP projects 173 FP"),
    ("Jalen Coker",           "WR", "CAR", +1.00, "CAR WR1 with full role in 2026 — model sees very limited 2025 data; FP projects 126 FP"),
    ("Luther Burden",         "WR", "CHI", +0.40, "Clear CHI WR1 opportunity in 2026; FP projects 174 FP"),
    ("Ryan Flournoy",         "WR", "DAL", +0.60, "New DAL WR with expanded starting role; FP projects 141 FP"),
    ("Kenneth Walker",        "RB",  "KC", +0.15, "KC featured back; scheme already applied but FP projects 239 FP vs our 188"),
    ("Pat Freiermuth",        "TE", "PIT", +0.42, "PIT TE1 with full starting role; FP projects 134 FP vs our 80"),
    ("Jalen Nailor",          "WR",  "LV", +0.60, "New LV WR1 with expanded opportunity; FP projects 127 FP vs our 68"),
    ("Rhamondre Stevenson",   "RB",  "NE", +0.25, "NE featured back with expanded role in 2026; FP projects 209 FP"),
    ("Javonte Williams",      "RB", "DAL", +0.21, "DAL RB with renewed health and clear role; FP projects 224 FP vs our 167"),
    ("Parker Washington",     "WR", "JAX", +0.10, "JAX WR with expanded starting opportunity; FP projects 185 FP"),
]


# ==============================================================================
#   SCHEME TRANSFER DEFINITIONS
#   OC moves: "OC X is moving from team A to team B."
#   Formula: (A's scheme YPRR) / (B's prior YPRR) — if A > B, positive for B players.
# ==============================================================================

SCHEME_TRANSFERS = [
    # ── Mike McDaniel: MIA -> LAC (2026) ─────────────────────────────────────
    # McDaniel's MIA YPRR >> prior LAC (Roman era) => large positive for LAC players.
    # De'Von Achane (moving with McDaniel) already played in this system;
    # his base projection captures it, so he's intentionally excluded here.
    {
        "oc_name":     "Mike McDaniel",
        "from_team":   "MIA",
        "from_years":  [2022, 2023, 2024, 2025],
        "to_team":     "LAC",
        "prior_years": [2024, 2025],
        "translation": 0.55,
        "player_weights": {
            # McConkey: prototypical McDaniel slot WR — short-medium routes, YAC machine
            "Ladd McConkey":    1.00,
            # Johnston: outside speed WR; McDaniel used Hill similarly
            "Quentin Johnston": 0.60,
            # Tre Harris: young outside WR, beneficiary of scheme-driven volume
            "Tre Harris":       0.55,
            # Hampton: new LAC RB in McDaniel system (Achane did NOT follow McDaniel)
            "Omarion Hampton":  0.70,
            "Kimani Vidal":     0.30,
            # Keaton Mitchell: moved BLT -> LAC, new McDaniel system RB
            "Keaton Mitchell":  0.55,
            # TE: McDaniel used TEs as checkdowns / seam threats
            "Oronde Gadsden":   0.65,
            # David Njoku: moved CLV -> LAC, established TE in McDaniel seam-threat role
            "David Njoku":      0.60,
        },
    },

    # ── Kevin Stefanski: CLV HC -> ATL (2026) ────────────────────────────────
    # CLV YPRR was suppressed (~1.68) by Watson's issues; ATL prior ~1.77.
    # Raw signal is slightly negative but Stefanski's outside-zone run scheme
    # is perfect for Bijan Robinson. Tua manual override offsets the YPRR gap.
    {
        "oc_name":     "Kevin Stefanski",
        "from_team":   "CLV",
        "from_years":  [2021, 2022, 2023, 2024],
        "to_team":     "ATL",
        "prior_years": [2024, 2025],
        "translation": 0.45,
        "player_weights": {
            # Bijan Robinson: textbook outside-zone RB — Stefanski's scheme is built for him
            "Bijan Robinson": 1.00,
            # Drake London: big-bodied outside WR, similar to Amari Cooper in CLV
            "Drake London":   0.70,
            # Kyle Pitts: Stefanski uses TEs well as underneath/seam options
            "Kyle Pitts":     0.65,
            "Darnell Mooney": 0.45,
        },
    },

    # ── Zac Robinson: ATL -> TB (2026) ───────────────────────────────────────
    # Robinson's ATL offense (~1.77 YPRR) is below TB's recent (~2.10).
    # Slight scheme downgrade for TB, tempered by Robinson's modern RPO/spread system.
    {
        "oc_name":     "Zac Robinson",
        "from_team":   "ATL",
        "from_years":  [2024, 2025],
        "to_team":     "TB",
        "prior_years": [2024, 2025],
        "translation": 0.40,
        "player_weights": {
            "Chris Godwin":     0.80,
            "Jalen McMillan":   0.70,
            "Trey Palmer":      0.60,
            "Cade Otton":       0.55,
            "Kenneth Gainwell": 0.60,
        },
    },

    # ── Bobby Slowik: HST -> MIA (2026) ──────────────────────────────────────
    # Slowik's HST offense (~1.74) is well below MIA's McDaniel era (~2.02).
    # Net negative for MIA. Achane stayed on MIA (did not follow McDaniel to LAC)
    # so his XGB base captures McDaniel's system — he takes the full Slowik downgrade.
    # Waddle moved to DEN; Hill not in FP rankings (retired/FA) — removed from weights.
    {
        "oc_name":     "Bobby Slowik",
        "from_team":   "HST",
        "from_years":  [2023, 2024, 2025],
        "to_team":     "MIA",
        "prior_years": [2024, 2025],
        "translation": 0.38,
        "player_weights": {
            # Achane's 2025 stats were built in McDaniel's system; Slowik is a clear downgrade
            "De'Von Achane": 0.90,
            # Malik Willis QB downgrade compounds the scheme hit for MIA pass-catchers
            "Tutu Atwell":   0.55,
        },
    },

    # ── Matt Nagy: KC -> NYG (2026) ───────────────────────────────────────────
    # Nagy's KC stint (~1.73) vs NYG's recent (~1.76) — essentially flat.
    # Nabers is an elite talent; even modest system change has small effect.
    {
        "oc_name":     "Matt Nagy",
        "from_team":   "KC",
        "from_years":  [2023, 2024, 2025],
        "to_team":     "NYG",
        "prior_years": [2024, 2025],
        "translation": 0.40,
        "player_weights": {
            "Malik Nabers":  0.90,
            "Theo Johnson":  0.60,
            # Wan'Dale Robinson moved to TEN — removed from NYG weights
        },
    },

    # ── Brian Daboll: NYG -> TEN (2026) ──────────────────────────────────────
    # Daboll's NYG (~1.71) is meaningfully below TEN's prior (~2.17).
    # Net negative — Daboll's offense was constrained by poor NYG QBs.
    # Chig Okonkwo moved to WAS (removed). Wan'Dale Robinson and Daniel Bellinger
    # moved from NYG TO TEN — they get the scheme delta applied.
    {
        "oc_name":     "Brian Daboll",
        "from_team":   "NYG",
        "from_years":  [2022, 2023, 2024, 2025],
        "to_team":     "TEN",
        "prior_years": [2024, 2025],
        "translation": 0.42,
        "player_weights": {
            "Calvin Ridley":      0.70,
            "Tony Pollard":       0.55,
            "Wan'Dale Robinson":  0.65,  # moved NYG -> TEN with Daboll
            "Daniel Bellinger":   0.50,  # moved NYG -> TEN with Daboll
            "Carnell Tate":       0.55,  # new TEN WR
        },
    },

    # ── Nathaniel Hackett: GB -> ARZ (2026) ──────────────────────────────────
    # Hackett's 2021 GB offense (~2.16) vs ARZ Petzing era (~1.53).
    # Largest positive scheme delta of all transfers. ARZ players benefit greatly.
    # Translation tempered since Hackett hasn't called plays since 2021.
    {
        "oc_name":     "Nathaniel Hackett",
        "from_team":   "GB",
        "from_years":  [2021],
        "to_team":     "ARZ",
        "prior_years": [2024, 2025],
        "translation": 0.48,
        "player_weights": {
            # Name in data is "Marvin Harrison" (no Jr. suffix)
            "Marvin Harrison": 0.95,
            "Trey McBride":    0.85,
            "James Conner":    0.65,
            "Michael Wilson":  0.55,
        },
    },
]


# ==============================================================================
#   PLAYER TEAM CHANGES
#   Individual players who switched teams entering 2026.
#   IMPORTANT: ratio is INVERTED vs. SCHEME_TRANSFERS:
#     positive = new team's scheme is more efficient than old team's scheme.
#   Mark with 'is_player_move': True — handled separately in the engine.
# ==============================================================================

PLAYER_TEAM_CHANGES = [

    # ── Wide Receivers ────────────────────────────────────────────────────────

    # DJ Moore: CHI -> BUF
    # BUF historically the highest YPRR environment in the NFL (~2.78).
    # CHI has been a pass-scheme wasteland. Largest WR team-change positive.
    {
        "oc_name":      "DJ Moore (CHI->BUF)",
        "from_team":    "CHI",
        "from_years":   [2023, 2024, 2025],
        "to_team":      "BUF",
        "prior_years":  [2023, 2024, 2025],
        "translation":  0.60,
        "is_player_move": True,
        "player_weights": {"DJ Moore": 0.85},
    },

    # Michael Pittman: IND -> PIT
    # IND had very high YPRR (~2.39); PIT is lower (~1.99).
    # Negative — PIT is not a great landing spot for a WR2.
    {
        "oc_name":      "Michael Pittman (IND->PIT)",
        "from_team":    "IND",
        "from_years":   [2023, 2024, 2025],
        "to_team":      "PIT",
        "prior_years":  [2023, 2024, 2025],
        "translation":  0.50,
        "is_player_move": True,
        "player_weights": {"Michael Pittman": 0.65},
    },

    # Mike Evans: TB -> SF
    # TB YPRR ~2.10 vs SF ~1.87. Slight negative — Evans is 32+,
    # sharing a crowded SF target tree (Deebo, Aiyuk, Kittle).
    {
        "oc_name":      "Mike Evans (TB->SF)",
        "from_team":    "TB",
        "from_years":   [2023, 2024, 2025],
        "to_team":      "SF",
        "prior_years":  [2023, 2024, 2025],
        "translation":  0.45,
        "is_player_move": True,
        "player_weights": {"Mike Evans": 0.55},
    },

    # ── Running Backs ─────────────────────────────────────────────────────────

    # Rico Dowdle: CAR -> PIT
    # CAR had low YPRR (~1.66); PIT better (~1.99). Modest positive —
    # Dowdle should be the clear lead back in Pittsburgh.
    {
        "oc_name":      "Rico Dowdle (CAR->PIT)",
        "from_team":    "CAR",
        "from_years":   [2024, 2025],
        "to_team":      "PIT",
        "prior_years":  [2023, 2024, 2025],
        "translation":  0.55,
        "is_player_move": True,
        "player_weights": {"Rico Dowdle": 0.80},
    },

    # Kenneth Walker: SEA -> KC
    # SEA had high YPRR (~2.21); KC lower (~1.73). KC uses RBs as system
    # pieces, not bell cows. Negative for Walker's volume upside.
    {
        "oc_name":      "Kenneth Walker (SEA->KC)",
        "from_team":    "SEA",
        "from_years":   [2022, 2023, 2024, 2025],
        "to_team":      "KC",
        "prior_years":  [2023, 2024, 2025],
        "translation":  0.50,
        "is_player_move": True,
        "player_weights": {"Kenneth Walker": 0.65},
    },

    # David Montgomery: DET -> HST
    # DET had very high YPRR (~2.27); HST lower (~1.74).
    # Not a clear bell cow — sharing with Pierce/Collins. Negative.
    {
        "oc_name":      "David Montgomery (DET->HST)",
        "from_team":    "DET",
        "from_years":   [2023, 2024, 2025],
        "to_team":      "HST",
        "prior_years":  [2023, 2024, 2025],
        "translation":  0.50,
        "is_player_move": True,
        "player_weights": {"David Montgomery": 0.65},
    },

    # Travis Etienne: JAX -> NO
    # JAX YPRR ~1.80; NO ~1.86. Nearly flat, slight positive.
    # Etienne likely the lead back in NO — fresh start.
    {
        "oc_name":      "Travis Etienne (JAX->NO)",
        "from_team":    "JAX",
        "from_years":   [2022, 2023, 2024, 2025],
        "to_team":      "NO",
        "prior_years":  [2023, 2024, 2025],
        "translation":  0.50,
        "is_player_move": True,
        "player_weights": {"Travis Etienne": 0.75},
    },

    # Kenneth Gainwell: PIT -> TB
    # PIT YPRR ~1.99; TB ~2.10. Slight positive for a change-of-pace back.
    {
        "oc_name":      "Kenneth Gainwell (PIT->TB)",
        "from_team":    "PIT",
        "from_years":   [2023, 2024, 2025],
        "to_team":      "TB",
        "prior_years":  [2023, 2024, 2025],
        "translation":  0.40,
        "is_player_move": True,
        "player_weights": {"Kenneth Gainwell": 0.45},
    },
]


# ── Scheme efficiency metrics to compare ─────────────────────────────────────

_SCHEME_METRICS = ["YPRR_mean", "FPG_mean", "XFPG_mean", "aDOT_mean", "TGT_per_game"]


def _team_scheme_profile(season_stats: pd.DataFrame,
                          team: str, years: list,
                          pos: str = None) -> pd.Series:
    """
    Average skill-player efficiency metrics for a team over specified years.
    Weighted by games played so that full-season contributors dominate.
    """
    mask = (season_stats["Team"] == team) & (season_stats["Year"].isin(years))
    if pos:
        mask &= (season_stats["POS"] == pos)

    sub = season_stats[mask].copy()
    if sub.empty:
        return pd.Series({m: np.nan for m in _SCHEME_METRICS})

    w = sub["Games"].clip(lower=1)
    out = {}
    for m in _SCHEME_METRICS:
        if m in sub.columns:
            out[m] = np.average(sub[m].fillna(0), weights=w)
        else:
            out[m] = np.nan
    return pd.Series(out)


def _compute_scheme_multiplier(from_profile: pd.Series,
                                to_profile: pd.Series,
                                translation: float) -> float:
    """
    Compute a single efficiency multiplier.

    from_profile is the "source of scheme quality":
      - For OC moves:     from_team = OC's origin  (the better scheme being imported)
      - For player moves: from_team = player's NEW team (what they're moving INTO)

    multiplier > 1 => from_profile scheme is more efficient than to_profile
    multiplier < 1 => to_profile scheme is more efficient (downgrade)
    """
    yprr_from = from_profile.get("YPRR_mean", np.nan)
    yprr_to   = to_profile.get("YPRR_mean",   np.nan)
    fpg_from  = from_profile.get("FPG_mean",  np.nan)
    fpg_to    = to_profile.get("FPG_mean",    np.nan)

    if pd.isna(yprr_from) or pd.isna(yprr_to) or yprr_to == 0:
        return 1.0

    yprr_ratio = yprr_from / yprr_to

    if not pd.isna(fpg_from) and not pd.isna(fpg_to) and fpg_to > 0:
        fpg_ratio = fpg_from / fpg_to
        raw_ratio = 0.7 * yprr_ratio + 0.3 * fpg_ratio
    else:
        raw_ratio = yprr_ratio

    delta = (raw_ratio - 1.0) * translation
    delta = np.clip(delta, -0.40, +0.40)
    return 1.0 + delta


def _compute_scheme_adjustments(season_stats: pd.DataFrame,
                                  transfer: dict) -> pd.DataFrame:
    """
    For one transfer definition, return a DataFrame of
    (Name, POS, Team, scheme_adj, adj_note).

    If transfer['is_player_move'] is True, the ratio is inverted:
      new_team / old_team  (positive = moving to a better scheme environment).
    Otherwise, the standard OC-move ratio is used:
      oc_origin / destination_prior.
    """
    from_team       = transfer["from_team"]
    from_years      = transfer["from_years"]
    to_team         = transfer["to_team"]
    prior_years     = transfer["prior_years"]
    translation     = transfer["translation"]
    oc_name         = transfer["oc_name"]
    weights         = transfer["player_weights"]
    is_player_move  = transfer.get("is_player_move", False)

    rows = []
    for player_name, profile_weight in weights.items():
        if profile_weight <= 0:
            continue

        # ── Determine position ────────────────────────────────────────────────
        if is_player_move:
            # Player is new to to_team; look them up at their old team first
            p_rows = season_stats[
                (season_stats["Name"] == player_name) &
                (season_stats["Team"] == from_team)
            ].sort_values("Year", ascending=False)
        else:
            p_rows = season_stats[
                (season_stats["Name"] == player_name) &
                (season_stats["Team"] == to_team)
            ].sort_values("Year", ascending=False)

        if p_rows.empty:
            p_rows = season_stats[
                season_stats["Name"] == player_name
            ].sort_values("Year", ascending=False)

        pos = p_rows["POS"].iloc[0] if not p_rows.empty else "WR"

        # ── Compute scheme profiles ───────────────────────────────────────────
        if is_player_move:
            # Positive = new team (to_team) is more efficient than old team (from_team)
            new_profile = _team_scheme_profile(season_stats, to_team,   prior_years, pos)
            old_profile = _team_scheme_profile(season_stats, from_team, from_years,  pos)
            scheme_mult = _compute_scheme_multiplier(new_profile, old_profile, translation)
            note = (f"{oc_name}: "
                    f"{to_team} YPRR {new_profile.get('YPRR_mean', 0):.2f} vs "
                    f"{from_team} {old_profile.get('YPRR_mean', 0):.2f}, "
                    f"fit={profile_weight:.0%}, translation={translation:.0%}")
        else:
            src_profile  = _team_scheme_profile(season_stats, from_team, from_years,  pos)
            dest_profile = _team_scheme_profile(season_stats, to_team,   prior_years, pos)
            scheme_mult  = _compute_scheme_multiplier(src_profile, dest_profile, translation)
            note = (f"{oc_name}: "
                    f"YPRR {src_profile.get('YPRR_mean', 0):.2f} vs "
                    f"{dest_profile.get('YPRR_mean', 0):.2f}, "
                    f"fit={profile_weight:.0%}, translation={translation:.0%}")

        adj_pct = (scheme_mult - 1.0) * profile_weight

        rows.append({
            "Name":       player_name,
            "POS":        pos,
            "Team":       to_team,
            "scheme_adj": round(adj_pct, 4),
            "adj_note":   note,
        })

    return pd.DataFrame(rows)


# ── Public API ────────────────────────────────────────────────────────────────

def build_all_adjustments(season_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ALL adjustments (scheme transfers + player team changes + manual overrides).

    Returns a DataFrame with columns:
      Name, POS, Team, scheme_adj, manual_adj, total_adj, adj_note
    """
    adj_frames = []

    for transfer in SCHEME_TRANSFERS:
        df = _compute_scheme_adjustments(season_stats, transfer)
        if not df.empty:
            adj_frames.append(df)

    for move in PLAYER_TEAM_CHANGES:
        df = _compute_scheme_adjustments(season_stats, move)
        if not df.empty:
            adj_frames.append(df)

    if adj_frames:
        raw = pd.concat(adj_frames, ignore_index=True)
        # Keep Team for all_names lookup (first occurrence per player)
        scheme_adjs = (raw.groupby(["Name", "POS", "Team"], as_index=False)
                          .agg(scheme_adj=("scheme_adj", "sum"),
                               adj_note=("adj_note", " | ".join)))
        # Aggregate by Name+POS only to avoid duplicate rows on merge
        scheme_adjs_agg = (raw.groupby(["Name", "POS"], as_index=False)
                              .agg(scheme_adj=("scheme_adj", "sum"),
                                   adj_note=("adj_note", " | ".join)))
    else:
        scheme_adjs     = pd.DataFrame(columns=["Name", "POS", "Team",
                                                 "scheme_adj", "adj_note"])
        scheme_adjs_agg = pd.DataFrame(columns=["Name", "POS",
                                                 "scheme_adj", "adj_note"])

    manual_rows = []
    for name, pos, team, pct, reason in MANUAL_OVERRIDES:
        manual_rows.append({
            "Name": name, "POS": pos, "Team": team,
            "manual_adj": pct, "manual_note": reason
        })
    manual_adjs = (pd.DataFrame(manual_rows)
                   if manual_rows else
                   pd.DataFrame(columns=["Name", "POS", "Team",
                                          "manual_adj", "manual_note"]))

    if scheme_adjs.empty and manual_adjs.empty:
        return pd.DataFrame()

    all_names = pd.concat([
        scheme_adjs[["Name", "POS", "Team"]],
        manual_adjs[["Name", "POS", "Team"]] if not manual_adjs.empty else pd.DataFrame()
    ]).drop_duplicates(subset=["Name", "POS"])

    # Use Name+POS aggregated scheme_adjs to prevent row duplication on merge
    merged = all_names.merge(
        scheme_adjs_agg[["Name", "POS", "scheme_adj", "adj_note"]],
        on=["Name", "POS"], how="left"
    )
    if not manual_adjs.empty:
        merged = merged.merge(
            manual_adjs[["Name", "POS", "manual_adj", "manual_note"]],
            on=["Name", "POS"], how="left"
        )
    else:
        merged["manual_adj"]  = 0.0
        merged["manual_note"] = ""

    merged["scheme_adj"] = merged["scheme_adj"].fillna(0.0)
    merged["manual_adj"] = merged["manual_adj"].fillna(0.0)
    merged["adj_note"]   = merged["adj_note"].fillna("")

    merged["total_adj"] = (
        (1 + merged["scheme_adj"]) * (1 + merged["manual_adj"]) - 1
    ).round(4)

    merged["adj_note"] = merged.apply(
        lambda r: " | ".join(filter(None, [
            str(r.get("adj_note", "") or ""),
            (f"Manual: {r['manual_note']} ({r['manual_adj']:+.0%})"
             if r.get("manual_note") else "")
        ])),
        axis=1
    )

    return merged[["Name", "POS", "Team", "scheme_adj",
                   "manual_adj", "total_adj", "adj_note"]]


def apply_all_adjustments(projections: pd.DataFrame,
                           season_stats: pd.DataFrame,
                           verbose: bool = True) -> pd.DataFrame:
    """
    Apply all adjustments to the projections DataFrame.
    Adds: Scheme_Adj, Manual_Adj, Total_Adj, Adj_Note, Proj_FP_Adj.
    Updates the Team column for players who changed teams.
    Recomputes POS_Rank and Overall_Rank from Proj_FP_Adj.
    """
    # ── Apply team updates first ──────────────────────────────────────────────
    for (player_name, pos), new_team in TEAM_UPDATES_2026.items():
        mask = (projections["Name"] == player_name) & (projections["POS"] == pos)
        if mask.any():
            projections.loc[mask, "Team"] = new_team

    # ── Compute and merge adjustments ────────────────────────────────────────
    adjs = build_all_adjustments(season_stats)
    if adjs.empty:
        projections["Scheme_Adj"]  = 0.0
        projections["Manual_Adj"]  = 0.0
        projections["Total_Adj"]   = 0.0
        projections["Adj_Note"]    = ""
        projections["Proj_FP_Adj"] = projections["Proj_FP_2026"]
        return projections

    proj = projections.merge(
        adjs[["Name", "POS", "scheme_adj", "manual_adj", "total_adj", "adj_note"]],
        on=["Name", "POS"], how="left"
    )
    proj["scheme_adj"] = proj["scheme_adj"].fillna(0.0)
    proj["manual_adj"] = proj["manual_adj"].fillna(0.0)
    proj["total_adj"]  = proj["total_adj"].fillna(0.0)
    proj["adj_note"]   = proj["adj_note"].fillna("")

    proj = proj.rename(columns={
        "scheme_adj": "Scheme_Adj",
        "manual_adj": "Manual_Adj",
        "total_adj":  "Total_Adj",
        "adj_note":   "Adj_Note",
    })

    proj["Proj_FP_Adj"] = (
        proj["Proj_FP_2026"] * (1 + proj["Total_Adj"])
    ).round(1)

    proj["POS_Rank"] = (
        proj.groupby("POS")["Proj_FP_Adj"]
            .rank(ascending=False, method="min")
            .astype(int)
    )
    proj["Overall_Rank"] = (
        proj["Proj_FP_Adj"]
            .rank(ascending=False, method="min")
            .astype(int)
    )

    if verbose:
        adjusted = proj[proj["Total_Adj"] != 0].sort_values("Total_Adj", ascending=False)
        if not adjusted.empty:
            print("\n  Expert adjustments applied:")
            print(f"  {'Player':<24} {'POS':<4} {'Team':<4} {'Base':>7} {'Adj%':>7}"
                  f"  {'Adj FP':>7}  Note")
            print(f"  {'-'*24} {'-'*4} {'-'*4} {'-'*7} {'-'*7}  {'-'*7}  {'-'*40}")
            for _, r in adjusted.iterrows():
                print(f"  {r['Name']:<24} {r['POS']:<4} {r['Team']:<4}"
                      f" {r['Proj_FP_2026']:>7.1f} {r['Total_Adj']:>+7.1%}"
                      f"  {r['Proj_FP_Adj']:>7.1f}  {r['Adj_Note'][:65]}")

    return proj
