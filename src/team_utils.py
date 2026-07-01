"""
team_utils.py
Team abbreviation <-> full name mapping and schedule helpers.
FantasyPoints uses non-standard abbreviations (BLT, HST, CLV, ARZ).
"""

# FantasyPoints abbreviation -> NFL full team name
ABBREV_TO_FULL = {
    'ARZ': 'Arizona Cardinals',
    'ATL': 'Atlanta Falcons',
    'BLT': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills',
    'CAR': 'Carolina Panthers',
    'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals',
    'CLV': 'Cleveland Browns',
    'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos',
    'DET': 'Detroit Lions',
    'GB':  'Green Bay Packers',
    'HST': 'Houston Texans',
    'IND': 'Indianapolis Colts',
    'JAX': 'Jacksonville Jaguars',
    'KC':  'Kansas City Chiefs',
    'LA':  'Los Angeles Rams',
    'LAC': 'Los Angeles Chargers',
    'LV':  'Las Vegas Raiders',
    'MIA': 'Miami Dolphins',
    'MIN': 'Minnesota Vikings',
    'NE':  'New England Patriots',
    'NO':  'New Orleans Saints',
    'NYG': 'New York Giants',
    'NYJ': 'New York Jets',
    'PHI': 'Philadelphia Eagles',
    'PIT': 'Pittsburgh Steelers',
    'SEA': 'Seattle Seahawks',
    'SF':  'San Francisco 49ers',
    'TB':  'Tampa Bay Buccaneers',
    'TEN': 'Tennessee Titans',
    'WAS': 'Washington Commanders',
}

FULL_TO_ABBREV = {v: k for k, v in ABBREV_TO_FULL.items()}


def abbrev_to_full(abbrev: str) -> str:
    return ABBREV_TO_FULL.get(str(abbrev).strip(), abbrev)


def full_to_abbrev(full_name: str) -> str:
    return FULL_TO_ABBREV.get(str(full_name).strip(), full_name)
