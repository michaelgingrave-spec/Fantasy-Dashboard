from dfs.names import fuzzy_match, name_team_key, norm_team, normalize_name, player_key


def test_team_crosswalk_fantasypoints_to_dk():
    assert norm_team("ARZ") == "ARI"
    assert norm_team("BLT") == "BAL"
    assert norm_team("CLV") == "CLE"
    assert norm_team("HST") == "HOU"
    assert norm_team("LA") == "LAR"
    assert norm_team("JAX") == "JAX"


def test_team_crosswalk_full_names_and_passthrough():
    assert norm_team("Philadelphia Eagles") == "PHI"
    assert norm_team("San Francisco 49ers") == "SF"
    assert norm_team("KC") == "KC"


def test_normalize_name_suffixes_and_punctuation():
    assert normalize_name("Patrick Mahomes II") == "patrick mahomes"
    assert normalize_name("Michael Pittman Jr.") == "michael pittman"
    assert normalize_name("Ja'Marr Chase") == "jamarr chase"
    assert normalize_name("A.J. Brown") == "aj brown"


def test_player_key_and_dst_alignment():
    assert player_key("Justin Jefferson", "MIN", "WR") == "justin jefferson MIN"
    a = player_key("Eagles", "PHI", "DST")
    b = player_key("Philadelphia Eagles", "", "DST")
    c = player_key("Eagles", "", "DEF")
    assert a == b == c == "dst PHI"


def test_name_team_key_ignores_position():
    assert name_team_key("Bijan Robinson", "ATL") == "bijan robinson ATL"


def test_fuzzy_match_threshold():
    cands = ["deandre hopkins", "davante adams", "stefon diggs"]
    assert fuzzy_match("deandre hopkins", cands) == "deandre hopkins"
    assert fuzzy_match("totally different name", cands) is None
