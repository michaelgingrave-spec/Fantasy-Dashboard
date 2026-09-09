import shutil
from pathlib import Path

from dfs.datasuite.features import defense_features, player_features
from dfs.datasuite.registry import discover, load_table

FIX = Path(__file__).parent / "fixtures"


def test_generic_adapter_detects_name_team_and_numeric_feats(tmp_path):
    shutil.copy(FIX / "datasuite_receiving_advanced.sample.csv",
                tmp_path / "receiving_advanced_player.csv")
    files = discover(tmp_path)
    assert len(files) == 1
    tbl = load_table(files[0])
    assert "nt_key" in tbl.columns
    assert tbl["nt_key"].str.strip().astype(bool).all()
    feat_cols = [c for c in tbl.columns if c.startswith("ds__")]
    assert len(feat_cols) >= 3  # routes, tgt share, yprr, adot, ez tgt...


def test_defense_scope_keyed_by_team(tmp_path):
    shutil.copy(FIX / "datasuite_defense_receiving.sample.csv",
                tmp_path / "defense_receiving_team.csv")
    feats, srcs = defense_features(tmp_path)
    assert "team_key" in feats.columns
    assert len(feats) >= 20
    assert any(c.startswith("ds__") for c in feats.columns)


def test_features_tolerate_missing_or_empty_folder(tmp_path):
    pf, ps = player_features(tmp_path)   # empty dir
    df, ds = defense_features(tmp_path)
    assert pf.empty and df.empty
    assert ps == [] and ds == []
