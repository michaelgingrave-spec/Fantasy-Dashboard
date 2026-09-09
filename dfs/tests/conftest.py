"""Make the project root importable and expose shared fixtures."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def getcontests_json():
    return json.loads((FIX / "getcontests.json").read_text())


@pytest.fixture
def draftables_json():
    return json.loads((FIX / "draftables.json").read_text())


@pytest.fixture
def salaries_df(draftables_json):
    from dfs.dk_client import parse_draftables
    return parse_draftables(draftables_json, 153069)


@pytest.fixture
def projections_df():
    from dfs.projections import load_weekly_projections
    return load_weekly_projections(1, FIX / "projections.week1.sample.csv")


@pytest.fixture
def slate_df(salaries_df, projections_df):
    from dfs.slate import build_slate
    slate, _ = build_slate(salaries_df, projections_df)
    return slate
