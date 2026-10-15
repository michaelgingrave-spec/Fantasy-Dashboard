"""The Data Suite pull recipe: week batches must filter on the server (regWeeks) so the site's 1,500-row cap cannot cut the quiet games."""
from matchup_model import weekly_pull as W


def test_backfill_week_batches_filter_by_regWeeks_not_a_client_side_split_value():
    rec = [t for t in W.backfill_tables(2025) if t["name"].startswith("receiving/")]
    assert len(rec) == 9                                                    # 3 receiving tables x 3 week batches
    for t in rec:
        assert "regWeeks=" in t["url"] and "splitValues" not in t["url"]
    batches = {t["dest"] for t in rec if t["name"] == "receiving/advanced"}
    assert batches == {"receiving-advanced_2025wk1-6_week.csv", "receiving-advanced_2025wk7-12_week.csv", "receiving-advanced_2025wk13-18_week.csv"}
    wk = next(t for t in rec if t["dest"] == "receiving-manvszone_2025wk7-12_week.csv")
    assert "regWeeks=7,8,9,10,11,12" in wk["url"] and "positions=WR,TE" in wk["url"] and wk["url"].startswith(W.BASE + "/receiving/man-vs-zone")


def test_single_week_tables_use_regWeeks_too():
    assert all(f"regWeeks=4" in t["url"] for t in W.tables(2026, 4))
