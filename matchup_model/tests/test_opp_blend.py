"""Unit tests for the FantasyPoints projection nudge (blend._fp_shift / _fp_projections).
Pure-function logic — no nflverse cache needed, unlike the rest of matchup_model/tests."""
import pandas as pd
import pytest

from matchup_model.opp import blend as B


def test_fp_shift_no_signal_below_floor():
    # role_ratio 1.10 -> dev 0.10, under FP_DEV_FLOOR (0.12) -> untouched
    adj, rr = B._fp_shift(10.0, 11.0)
    assert adj == 10.0
    assert rr == pytest.approx(1.1)


def test_fp_shift_ramps_linearly_mid_range():
    # role_ratio 1.30 -> dev 0.30 -> shift = (0.30-0.12)/0.40 = 0.45
    adj, rr = B._fp_shift(10.0, 13.0)
    shift = (0.30 - B.FP_DEV_FLOOR) / B.FP_DEV_SPAN
    assert shift < B.FP_SHIFT_CAP                    # sanity: this case shouldn't be capped
    expected = (1 - shift) * 10.0 + shift * 13.0
    assert adj == pytest.approx(expected)
    assert rr == pytest.approx(1.3)


def test_fp_shift_caps_at_wide_mismatch():
    # role_ratio 2.0 -> dev 1.0, way past the ramp -> capped at FP_SHIFT_CAP, never 100%
    adj, rr = B._fp_shift(10.0, 20.0)
    expected = (1 - B.FP_SHIFT_CAP) * 10.0 + B.FP_SHIFT_CAP * 20.0
    assert adj == pytest.approx(expected)
    assert adj < 20.0                                 # never fully replaces our own number


def test_fp_shift_symmetric_for_under_projection():
    # FantasyPoints LOWER than us should pull down the same way it pulls up
    adj, rr = B._fp_shift(10.0, 5.0)
    assert adj < 10.0
    assert rr == pytest.approx(0.5)


def test_fp_shift_no_fp_number():
    adj, rr = B._fp_shift(10.0, None)
    assert adj == 10.0 and rr is None


def test_fp_shift_guards_tiny_base():
    # near-zero base_fp would make role_ratio meaningless/unstable — skip the nudge
    adj, rr = B._fp_shift(0.3, 5.0)
    assert adj == 0.3 and rr is None


def test_fp_projections_missing_file_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "dfs.projections.load_weekly_projections",
        lambda *_a, **_k: (_ for _ in ()).throw(FileNotFoundError("no file")),
    )
    B._fp_projections.cache_clear()
    try:
        assert B._fp_projections(999) == {}
    finally:
        B._fp_projections.cache_clear()


def test_fp_projections_keys_by_normalized_name(monkeypatch):
    monkeypatch.setattr(
        "dfs.projections.load_weekly_projections",
        lambda *_a, **_k: pd.DataFrame({"name": ["A.J. Brown"], "proj": [18.4]}),
    )
    B._fp_projections.cache_clear()
    try:
        out = B._fp_projections(1)
        assert out == {"aj brown": 18.4}
    finally:
        B._fp_projections.cache_clear()
