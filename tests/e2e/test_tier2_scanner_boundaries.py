"""
Tier 2 Boundary & Corner Cases — Scanner & Filters (F1–F7): 35 tests (5 per feature).
Opaque-box boundary value analysis and degenerate/malformed input guards.
"""

from datetime import date, datetime
import math
import numpy as np
import pandas as pd
import pytest

from day_option_bot.src import fibo_scanner
from day_option_bot.tests.e2e.conftest import (
    call_check_earnings,
    make_intraday_2m_df,
)


# ===========================================================================
# F1 Boundaries: TF 2m EMA 200 Trend Filter (5 tests)
# ===========================================================================

def test_t2_f01_insufficient_bars_under_200_returns_no_data(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, prev_bars=125, curr_bars=25)
    assert len(df) == 150
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["status"] == "NO_DATA"


def test_t2_f01_exact_200_bars_minimum_warmup_succeeds(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, prev_bars=175, curr_bars=25)
    assert len(df) == 200
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["status"] != "NO_DATA"
    assert res["trend_direction"] == "DOWNTREND"


def test_t2_f01_no_silent_5m_fallback_when_2m_empty(sample_config, mock_yfinance_factory):
    valid_5m = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18)
    call_log = mock_yfinance_factory({
        "AMD": {
            "history_2m": pd.DataFrame(),
            "history_5m": valid_5m,
            "calendar": {},
        }
    })

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["status"] == "NO_DATA"
    assert not any(c.get("interval") == "5m" for c in call_log)


def test_t2_f01_nan_in_last_close_handled_as_neutral_or_no_data(
    sample_config, mock_yfinance_factory, monkeypatch
):
    # (a) 175 prev + 25 curr + NAN_CLOSE -> dropping NaN bar leaves 199 < 200 -> NO_DATA
    df_200 = make_intraday_2m_df(
        trend="DOWNTREND", active_level=106.18, prev_bars=175, curr_bars=25, candle_mode="NAN_CLOSE"
    )
    assert len(df_200) == 200
    mock_yfinance_factory({"AMD": {"history_2m": df_200, "calendar": {}}})

    res_a = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res_a["status"] == "NO_DATA"
    assert res_a.get("option_action") not in ("BUY CALL", "BUY PUT")

    # (b) When ensure_ny_regular_hours yields a 220-bar DataFrame whose final Close is np.nan,
    # fibo_scanner's NaN Close/EMA200 guard returns NO_DATA or NEUTRAL (never BUY CALL / BUY PUT)
    df_220 = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="CONFIRMED_REVERSAL")
    assert len(df_220) == 220
    df_220.iloc[-1, df_220.columns.get_loc("Close")] = np.nan
    monkeypatch.setattr(fibo_scanner, "ensure_ny_regular_hours", lambda raw_df: df_220.copy())
    mock_yfinance_factory({"AMD": {"history_2m": df_220, "calendar": {}}})

    res_b = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res_b["status"] == "NO_DATA" or res_b.get("trend_direction") == "NEUTRAL"
    assert res_b.get("option_action") not in ("BUY CALL", "BUY PUT")


def test_t2_f01_spot_one_cent_below_vs_above_ema200(sample_config, mock_yfinance_factory):
    # Flat 195 prev + 24 curr at 105.00 -> EMA200 == 105.00; last bar 104.99 vs 105.01
    df_below = make_intraday_2m_df(trend="NEUTRAL")
    df_below.iloc[-1, df_below.columns.get_loc("Close")] = 104.99
    df_below.iloc[-1, df_below.columns.get_loc("Low")] = 104.95

    df_above = make_intraday_2m_df(trend="NEUTRAL")
    df_above.iloc[-1, df_above.columns.get_loc("Close")] = 105.01
    df_above.iloc[-1, df_above.columns.get_loc("High")] = 105.05

    mock_yfinance_factory({"AMD": {"history_2m": df_below, "calendar": {}}})
    res_below = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res_below["trend_direction"] == "DOWNTREND"
    assert res_below["option_action"] == "BUY PUT"

    mock_yfinance_factory({"AMD": {"history_2m": df_above, "calendar": {}}})
    res_above = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res_above["trend_direction"] == "UPTREND"
    assert res_above["option_action"] == "BUY CALL"


# ===========================================================================
# F2 Boundaries: Previous-Day Asymmetric Fibo (5 tests)
# ===========================================================================

def test_t2_f02_single_trading_day_only_returns_no_data(sample_config, mock_yfinance_factory):
    idx = pd.date_range("2026-11-18 09:30", "2026-11-18 15:58", freq="2min", tz="America/New_York")
    df_one_day = pd.DataFrame(
        {"Open": 105.0, "High": 106.0, "Low": 104.0, "Close": 105.0, "Volume": 1000.0},
        index=idx,
    )
    mock_yfinance_factory({"AMD": {"history_2m": df_one_day, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["status"] == "NO_DATA"


def test_t2_f02_prev_day_too_few_bars_rejected(sample_config, mock_yfinance_factory):
    # D-2 has 195 bars (warmup), D-1 (prev_date) has only 4 bars, D has 25 bars
    d_m2 = make_intraday_2m_df(prev_date="2026-11-13", curr_date="2026-11-16", trend="DOWNTREND", active_level=106.18, curr_bars=195)
    d_m2 = d_m2[d_m2.index.date == date(2026, 11, 16)]

    d_m1_idx = pd.date_range("2026-11-17 09:30", periods=4, freq="2min", tz="America/New_York")
    d_m1 = pd.DataFrame(
        {"Open": 105.0, "High": 112.0, "Low": 100.0, "Close": 106.0, "Volume": 1000.0},
        index=d_m1_idx,
    )
    d_curr = make_intraday_2m_df(prev_date="2026-11-16", curr_date="2026-11-18", trend="DOWNTREND", active_level=106.18)
    d_curr = d_curr[d_curr.index.date == date(2026, 11, 18)]

    df = pd.concat([d_m2, d_m1, d_curr]).sort_index()
    assert len(df) >= 200
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["status"] == "NO_DATA"


def test_t2_f02_flat_zero_range_prev_day_rejected(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(trend="NEUTRAL")
    prev_mask = df.index.date == date(2026, 11, 17)
    df.loc[prev_mask, ["Open", "High", "Low", "Close"]] = 105.00
    # Current day drops slightly below 105.00 while prev_df remains completely flat 105.00
    curr_mask = df.index.date == date(2026, 11, 18)
    df.loc[curr_mask, "Close"] = 104.50
    df.loc[curr_mask, "Low"] = 104.40
    df.loc[curr_mask, "High"] = 104.60
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["status"] == "NO_DATA"


def test_t2_f02_sub_threshold_tiny_range_rejected(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(trend="NEUTRAL")
    prev_mask = df.index.date == date(2026, 11, 17)
    # 0.10 range on $105 stock (~0.095% < 0.8% min range)
    df.loc[prev_mask, "High"] = 105.05
    df.loc[prev_mask, "Low"] = 104.95
    curr_mask = df.index.date == date(2026, 11, 18)
    df.loc[curr_mask, "Close"] = 104.00
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["status"] == "NO_DATA"


def test_t2_f02_doji_and_extreme_wick_anchor_asymmetry():
    prev_df = pd.DataFrame({
        "Open": [102.0, 108.0, 105.0],
        "High": [104.0, 118.0, 105.5],
        "Low": [95.0, 106.0, 104.5],
        "Close": [103.0, 107.0, 105.0],
        "BodyHigh": [103.0, 108.0, 105.0],
        "BodyLow": [102.0, 107.0, 105.0],
    })
    anchors = fibo_scanner.extract_prev_day_fibo_anchors(prev_df)
    assert anchors["wick_low"] == 95.0
    assert anchors["body_low"] == 102.0
    assert anchors["wick_high"] == 118.0
    assert anchors["body_high"] == 108.0


# ===========================================================================
# F3 Boundaries: Entry Levels & Smart Routing (5 tests)
# ===========================================================================

def test_t2_f03_etf_near_0618_with_rvol_149_stays_on_0500(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, rvol_target=1.49)
    mock_yfinance_factory({"SPY": {"history_2m": df}})

    res = fibo_scanner.analyze_single_ticker("SPY", sample_config)
    assert res["primary_fibo_level"] == "0.500"
    assert res["active_fibo_level"] == "0.500"


def test_t2_f03_stock_near_0500_with_rvol_149_stays_on_0618(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=105.00, rvol_target=1.49)
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["primary_fibo_level"] == "0.618"
    assert res["active_fibo_level"] == "0.618"


def test_t2_f03_secondary_disabled_blocks_even_at_rvol_300(sample_config, mock_yfinance_factory):
    sample_config["strategy_rules"]["smart_fibo_routing"]["allow_secondary_with_high_rvol"] = False
    df_spy = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, rvol_target=3.00)
    df_amd = make_intraday_2m_df(trend="DOWNTREND", active_level=105.00, rvol_target=3.00)
    mock_yfinance_factory({
        "SPY": {"history_2m": df_spy},
        "AMD": {"history_2m": df_amd, "calendar": {}},
    })

    assert fibo_scanner.analyze_single_ticker("SPY", sample_config)["active_fibo_level"] == "0.500"
    assert fibo_scanner.analyze_single_ticker("AMD", sample_config)["active_fibo_level"] == "0.618"


def test_t2_f03_exact_150_rvol_boundary_allows_secondary(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, rvol_target=1.50)
    mock_yfinance_factory({"SPY": {"history_2m": df}})

    res = fibo_scanner.analyze_single_ticker("SPY", sample_config)
    assert res["rvol"] >= 1.50
    assert res["active_fibo_level"] == "0.618"


def test_t2_f03_unknown_ticker_defaults_to_stock_0618_primary(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(trend="UPTREND", active_level=105.20, rvol_target=1.30)
    mock_yfinance_factory({"MSFT": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("MSFT", sample_config)
    assert res["is_etf"] is False
    assert res["primary_fibo_level"] == "0.618"


# ===========================================================================
# F4 Boundaries: Volume RVOL Filter (5 tests)
# ===========================================================================

def test_t2_f04_exact_115_rvol_passes(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="CONFIRMED_REVERSAL", rvol_target=1.151)
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["rvol"] >= 1.15
    assert res["rvol_passed"] is True
    assert res["status"] == "ACTIONABLE"


def test_t2_f04_rvol_114_fails(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="CONFIRMED_REVERSAL", rvol_target=1.135)
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["rvol"] <= 1.14
    assert res["rvol_passed"] is False
    assert res["status"] != "ACTIONABLE"


def test_t2_f04_zero_volume_all_bars_no_zero_division(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="CONFIRMED_REVERSAL")
    df["Volume"] = 0.0
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert not math.isnan(res["rvol"])
    assert res["rvol"] <= 1.0
    assert res["rvol_passed"] is False


def test_t2_f04_nan_in_last_bar_volume_safe(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="CONFIRMED_REVERSAL", rvol_target=1.00)
    df.iloc[-2, df.columns.get_loc("Volume")] = 1400.0
    df.iloc[-1, df.columns.get_loc("Volume")] = np.nan
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert not math.isnan(res["rvol"])
    assert res["rvol"] >= 1.30
    assert res["rvol_passed"] is True


def test_t2_f04_both_last_bars_nan_volume_does_not_crash(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="CONFIRMED_REVERSAL")
    df.iloc[-2, df.columns.get_loc("Volume")] = np.nan
    df.iloc[-1, df.columns.get_loc("Volume")] = np.nan
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["status"] != "NO_DATA"
    assert not math.isnan(res["rvol"])
    assert res["rvol_passed"] is False


# ===========================================================================
# F5 Boundaries: Reversal Candlestick Gate (5 tests)
# ===========================================================================

def test_t2_f05_untouched_bearish_candle_never_labeled_reversal():
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="NO_TOUCH", spot_override=102.00)
    curr_df = df.iloc[-25:]
    res = fibo_scanner.detect_reversal_candle(curr_df, "DOWNTREND", 106.18, 10.0)
    assert res["touched"] is False
    assert res["confirmed"] is False
    assert "Reversal" not in res["pattern_name"]
    assert "Shooting Star" not in res["pattern_name"]


def test_t2_f05_untouched_bullish_candle_never_labeled_reversal():
    df = make_intraday_2m_df(trend="UPTREND", active_level=105.20, candle_mode="NO_TOUCH", spot_override=109.50)
    curr_df = df.iloc[-25:]
    res = fibo_scanner.detect_reversal_candle(curr_df, "UPTREND", 105.20, 11.0)
    assert res["touched"] is False
    assert res["confirmed"] is False
    assert "Reversal" not in res["pattern_name"]
    assert "Hammer" not in res["pattern_name"]


def test_t2_f05_zero_range_doji_bar_no_zero_division():
    curr_df = pd.DataFrame([
        {"Open": 106.0, "High": 106.1, "Low": 105.9, "Close": 106.0},
        {"Open": 106.18, "High": 106.18, "Low": 106.18, "Close": 106.18},
    ])
    res = fibo_scanner.detect_reversal_candle(curr_df, "DOWNTREND", 106.18, 10.0)
    assert res["touched"] is True
    assert res["confirmed"] is False


def test_t2_f05_wick_ratio_044_vs_045_boundary():
    # Shooting star test with Open=106.00, Close=105.98 (does NOT engulf prev_low=105.90, wait: Close=106.19 > lvl=106.18)
    # Wait: is_bearish_engulf = is_bearish_body and (c_close <= p_low or c_close <= level_price)
    # Use c_close > level_price and c_close > p_low (e.g. level=106.18, p_low=106.00, tol=0.20, c_close=106.20 <= level+tol*0.5=106.28)
    # Bar range = 1.00, c_high = 106.70, c_low = 105.70 -> c_open = 106.26 (wick = 106.70 - 106.26 = 0.44) vs 106.25 (wick = 0.45)
    df_44 = pd.DataFrame([
        {"Open": 106.05, "High": 106.20, "Low": 106.00, "Close": 106.15},
        {"Open": 106.26, "High": 106.70, "Low": 105.70, "Close": 106.20},
    ])
    df_45 = pd.DataFrame([
        {"Open": 106.05, "High": 106.20, "Low": 106.00, "Close": 106.15},
        {"Open": 106.25, "High": 106.70, "Low": 105.70, "Close": 106.20},
    ])
    res_44 = fibo_scanner.detect_reversal_candle(df_44, "DOWNTREND", 106.18, 10.0)
    res_45 = fibo_scanner.detect_reversal_candle(df_45, "DOWNTREND", 106.18, 10.0)
    assert res_44["confirmed"] is False
    assert res_45["confirmed"] is True


def test_t2_f05_single_bar_curr_df_no_index_error():
    curr_df = pd.DataFrame([{"Open": 106.15, "High": 106.55, "Low": 106.05, "Close": 106.10}])
    res = fibo_scanner.detect_reversal_candle(curr_df, "DOWNTREND", 106.18, 10.0)
    assert isinstance(res["touched"], bool)
    assert isinstance(res["confirmed"], bool)


# ===========================================================================
# F6 Boundaries: Morning Gap Filter (5 tests)
# ===========================================================================

def test_t2_f06_exact_200_gap_not_skipped(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(
        trend="DOWNTREND", active_level=106.18, candle_mode="NO_TOUCH",
        morning_gap_pct=2.00, spot_override=101.00,
    )
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["morning_gap_pct"] == 2.00
    assert res["status"] != "GAP_SKIP"


def test_t2_f06_201_gap_and_dist_151_triggers_gap_skip(sample_config, mock_yfinance_factory):
    # At 106.18 active level, spot=104.60 -> dist = 1.58/104.60*100 = 1.51% > 1.50%
    df = make_intraday_2m_df(
        trend="DOWNTREND", active_level=106.18, candle_mode="NO_TOUCH",
        morning_gap_pct=2.01, spot_override=104.60, rvol_target=1.30,
    )
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["morning_gap_pct"] > 2.00
    assert res["distance_to_active_fibo_pct"] == 1.51
    assert res["status"] == "GAP_SKIP"


def test_t2_f06_201_gap_and_dist_150_not_skipped(sample_config, mock_yfinance_factory):
    # At 106.18 active level, spot=104.61 -> dist = 1.57/104.61*100 = 1.5008% -> 1.50 <= 1.50
    df = make_intraday_2m_df(
        trend="DOWNTREND", active_level=106.18, candle_mode="NO_TOUCH",
        morning_gap_pct=2.01, spot_override=104.61, rvol_target=1.30,
    )
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["morning_gap_pct"] > 2.00
    assert res["distance_to_active_fibo_pct"] <= 1.50
    assert res["status"] != "GAP_SKIP"


def test_t2_f06_gap_down_uses_abs_magnitude(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(
        trend="DOWNTREND", active_level=106.18, candle_mode="NO_TOUCH",
        morning_gap_pct=3.5, gap_direction="DOWN", spot_override=101.00,
    )
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["morning_gap_pct"] == pytest.approx(3.5, abs=0.05)
    assert res["status"] == "GAP_SKIP"


def test_t2_f06_zero_prev_close_guarded(sample_config, mock_yfinance_factory, monkeypatch):
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="CONFIRMED_REVERSAL")
    # Set final bar of prev_df Close to 0.0
    prev_last_idx = 194
    df.iloc[prev_last_idx, df.columns.get_loc("Close")] = 0.0
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    # (a) End-to-end through real ensure_ny_regular_hours
    res_e2e = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert isinstance(res_e2e["morning_gap_pct"], float)
    assert math.isfinite(res_e2e["morning_gap_pct"])

    # (b) When ensure_ny_regular_hours preserves prev_close=0.0, fibo_scanner's zero-close guard runs without ZeroDivisionError
    monkeypatch.setattr(fibo_scanner, "ensure_ny_regular_hours", lambda raw_df: df.copy())
    res_preserved = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert isinstance(res_preserved["morning_gap_pct"], float)
    assert math.isfinite(res_preserved["morning_gap_pct"])
    assert res_preserved["morning_gap_pct"] > 0.0


# ===========================================================================
# F7 Boundaries: Earnings Blackout Filter (5 tests)
# ===========================================================================

def test_t2_f07_exact_blackout_window_boundaries(mock_yfinance_factory):
    ref = date(2026, 11, 18)
    for diff, expected in ((-2, False), (-1, True), (0, True), (1, True), (2, True), (3, False)):
        earn_dt = date(2026, 11, 18 + diff)
        mock_yfinance_factory({
            "AMD": {"calendar": {"Earnings Date": [datetime(earn_dt.year, earn_dt.month, earn_dt.day)]}}
        })
        chk = call_check_earnings("AMD", reference_date=ref, days_before=2, days_after=1)
        assert chk["is_blackout"] is expected, f"Failed for diff={diff}"


def test_t2_f07_dataframe_and_string_calendar_formats(mock_yfinance_factory):
    ref = date(2026, 11, 18)
    # 1. DataFrame with column "Earnings Date"
    df_col = pd.DataFrame({"Earnings Date": [pd.Timestamp("2026-11-19")]})
    mock_yfinance_factory({"AMD": {"calendar": df_col}})
    assert call_check_earnings("AMD", reference_date=ref)["is_blackout"] is True

    # 2. DataFrame indexed by "Earnings Date"
    df_idx = pd.DataFrame({0: [pd.Timestamp("2026-11-19")]}, index=["Earnings Date"])
    mock_yfinance_factory({"NVDA": {"calendar": df_idx}})
    assert call_check_earnings("NVDA", reference_date=ref)["is_blackout"] is True

    # 3. Dict with ISO string
    mock_yfinance_factory({"META": {"calendar": {"Earnings Date": ["2026-11-19"]}}})
    assert call_check_earnings("META", reference_date=ref)["is_blackout"] is True


def test_t2_f07_empty_none_or_exception_calendar_safe(mock_yfinance_factory):
    ref = date(2026, 11, 18)
    for cal in ({}, None, RuntimeError("Network timeout")):
        mock_yfinance_factory({"AMD": {"calendar": cal}})
        chk = call_check_earnings("AMD", reference_date=ref)
        assert chk["is_blackout"] is False


def test_t2_f07_ny_vs_bangkok_midnight_date_shift(mock_yfinance_factory, freeze_ny_time):
    # At 2026-11-18 19:05 NY (when UTC is 2026-11-19 00:05 and TH is 2026-11-19 07:05)
    freeze_ny_time(datetime(2026, 11, 18, 19, 5))
    mock_yfinance_factory({
        "NVDA": {"calendar": {"Earnings Date": [datetime(2026, 11, 21)]}}
    })
    chk = call_check_earnings("NVDA", days_before=2, days_after=1)
    # 2026-11-21 - 2026-11-18 == 3 days -> SAFE (not 2 days!)
    assert chk["days_until_earnings"] == 3
    assert chk["is_blackout"] is False


def test_t2_f07_lowercase_whitespace_ticker_normalized(mock_yfinance_factory):
    mock_yfinance_factory({
        "NVDA": {"calendar": {"Earnings Date": [datetime(2026, 11, 19)]}},
        "SPY": {"calendar": RuntimeError("404")},
    })
    chk_nvda = call_check_earnings(" nvda ", reference_date=date(2026, 11, 18))
    assert chk_nvda["ticker"] == "NVDA"
    assert chk_nvda["is_blackout"] is True

    chk_spy = call_check_earnings(" spy ", reference_date=date(2026, 11, 18))
    assert chk_spy["ticker"] == "SPY"
    assert chk_spy["is_blackout"] is False
