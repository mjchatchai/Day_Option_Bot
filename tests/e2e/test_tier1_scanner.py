"""
Tier 1 Feature Coverage — Scanner & Filters (F1–F7): 35 tests (5 per feature).
Opaque-box tests derived from ORIGINAL_REQUEST.md R1 and PROJECT.md F1–F7.
"""

from datetime import date, datetime
import pandas as pd
import pytest

from day_option_bot.src import earnings_filter, fibo_scanner
from day_option_bot.tests.e2e.conftest import (
    call_check_earnings,
    make_intraday_2m_df,
)


# ===========================================================================
# F1: TF 2m EMA 200 Trend Filter (5 tests)
# ===========================================================================

def test_t1_f01_downtrend_below_ema200_buy_put(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, rvol_target=1.30)
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["status"] != "NO_DATA"
    assert res["spot_price"] < res["ema200"]
    assert res["trend_direction"] == "DOWNTREND"
    assert res["option_action"] == "BUY PUT"


def test_t1_f01_uptrend_above_ema200_buy_call(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(trend="UPTREND", active_level=105.20, rvol_target=1.30)
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["status"] != "NO_DATA"
    assert res["spot_price"] > res["ema200"]
    assert res["trend_direction"] == "UPTREND"
    assert res["option_action"] == "BUY CALL"


def test_t1_f01_ema200_exact_calculation_matches_ewm(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, rvol_target=1.30)
    expected_ema = round(float(df["Close"].ewm(span=200, adjust=False).mean().iloc[-1]), 2)
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["ema200"] == pytest.approx(expected_ema, abs=0.01)


def test_t1_f01_neutral_when_spot_equals_ema200(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(trend="NEUTRAL")
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["status"] != "ACTIONABLE"
    assert res["trend_direction"] == "NEUTRAL"


def test_t1_f01_config_ema_period_respected(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, rvol_target=1.30)
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res_200 = fibo_scanner.analyze_single_ticker("AMD", sample_config)

    cfg_50 = sample_config
    cfg_50["strategy_rules"]["ema_period"] = 50
    res_50 = fibo_scanner.analyze_single_ticker("AMD", cfg_50)

    expected_50 = round(float(df["Close"].ewm(span=50, adjust=False).mean().iloc[-1]), 2)
    assert res_50["ema200"] == pytest.approx(expected_50, abs=0.01)
    assert res_50["ema200"] != res_200["ema200"]


# ===========================================================================
# F2: Previous-Day Asymmetric Fibo (5 tests)
# ===========================================================================

def test_t1_f02_downtrend_asymmetric_anchors_wick_low_to_body_high():
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18)
    ind = fibo_scanner.compute_intraday_indicators(df)
    dates = sorted(ind["Date"].unique())
    prev_df = ind[ind["Date"] == dates[-2]]

    anchors = fibo_scanner.extract_prev_day_fibo_anchors(prev_df)
    assert anchors["wick_low"] == 100.00
    assert anchors["body_low"] == 101.00
    assert anchors["body_high"] == 110.00
    assert anchors["wick_high"] == 112.00

    fibo = fibo_scanner.build_fibo_levels(anchors, "DOWNTREND")
    assert fibo["0.000"] == 100.00
    assert fibo["1.000"] == 110.00
    assert fibo["range_usd"] == 10.00


def test_t1_f02_uptrend_asymmetric_anchors_wick_high_to_body_low():
    df = make_intraday_2m_df(trend="UPTREND", active_level=105.20)
    ind = fibo_scanner.compute_intraday_indicators(df)
    dates = sorted(ind["Date"].unique())
    prev_df = ind[ind["Date"] == dates[-2]]

    anchors = fibo_scanner.extract_prev_day_fibo_anchors(prev_df)
    fibo = fibo_scanner.build_fibo_levels(anchors, "UPTREND")
    assert fibo["0.000"] == 112.00
    assert fibo["1.000"] == 101.00
    assert fibo["range_usd"] == 11.00


def test_t1_f02_all_fibo_ratios_computed_accurately():
    anchors = {"wick_low": 100.0, "body_low": 101.0, "wick_high": 112.0, "body_high": 110.0}

    down = fibo_scanner.build_fibo_levels(anchors, "DOWNTREND")
    for ratio in (0.236, 0.382, 0.500, 0.618, 0.786):
        assert down[f"{ratio:.3f}"] == pytest.approx(round(100.0 + ratio * 10.0, 2), abs=0.01)

    up = fibo_scanner.build_fibo_levels(anchors, "UPTREND")
    for ratio in (0.236, 0.382, 0.500, 0.618, 0.786):
        assert up[f"{ratio:.3f}"] == pytest.approx(round(112.0 - ratio * 11.0, 2), abs=0.01)


def test_t1_f02_uses_only_previous_trading_day_regular_hours(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(
        trend="DOWNTREND",
        active_level=106.18,
        include_extended_hours=True,
    )
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["status"] != "NO_DATA"
    assert res["fibo_levels"]["0.000"] == 100.00
    assert res["fibo_levels"]["1.000"] == 110.00


def test_t1_f02_multi_day_history_uses_second_to_last_trading_day(sample_config, mock_yfinance_factory):
    # D-2 has wild range 50..200; D-1..D has 100..112
    d_minus_2 = make_intraday_2m_df(prev_date="2026-11-13", curr_date="2026-11-16", trend="DOWNTREND", active_level=106.18)
    d_minus_2.iloc[:, d_minus_2.columns.get_loc("Low")] = 50.0
    d_minus_2.iloc[:, d_minus_2.columns.get_loc("High")] = 200.0
    d_minus_2 = d_minus_2[d_minus_2.index.date == date(2026, 11, 16)]

    d_recent = make_intraday_2m_df(prev_date="2026-11-17", curr_date="2026-11-18", trend="DOWNTREND", active_level=106.18)
    df_3days = pd.concat([d_minus_2, d_recent]).sort_index()

    mock_yfinance_factory({"AMD": {"history_2m": df_3days, "calendar": {}}})
    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["prev_trading_date"] == "2026-11-17"
    assert res["fibo_levels"]["0.000"] == 100.00
    assert res["fibo_levels"]["1.000"] == 110.00


# ===========================================================================
# F3: Entry Levels & Smart Routing (5 tests)
# ===========================================================================

def test_t1_f03_etf_spy_qqq_primary_0500(sample_config, mock_yfinance_factory):
    df_spy = make_intraday_2m_df(trend="DOWNTREND", active_level=105.00, rvol_target=1.30)
    df_qqq = make_intraday_2m_df(trend="UPTREND", active_level=106.50, rvol_target=1.30)
    mock_yfinance_factory({
        "SPY": {"history_2m": df_spy},
        "QQQ": {"history_2m": df_qqq},
    })

    for etf in ("SPY", "QQQ"):
        res = fibo_scanner.analyze_single_ticker(etf, sample_config)
        assert res["is_etf"] is True
        assert res["primary_fibo_level"] == "0.500"
        assert res["active_fibo_level"] == "0.500"


def test_t1_f03_golden_stock_primary_0618(sample_config, mock_yfinance_factory):
    df_amd = make_intraday_2m_df(trend="UPTREND", active_level=105.20, rvol_target=1.30)
    df_nvda = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, rvol_target=1.30)
    mock_yfinance_factory({
        "AMD": {"history_2m": df_amd, "calendar": {}},
        "NVDA": {"history_2m": df_nvda, "calendar": {}},
    })

    for tk in ("AMD", "NVDA"):
        res = fibo_scanner.analyze_single_ticker(tk, sample_config)
        assert res["is_etf"] is False
        assert res["primary_fibo_level"] == "0.618"
        assert res["active_fibo_level"] == "0.618"


def test_t1_f03_secondary_allowed_with_high_rvol_ge_150(sample_config, mock_yfinance_factory):
    # SPY primary is 0.500 (105.00), spot near secondary 0.618 (106.18) with RVOL 1.65 >= 1.50
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, rvol_target=1.65)
    mock_yfinance_factory({"SPY": {"history_2m": df}})

    res = fibo_scanner.analyze_single_ticker("SPY", sample_config)
    assert res["primary_fibo_level"] == "0.500"
    assert res["active_fibo_level"] == "0.618"
    assert res["status"] == "ACTIONABLE"


def test_t1_f03_tpsl_mapping_0618_and_0500(sample_config, mock_yfinance_factory):
    df_618 = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, rvol_target=1.30)
    df_500 = make_intraday_2m_df(trend="DOWNTREND", active_level=105.00, rvol_target=1.30)
    mock_yfinance_factory({
        "AMD": {"history_2m": df_618, "calendar": {}},
        "SPY": {"history_2m": df_500},
    })

    res_618 = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    f618 = res_618["fibo_levels"]
    assert res_618["trade_plan"]["tp1_underlying"] == f618["0.500"]
    assert res_618["trade_plan"]["tp2_underlying"] == f618["0.382"]
    assert res_618["trade_plan"]["sl_underlying"] == f618["0.786"]

    res_500 = fibo_scanner.analyze_single_ticker("SPY", sample_config)
    f500 = res_500["fibo_levels"]
    assert res_500["trade_plan"]["tp1_underlying"] == f500["0.382"]
    assert res_500["trade_plan"]["tp2_underlying"] == f500["0.236"]
    assert res_500["trade_plan"]["sl_underlying"] == f500["0.618"]


def test_t1_f03_watching_label_consistent_with_active_level(sample_config, mock_yfinance_factory):
    # Price far from both 0.500 and 0.618 -> WATCHING
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="NO_TOUCH", spot_override=102.00)
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["status"] == "WATCHING"
    assert res["active_fibo_level"] in res["status_thai"]
    assert f"${res['active_fibo_price']:.2f}" in res["status_thai"]


# ===========================================================================
# F4: Volume RVOL Filter (5 tests)
# ===========================================================================

def test_t1_f04_rvol_pass_ge_115_with_reversal_becomes_actionable(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="CONFIRMED_REVERSAL", rvol_target=1.30)
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["rvol"] == pytest.approx(1.30, abs=0.02)
    assert res["rvol_passed"] is True
    assert res["status"] == "ACTIONABLE"


def test_t1_f04_rvol_below_115_blocks_actionable(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="CONFIRMED_REVERSAL", rvol_target=0.95)
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["rvol_passed"] is False
    assert res["status"] != "ACTIONABLE"


def test_t1_f04_rvol_uses_max_of_last_two_bars(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="CONFIRMED_REVERSAL", rvol_target=1.00)
    # Set second-to-last bar Volume spike so prev_row RVOL ~= 1.40 while last_row ~= 1.00
    df.iloc[-2, df.columns.get_loc("Volume")] = 1430.0
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["rvol"] >= 1.35
    assert res["rvol_passed"] is True


def test_t1_f04_custom_vol_ma_period_from_config(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="CONFIRMED_REVERSAL", rvol_target=1.00)
    # Bars -20..-11 high volume (3000), bars -10..-2 low volume (1000), last bar 2000
    vol_col = df.columns.get_loc("Volume")
    df.iloc[-20:-10, vol_col] = 3000.0
    df.iloc[-10:-1, vol_col] = 1000.0
    df.iloc[-1, vol_col] = 2000.0
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    sample_config["strategy_rules"]["volume_filter"]["vol_ma_period"] = 10
    res_10 = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    # With N=10: mean = (9*1000 + 2000)/10 = 1100 -> RVOL = 2000/1100 = 1.82
    assert res_10["rvol"] >= 1.75
    assert res_10["rvol_passed"] is True


def test_t1_f04_volume_filter_disabled_in_config(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="CONFIRMED_REVERSAL", rvol_target=0.60)
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    sample_config["strategy_rules"]["volume_filter"]["enabled"] = False
    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["rvol_passed"] is True
    assert res["status"] == "ACTIONABLE"


# ===========================================================================
# F5: Reversal Candlestick Gate (5 tests)
# ===========================================================================

def test_t1_f05_downtrend_shooting_star_at_fibo_confirmed():
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="CONFIRMED_REVERSAL")
    curr_df = df.iloc[-25:]
    res = fibo_scanner.detect_reversal_candle(curr_df, "DOWNTREND", 106.18, 10.0)
    assert res["touched"] is True
    assert res["confirmed"] is True
    assert "Shooting Star" in res["pattern_name"]


def test_t1_f05_uptrend_hammer_at_fibo_confirmed():
    df = make_intraday_2m_df(trend="UPTREND", active_level=105.20, candle_mode="CONFIRMED_REVERSAL")
    curr_df = df.iloc[-25:]
    res = fibo_scanner.detect_reversal_candle(curr_df, "UPTREND", 105.20, 11.0)
    assert res["touched"] is True
    assert res["confirmed"] is True
    assert "Hammer" in res["pattern_name"]


def test_t1_f05_downtrend_bearish_engulfing_confirmed():
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=105.00, candle_mode="ENGULFING")
    curr_df = df.iloc[-25:]
    res = fibo_scanner.detect_reversal_candle(curr_df, "DOWNTREND", 105.00, 10.0)
    assert res["touched"] is True
    assert res["confirmed"] is True
    assert "Bearish" in res["pattern_name"]


def test_t1_f05_uptrend_bullish_engulfing_confirmed():
    df = make_intraday_2m_df(trend="UPTREND", active_level=106.50, candle_mode="ENGULFING")
    curr_df = df.iloc[-25:]
    res = fibo_scanner.detect_reversal_candle(curr_df, "UPTREND", 106.50, 11.0)
    assert res["touched"] is True
    assert res["confirmed"] is True
    assert "Bullish" in res["pattern_name"]


def test_t1_f05_touched_without_reversal_is_near_setup(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="TOUCHED_ONLY", rvol_target=1.30)
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["candle_reversal"]["touched"] is True
    assert res["candle_reversal"]["confirmed"] is False
    assert res["status"] == "NEAR_SETUP"


# ===========================================================================
# F6: Morning Gap Filter (5 tests)
# ===========================================================================

def test_t1_f06_gap_pct_computed_accurately(sample_config, mock_yfinance_factory):
    # prev_close = 105.00, 3.5% gap -> first_open = 108.675
    df = make_intraday_2m_df(
        trend="DOWNTREND",
        active_level=106.18,
        candle_mode="NO_TOUCH",
        morning_gap_pct=3.5,
        spot_override=102.00,
    )
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["morning_gap_pct"] == pytest.approx(3.5, abs=0.05)


def test_t1_f06_large_gap_away_from_fibo_triggers_gap_skip(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(
        trend="DOWNTREND",
        active_level=106.18,
        candle_mode="NO_TOUCH",
        morning_gap_pct=3.0,
        spot_override=101.00,
    )
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["morning_gap_pct"] > 2.0
    assert res["distance_to_active_fibo_pct"] > 1.5
    assert res["status"] == "GAP_SKIP"


def test_t1_f06_small_gap_under_threshold_not_skipped(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(
        trend="DOWNTREND",
        active_level=106.18,
        candle_mode="NO_TOUCH",
        morning_gap_pct=1.2,
        spot_override=101.00,
    )
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["status"] != "GAP_SKIP"


def test_t1_f06_large_gap_but_near_fibo_not_skipped(sample_config, mock_yfinance_factory):
    # Gap 3.0% at open, then pulls back onto 0.618 (dist <= 1.5%)
    df = make_intraday_2m_df(
        trend="DOWNTREND",
        active_level=106.18,
        candle_mode="CONFIRMED_REVERSAL",
        morning_gap_pct=3.0,
        rvol_target=1.30,
    )
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["morning_gap_pct"] > 2.0
    assert res["distance_to_active_fibo_pct"] <= 1.5
    assert res["status"] != "GAP_SKIP"


def test_t1_f06_gap_filter_disabled_in_config(sample_config, mock_yfinance_factory):
    df = make_intraday_2m_df(
        trend="DOWNTREND",
        active_level=106.18,
        candle_mode="NO_TOUCH",
        morning_gap_pct=4.0,
        spot_override=101.00,
    )
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}}})

    sample_config["strategy_rules"]["gap_filter"]["enabled"] = False
    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["status"] != "GAP_SKIP"


# ===========================================================================
# F7: Earnings Blackout Filter (5 tests)
# ===========================================================================

def test_t1_f07_blocks_on_earnings_day_and_within_window(sample_config, mock_yfinance_factory, freeze_ny_time):
    ref = date(2026, 11, 18)
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, rvol_target=1.30)

    for offset in (-1, 0, 2):
        earn_dt = date(2026, 11, 18 + offset)
        mock_yfinance_factory({
            "NVDA": {
                "history_2m": df,
                "calendar": {"Earnings Date": [datetime(earn_dt.year, earn_dt.month, earn_dt.day)]},
            }
        })
        chk = call_check_earnings("NVDA", reference_date=ref, days_before=2, days_after=1)
        assert chk["is_blackout"] is True

    # Full scanner check on frozen NY ref date (2026-11-18)
    freeze_ny_time(datetime(2026, 11, 18, 10, 0))
    mock_yfinance_factory({
        "NVDA": {
            "history_2m": df,
            "calendar": {"Earnings Date": [datetime(2026, 11, 18)]},
        }
    })
    res = fibo_scanner.analyze_single_ticker("NVDA", sample_config)
    assert res["earnings"]["is_blackout"] is True
    assert res["status"] == "EARNINGS_BLACKOUT"
    assert res["score"] == 0


def test_t1_f07_safe_outside_blackout_window(mock_yfinance_factory):
    ref = date(2026, 11, 18)
    for earn_dt in (date(2026, 11, 21), date(2026, 11, 16)):
        mock_yfinance_factory({
            "NVDA": {"calendar": {"Earnings Date": [datetime(earn_dt.year, earn_dt.month, earn_dt.day)]}}
        })
        chk = call_check_earnings("NVDA", reference_date=ref, days_before=2, days_after=1)
        assert chk["is_blackout"] is False


def test_t1_f07_etf_safe_even_on_404(mock_yfinance_factory):
    mock_yfinance_factory({
        "SPY": {"calendar": RuntimeError("HTTP 404 Not Found")},
        "QQQ": {"calendar": RuntimeError("HTTP 404 Not Found")},
    })
    for etf in ("SPY", "QQQ"):
        chk = call_check_earnings(etf, reference_date=date(2026, 11, 18))
        assert chk["is_blackout"] is False


def test_t1_f07_wires_config_days_and_enabled_flag(sample_config, mock_yfinance_factory, freeze_ny_time):
    freeze_ny_time(datetime(2026, 11, 18, 10, 0))
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, rvol_target=1.30)
    # Earnings 4 days ahead (2026-11-22): safe under default days_before=2, blocked under days_before=5
    mock_yfinance_factory({
        "NVDA": {
            "history_2m": df,
            "calendar": {"Earnings Date": [datetime(2026, 11, 22)]},
        }
    })

    sample_config["strategy_rules"]["earnings_filter"] = {
        "enabled": True,
        "blackout_days_before": 5,
        "blackout_days_after": 3,
    }
    res_blocked = fibo_scanner.analyze_single_ticker("NVDA", sample_config)
    assert res_blocked["earnings"]["is_blackout"] is True
    assert res_blocked["status"] == "EARNINGS_BLACKOUT"

    sample_config["strategy_rules"]["earnings_filter"]["enabled"] = False
    res_disabled = fibo_scanner.analyze_single_ticker("NVDA", sample_config)
    assert res_disabled["earnings"]["is_blackout"] is False
    assert res_disabled["status"] != "EARNINGS_BLACKOUT"


def test_t1_f07_past_earnings_label_accurate(mock_yfinance_factory):
    ref = date(2026, 11, 18)
    mock_yfinance_factory({
        "AMD": {"calendar": {"Earnings Date": [datetime(2026, 11, 2)]}}
    })
    chk = call_check_earnings("AMD", reference_date=ref, days_before=2, days_after=1)
    assert chk["is_blackout"] is False
    assert chk["days_until_earnings"] == -16
    assert "งบถัดไป" not in chk["status_label"]
    assert "อีก -16" not in chk["status_label"]
