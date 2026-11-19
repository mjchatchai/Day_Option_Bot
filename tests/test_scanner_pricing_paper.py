"""
Deterministic, zero-network unit & integration tests for Milestone M1 (F1-F10):
- F1: TF 2m EMA 200 trend filter (>=200 warmup, ==/NaN -> NEUTRAL/NO_DATA, no 5m fallback)
- F2: Previous-day asymmetric Fibo (NY regular hours 09:30-15:59, min-bars & min-range guards)
- F3: Entry levels & smart Fibo routing (0.500 ETFs / 0.618 stocks, secondary_min_rvol, WATCHING label)
- F4: Volume RVOL filter (config wiring, NaN safety)
- F5: Reversal candlestick gate (touched == True required before reversal pattern name)
- F6: Morning gap filter (enabled/disabled config wiring)
- F7: Earnings blackout filter (NY date, config wiring, dict/DataFrame/str parsing, past label fix, ETF safety)
- F8: Real-time BSM option pricing & contract selection (priced at spot_realtime, NY DTE, liquidity, NaN safety)
- F9: time_utils NY TZ/DST/holidays/2h window + yfinance edge cases & retry
- F10: Paper trader 2h gate, EOD before TP/SL, TP1 lock_sl anchor, file lock/backup, session-end flush
"""

import copy
from datetime import date, datetime, timedelta, timezone
import json
import math
import os
import sys
import threading
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from day_option_bot.src import earnings_filter, fibo_scanner, options_pricing, paper_trader, time_utils


CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config", "config.json")
)


def load_base_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def make_2m_bars(
    prev_date: date = date(2026, 11, 17),
    curr_date: date = date(2026, 11, 18),
    n_prev: int = 195,
    n_curr: int = 30,
    prev_base: float = 100.0,
    curr_base: float = 105.0,
    prev_wick_low: float = 98.0,
    prev_wick_high: float = 112.0,
    prev_body_low: float = 99.0,
    prev_body_high: float = 110.0,
    curr_last_open: float = 106.0,
    curr_last_high: float = 107.0,
    curr_last_low: float = 104.5,
    curr_last_close: float = 105.0,
    curr_prev_open: float = 105.5,
    curr_prev_high: float = 106.8,
    curr_prev_low: float = 105.2,
    curr_prev_close: float = 106.2,
    base_vol: float = 1000.0,
    last_vol: float = 1500.0,
    prev_vol: float = 1000.0,
    gap_open_first: float = None,
) -> pd.DataFrame:
    """Build deterministic NY-timezone 2m OHLCV bars spanning prev_date and curr_date."""
    ny = time_utils.NY_TZ
    rows = []
    idx = []

    t_prev_start = datetime(prev_date.year, prev_date.month, prev_date.day, 9, 30, tzinfo=ny)
    for i in range(n_prev):
        ts = t_prev_start + timedelta(minutes=2 * i)
        idx.append(ts)
        # Keep baseline bars smoothly around prev_base, and inject explicit anchor bars
        if i == 5:
            # Lowest bar (wick low & body low)
            o = prev_body_low + 0.5
            c = prev_body_low
            h = o + 0.5
            l = prev_wick_low
        elif i == 15:
            # Highest bar (wick high & body high)
            o = prev_body_high - 0.5
            c = prev_body_high
            h = prev_wick_high
            l = o - 0.5
        else:
            o = prev_base
            c = prev_base + (0.1 if i % 2 == 0 else -0.1)
            h = max(o, c) + 0.2
            l = min(o, c) - 0.2
        rows.append({"Open": o, "High": h, "Low": l, "Close": c, "Volume": base_vol})

    t_curr_start = datetime(curr_date.year, curr_date.month, curr_date.day, 9, 30, tzinfo=ny)
    for i in range(n_curr):
        ts = t_curr_start + timedelta(minutes=2 * i)
        idx.append(ts)
        if i == 0 and gap_open_first is not None:
            o = gap_open_first
            c = curr_base
            h = max(o, c) + 0.2
            l = min(o, c) - 0.2
            v = base_vol
        elif i == n_curr - 2 and n_curr >= 2:
            o, h, l, c, v = curr_prev_open, curr_prev_high, curr_prev_low, curr_prev_close, prev_vol
        elif i == n_curr - 1:
            o, h, l, c, v = curr_last_open, curr_last_high, curr_last_low, curr_last_close, last_vol
        else:
            o = curr_base
            c = curr_base + (0.05 if i % 2 == 0 else -0.05)
            h = max(o, c) + 0.15
            l = min(o, c) - 0.15
            v = base_vol
        rows.append({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v})

    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def _patch_clock(monkeypatch, dt_ny: datetime):
    monkeypatch.setattr(time_utils, "now_ny", lambda: dt_ny)
    monkeypatch.setattr(time_utils, "today_ny", lambda: dt_ny.date())
    monkeypatch.setattr(fibo_scanner, "today_ny", lambda: dt_ny.date())
    monkeypatch.setattr(earnings_filter, "today_ny", lambda: dt_ny.date())
    monkeypatch.setattr(options_pricing, "today_ny", lambda: dt_ny.date())
    monkeypatch.setattr(paper_trader, "now_ny", lambda: dt_ny)
    monkeypatch.setattr(paper_trader, "today_ny", lambda: dt_ny.date())


def _patch_safe_earnings_and_option(monkeypatch):
    monkeypatch.setattr(
        fibo_scanner,
        "check_earnings_blackout",
        lambda ticker, **kw: {
            "ticker": ticker,
            "is_blackout": False,
            "earnings_date": None,
            "days_until_earnings": None,
            "status_label": "SAFE",
        },
    )
    monkeypatch.setattr(
        fibo_scanner,
        "select_and_price_realtime_option",
        lambda **kw: options_pricing.RealTimeOptionQuote(
            ticker=kw["ticker"],
            option_type="PUT" if "PUT" in kw["direction"] else "CALL",
            expiration="2026-11-20",
            dte=2,
            strike=round(kw["spot_realtime"], 2),
            spot_realtime=round(kw["spot_realtime"], 2),
            iv=0.35,
            realtime_theo_price=2.50,
            est_bid=2.45,
            est_ask=2.55,
            max_limit_buy=2.57,
            delta=-0.55 if "PUT" in kw["direction"] else 0.55,
            abs_delta=0.55,
            gamma=0.05,
            theta_daily=-0.12,
            tp1_option_price=3.40,
            tp1_gain_pct=36.0,
            tp2_option_price=4.30,
            tp2_gain_pct=72.0,
            sl_option_price=1.60,
            sl_loss_pct=-36.0,
            recommended_contracts=1,
            spread_pct=2.0,
            fibo_entry_theo_price=2.40,
        ),
    )


# ---------------------------------------------------------------------------
# T1 (F1): TF 2m EMA 200 Trend Filter
# ---------------------------------------------------------------------------
def test_ema200_trend_call_put_neutral_nan_warmup(monkeypatch):
    cfg = load_base_config()
    _patch_clock(monkeypatch, datetime(2026, 11, 18, 10, 15, tzinfo=time_utils.NY_TZ))
    _patch_safe_earnings_and_option(monkeypatch)

    # (a) spot > ema200 -> UPTREND / BUY CALL
    df_up = make_2m_bars(prev_base=100.0, curr_base=115.0, curr_last_open=115.0, curr_last_high=116.0, curr_last_low=114.5, curr_last_close=115.8)
    monkeypatch.setattr(fibo_scanner, "fetch_intraday_2m", lambda tk, **kw: df_up)
    res_up = fibo_scanner.analyze_single_ticker("NVDA", cfg)
    assert res_up["trend_direction"] == "UPTREND"
    assert res_up["option_action"] == "BUY CALL"
    assert res_up["is_current_session"] is True

    # (b) spot < ema200 -> DOWNTREND / BUY PUT
    df_down = make_2m_bars(prev_base=110.0, curr_base=95.0, curr_last_open=95.5, curr_last_high=96.0, curr_last_low=94.5, curr_last_close=94.8)
    monkeypatch.setattr(fibo_scanner, "fetch_intraday_2m", lambda tk, **kw: df_down)
    res_down = fibo_scanner.analyze_single_ticker("NVDA", cfg)
    assert res_down["trend_direction"] == "DOWNTREND"
    assert res_down["option_action"] == "BUY PUT"

    # (c) spot == ema200 -> NEUTRAL (never BUY CALL)
    df_flat = make_2m_bars(
        prev_base=100.0,
        curr_base=100.0,
        prev_wick_low=99.0,
        prev_body_low=99.5,
        prev_body_high=100.5,
        prev_wick_high=101.0,
        curr_last_open=100.0,
        curr_last_high=100.2,
        curr_last_low=99.8,
        curr_last_close=100.0,
    )
    # Force exact equality between rounded Close and EMA200 on final row
    orig_compute = fibo_scanner.compute_intraday_indicators

    def _compute_equal_ema(d, **kw):
        out = orig_compute(d, **kw)
        out.iloc[-1, out.columns.get_loc("EMA200")] = float(out.iloc[-1]["Close"])
        return out

    monkeypatch.setattr(fibo_scanner, "fetch_intraday_2m", lambda tk, **kw: df_flat)
    monkeypatch.setattr(fibo_scanner, "compute_intraday_indicators", _compute_equal_ema)
    res_neutral = fibo_scanner.analyze_single_ticker("NVDA", cfg)
    assert res_neutral["status"] == "NEUTRAL"
    assert res_neutral["option_action"] == "NEUTRAL"
    assert res_neutral["trend_direction"] == "NEUTRAL"

    # (d) NaN EMA200 -> NO_DATA
    def _compute_nan_ema(d, **kw):
        out = orig_compute(d, **kw)
        out.iloc[-1, out.columns.get_loc("EMA200")] = np.nan
        return out

    monkeypatch.setattr(fibo_scanner, "compute_intraday_indicators", _compute_nan_ema)
    res_nan = fibo_scanner.analyze_single_ticker("NVDA", cfg)
    assert res_nan["status"] == "NO_DATA"
    monkeypatch.setattr(fibo_scanner, "compute_intraday_indicators", orig_compute)

    # (e) < 200 bars -> NO_DATA
    df_short = make_2m_bars(n_prev=150, n_curr=30)
    monkeypatch.setattr(fibo_scanner, "fetch_intraday_2m", lambda tk, **kw: df_short)
    res_short = fibo_scanner.analyze_single_ticker("NVDA", cfg)
    assert res_short["status"] == "NO_DATA"

    # (f) empty 2m history does NOT fall back to 5m
    calls = []

    class _EmptyTicker:
        def __init__(self, tk):
            self.tk = tk

        def history(self, **kwargs):
            calls.append(kwargs)
            return pd.DataFrame()

    monkeypatch.undo()
    monkeypatch.setattr(fibo_scanner.yf, "Ticker", _EmptyTicker)
    df_fetched = fibo_scanner.fetch_intraday_2m("NVDA")
    assert df_fetched.empty
    assert len(calls) == 1
    assert calls[0]["interval"] == "2m"
    assert calls[0]["prepost"] is False


# ---------------------------------------------------------------------------
# T2 (F2): Previous-Day Asymmetric Fibo & Regular-Hours / Min-Bars / Min-Range Guards
# ---------------------------------------------------------------------------
def test_prev_day_fibo_regular_hours_and_guards(monkeypatch):
    cfg = load_base_config()
    _patch_clock(monkeypatch, datetime(2026, 11, 18, 10, 15, tzinfo=time_utils.NY_TZ))
    _patch_safe_earnings_and_option(monkeypatch)

    ny = time_utils.NY_TZ
    df_reg = make_2m_bars(
        prev_base=105.0,
        curr_base=101.0,
        prev_wick_low=100.0,
        prev_body_low=101.0,
        prev_body_high=110.0,
        prev_wick_high=112.0,
        curr_last_open=101.0,
        curr_last_high=101.5,
        curr_last_low=100.5,
        curr_last_close=100.8,
    )
    # Inject extreme pre-market (08:00 ET) and after-hours (17:00 ET) bars on prev_date
    ext_rows = pd.DataFrame(
        [
            {"Open": 50.0, "High": 55.0, "Low": 40.0, "Close": 50.0, "Volume": 9999.0},
            {"Open": 200.0, "High": 220.0, "Low": 190.0, "Close": 210.0, "Volume": 9999.0},
        ],
        index=pd.DatetimeIndex(
            [
                datetime(2026, 11, 17, 8, 0, tzinfo=ny),
                datetime(2026, 11, 17, 17, 0, tzinfo=ny),
            ]
        ),
    )
    df_with_ext = pd.concat([ext_rows, df_reg])
    monkeypatch.setattr(fibo_scanner, "fetch_intraday_2m", lambda tk, **kw: df_with_ext)
    res = fibo_scanner.analyze_single_ticker("NVDA", cfg)
    assert res["trend_direction"] == "DOWNTREND"
    # Anchors must ignore 40.0 / 220.0 extended hours and match regular hours 100.0 / 110.0
    assert res["fibo_levels"]["0.000"] == 100.0
    assert res["fibo_levels"]["1.000"] == 110.0
    assert res["fibo_levels"]["0.618"] == 106.18

    # (c) < min_prev_day_bars -> NO_DATA
    df_few_prev = make_2m_bars(n_prev=8, n_curr=195)
    monkeypatch.setattr(fibo_scanner, "fetch_intraday_2m", lambda tk, **kw: df_few_prev)
    res_few = fibo_scanner.analyze_single_ticker("NVDA", cfg)
    assert res_few["status"] == "NO_DATA"

    # (d) flat/tiny range day -> NO_DATA
    df_flat_range = make_2m_bars(
        prev_base=100.0,
        curr_base=100.5,
        prev_wick_low=99.98,
        prev_body_low=99.99,
        prev_body_high=100.01,
        prev_wick_high=100.02,
        curr_last_open=100.5,
        curr_last_high=100.6,
        curr_last_low=100.4,
        curr_last_close=100.55,
    )
    prev_mask = df_flat_range.index.date == date(2026, 11, 17)
    df_flat_range.loc[prev_mask, ["Open", "High", "Low", "Close"]] = [100.0, 100.01, 99.99, 100.0]
    monkeypatch.setattr(fibo_scanner, "fetch_intraday_2m", lambda tk, **kw: df_flat_range)
    res_flat_rng = fibo_scanner.analyze_single_ticker("NVDA", cfg)
    assert res_flat_rng["status"] == "NO_DATA"


# ---------------------------------------------------------------------------
# T3 (F3): Smart Fibo Routing & WATCHING Label Consistency
# ---------------------------------------------------------------------------
def test_smart_fibo_routing_and_watching_label(monkeypatch):
    cfg = load_base_config()
    _patch_clock(monkeypatch, datetime(2026, 11, 18, 10, 15, tzinfo=time_utils.NY_TZ))
    _patch_safe_earnings_and_option(monkeypatch)

    # Downtrend: 0.000=100, 1.000=110 -> 0.500=105.0, 0.618=106.18
    # Spot touches 0.618 (106.18) with bearish engulfing, RVOL = 1.48 (< 1.5)
    df_etf_low_rvol = make_2m_bars(
        prev_base=110.0,
        curr_base=106.0,
        prev_wick_low=100.0,
        prev_body_low=101.0,
        prev_body_high=110.0,
        prev_wick_high=112.0,
        curr_prev_open=105.8,
        curr_prev_high=106.25,
        curr_prev_low=105.7,
        curr_prev_close=106.15,
        curr_last_open=106.20,
        curr_last_high=106.30,
        curr_last_low=105.60,
        curr_last_close=105.65,
        base_vol=1000.0,
        prev_vol=1000.0,
        last_vol=1480.0,  # RVOL ~ 1.45 (< 1.5)
    )
    monkeypatch.setattr(fibo_scanner, "fetch_intraday_2m", lambda tk, **kw: df_etf_low_rvol)

    # (a) & (c) ETF QQQ stays on primary 0.500 when RVOL < 1.5
    res_qqq_low = fibo_scanner.analyze_single_ticker("QQQ", cfg)
    assert res_qqq_low["primary_fibo_level"] == "0.500"
    assert res_qqq_low["active_fibo_level"] == "0.500"
    assert res_qqq_low["status"] != "ACTIONABLE"

    # (b) Stock NVDA defaults to 0.618 even with RVOL < 1.5
    res_nvda = fibo_scanner.analyze_single_ticker("NVDA", cfg)
    assert res_nvda["primary_fibo_level"] == "0.618"
    assert res_nvda["active_fibo_level"] == "0.618"
    assert res_nvda["status"] == "ACTIONABLE"

    # (d) ETF QQQ switches to 0.618 when RVOL >= 1.5 and allow_secondary_with_high_rvol=True
    df_etf_high_rvol = make_2m_bars(
        prev_base=110.0,
        curr_base=106.0,
        prev_wick_low=100.0,
        prev_body_low=101.0,
        prev_body_high=110.0,
        prev_wick_high=112.0,
        curr_prev_open=105.8,
        curr_prev_high=106.25,
        curr_prev_low=105.7,
        curr_prev_close=106.15,
        curr_last_open=106.20,
        curr_last_high=106.30,
        curr_last_low=105.60,
        curr_last_close=105.65,
        base_vol=1000.0,
        prev_vol=1000.0,
        last_vol=1800.0,  # RVOL >= 1.5
    )
    monkeypatch.setattr(fibo_scanner, "fetch_intraday_2m", lambda tk, **kw: df_etf_high_rvol)
    res_qqq_high = fibo_scanner.analyze_single_ticker("QQQ", cfg)
    assert res_qqq_high["active_fibo_level"] == "0.618"
    assert res_qqq_high["status"] == "ACTIONABLE"

    # (e) allow_secondary_with_high_rvol=False blocks switching even at RVOL=2.0
    cfg_no_sec = copy.deepcopy(cfg)
    cfg_no_sec["strategy_rules"]["smart_fibo_routing"]["allow_secondary_with_high_rvol"] = False
    res_qqq_blocked = fibo_scanner.analyze_single_ticker("QQQ", cfg_no_sec)
    assert res_qqq_blocked["active_fibo_level"] == "0.500"

    # (f) WATCHING status_thai matches active_fibo_level and active_fibo_price
    df_watch = make_2m_bars(
        prev_base=110.0,
        curr_base=106.0,
        prev_wick_low=100.0,
        prev_body_low=101.0,
        prev_body_high=110.0,
        prev_wick_high=112.0,
        gap_open_first=110.0,
        curr_prev_open=106.85,
        curr_prev_high=106.90,
        curr_prev_low=106.70,
        curr_prev_close=106.80,
        curr_last_open=106.80,
        curr_last_high=106.85,
        curr_last_low=106.68,
        curr_last_close=106.75,  # ~0.53% above 106.18 (switches to 0.618, >0.45% so WATCHING)
        base_vol=1000.0,
        prev_vol=1000.0,
        last_vol=1800.0,
    )
    monkeypatch.setattr(fibo_scanner, "fetch_intraday_2m", lambda tk, **kw: df_watch)
    res_watch = fibo_scanner.analyze_single_ticker("QQQ", cfg)
    assert res_watch["status"] == "WATCHING"
    assert res_watch["active_fibo_level"] == "0.618"
    assert f"Fibo {res_watch['active_fibo_level']}" in res_watch["status_thai"]
    assert f"${res_watch['active_fibo_price']:.2f}" in res_watch["status_thai"]


# ---------------------------------------------------------------------------
# T4 (F4): Volume RVOL Filter Config & NaN Safety
# ---------------------------------------------------------------------------
def test_rvol_filter_config_and_nan_safety(monkeypatch):
    cfg = load_base_config()
    _patch_clock(monkeypatch, datetime(2026, 11, 18, 10, 15, tzinfo=time_utils.NY_TZ))
    _patch_safe_earnings_and_option(monkeypatch)

    # Low RVOL bar at Fibo 0.618 (106.18) with confirmed bearish engulfing
    df_low_vol = make_2m_bars(
        prev_base=110.0,
        curr_base=106.0,
        prev_wick_low=100.0,
        prev_body_low=101.0,
        prev_body_high=110.0,
        prev_wick_high=112.0,
        curr_prev_open=105.8,
        curr_prev_high=106.25,
        curr_prev_low=105.7,
        curr_prev_close=106.15,
        curr_last_open=106.20,
        curr_last_high=106.30,
        curr_last_low=105.60,
        curr_last_close=105.65,
        base_vol=1000.0,
        prev_vol=500.0,
        last_vol=600.0,  # RVOL < 1.15
    )
    monkeypatch.setattr(fibo_scanner, "fetch_intraday_2m", lambda tk, **kw: df_low_vol)
    res_fail = fibo_scanner.analyze_single_ticker("NVDA", cfg)
    assert res_fail["rvol_passed"] is False
    assert res_fail["status"] != "ACTIONABLE"

    # (c) volume_filter.enabled = False allows low RVOL to pass
    cfg_disabled = copy.deepcopy(cfg)
    cfg_disabled["strategy_rules"]["volume_filter"]["enabled"] = False
    res_disabled = fibo_scanner.analyze_single_ticker("NVDA", cfg_disabled)
    assert res_disabled["rvol_passed"] is True
    assert res_disabled["status"] == "ACTIONABLE"

    # (b) custom vol_ma_period changes RVOL
    ind_5 = fibo_scanner.compute_intraday_indicators(df_low_vol, ema_period=200, vol_ma_period=5)
    ind_20 = fibo_scanner.compute_intraday_indicators(df_low_vol, ema_period=200, vol_ma_period=20)
    assert float(ind_5["RVOL"].iloc[-1]) != float(ind_20["RVOL"].iloc[-1])

    # (d) NaN in prev_row["RVOL"] does not poison curr_rvol
    orig_compute = fibo_scanner.compute_intraday_indicators

    def _compute_nan_prev_rvol(d, **kw):
        out = orig_compute(d, **kw)
        out.iloc[-2, out.columns.get_loc("RVOL")] = np.nan
        return out

    monkeypatch.setattr(fibo_scanner, "compute_intraday_indicators", _compute_nan_prev_rvol)
    res_nan_rvol = fibo_scanner.analyze_single_ticker("NVDA", cfg)
    assert math.isfinite(res_nan_rvol["rvol"])
    assert not math.isnan(res_nan_rvol["rvol"])


# ---------------------------------------------------------------------------
# T5 (F5): Reversal Candlestick Gate (touched == True required)
# ---------------------------------------------------------------------------
def test_reversal_candle_requires_touch_gate(monkeypatch):
    ny = time_utils.NY_TZ
    idx = pd.DatetimeIndex(
        [
            datetime(2026, 11, 18, 10, 0, tzinfo=ny),
            datetime(2026, 11, 18, 10, 2, tzinfo=ny),
        ]
    )

    # (a) Shooting star far away from level 106.18 (touched=False)
    df_no_touch_down = pd.DataFrame(
        [
            {"Open": 100.0, "High": 100.2, "Low": 99.8, "Close": 100.1},
            {"Open": 100.1, "High": 101.0, "Low": 99.9, "Close": 100.0},  # big upper wick
        ],
        index=idx,
    )
    chk_down = fibo_scanner.detect_reversal_candle(df_no_touch_down, "DOWNTREND", 106.18, 10.0)
    assert chk_down["touched"] is False
    assert chk_down["confirmed"] is False
    assert "Shooting Star" not in chk_down["pattern_name"]
    assert "Reversal Candle" not in chk_down["pattern_name"]

    # Hammer far away from level 106.18 (touched=False)
    df_no_touch_up = pd.DataFrame(
        [
            {"Open": 110.0, "High": 110.2, "Low": 109.8, "Close": 109.9},
            {"Open": 109.9, "High": 110.0, "Low": 109.0, "Close": 109.95},  # big lower wick
        ],
        index=idx,
    )
    chk_up = fibo_scanner.detect_reversal_candle(df_no_touch_up, "UPTREND", 106.18, 10.0)
    assert chk_up["touched"] is False
    assert chk_up["confirmed"] is False
    assert "Hammer" not in chk_up["pattern_name"]
    assert "Reversal Candle" not in chk_up["pattern_name"]

    # (b) touched=True + bearish shooting star -> confirmed=True
    df_touch_down = pd.DataFrame(
        [
            {"Open": 105.9, "High": 106.2, "Low": 105.8, "Close": 106.1},
            {"Open": 106.0, "High": 106.9, "Low": 105.85, "Close": 105.9},
        ],
        index=idx,
    )
    chk_td = fibo_scanner.detect_reversal_candle(df_touch_down, "DOWNTREND", 106.18, 10.0)
    assert chk_td["touched"] is True
    assert chk_td["confirmed"] is True

    # (c) touched=True + bullish hammer -> confirmed=True
    df_touch_up = pd.DataFrame(
        [
            {"Open": 106.4, "High": 106.5, "Low": 106.1, "Close": 106.2},
            {"Open": 106.2, "High": 106.35, "Low": 105.4, "Close": 106.3},
        ],
        index=idx,
    )
    chk_tu = fibo_scanner.detect_reversal_candle(df_touch_up, "UPTREND", 106.18, 10.0)
    assert chk_tu["touched"] is True
    assert chk_tu["confirmed"] is True

    # (d) touched=True + neutral/indecisive bar -> confirmed=False, NEAR_SETUP in scanner
    cfg = load_base_config()
    _patch_clock(monkeypatch, datetime(2026, 11, 18, 10, 15, tzinfo=ny))
    _patch_safe_earnings_and_option(monkeypatch)
    df_near = make_2m_bars(
        prev_base=110.0,
        curr_base=106.0,
        prev_wick_low=100.0,
        prev_body_low=101.0,
        prev_body_high=110.0,
        prev_wick_high=112.0,
        curr_prev_open=106.10,
        curr_prev_high=106.22,
        curr_prev_low=106.05,
        curr_prev_close=106.18,
        curr_last_open=106.15,
        curr_last_high=106.26,
        curr_last_low=106.12,
        curr_last_close=106.25,  # bullish close at resistance -> touched=True, confirmed=False
    )
    monkeypatch.setattr(fibo_scanner, "fetch_intraday_2m", lambda tk, **kw: df_near)
    res_near = fibo_scanner.analyze_single_ticker("NVDA", cfg)
    assert res_near["candle_reversal"]["touched"] is True
    assert res_near["candle_reversal"]["confirmed"] is False
    assert res_near["status"] == "NEAR_SETUP"


# ---------------------------------------------------------------------------
# T6 (F6): Morning Gap Filter (Enabled vs Disabled)
# ---------------------------------------------------------------------------
def test_morning_gap_filter_enabled_and_disabled(monkeypatch):
    cfg = load_base_config()
    _patch_clock(monkeypatch, datetime(2026, 11, 18, 10, 15, tzinfo=time_utils.NY_TZ))
    _patch_safe_earnings_and_option(monkeypatch)

    # Prev close ~ 110.0, gap_open_first = 113.5 (~3.18% > 2.0%), spot=114.0 (>1.5% from 0.618=106.18)
    df_gap = make_2m_bars(
        prev_base=110.0,
        curr_base=114.0,
        prev_wick_low=100.0,
        prev_body_low=101.0,
        prev_body_high=110.0,
        prev_wick_high=112.0,
        gap_open_first=113.5,
        curr_last_open=113.8,
        curr_last_high=114.2,
        curr_last_low=113.6,
        curr_last_close=114.0,
    )
    monkeypatch.setattr(fibo_scanner, "fetch_intraday_2m", lambda tk, **kw: df_gap)

    res_skip = fibo_scanner.analyze_single_ticker("NVDA", cfg)
    assert res_skip["morning_gap_pct"] > 2.0
    assert res_skip["status"] == "GAP_SKIP"

    cfg_no_gap = copy.deepcopy(cfg)
    cfg_no_gap["strategy_rules"]["gap_filter"]["enabled"] = False
    res_allow = fibo_scanner.analyze_single_ticker("NVDA", cfg_no_gap)
    assert res_allow["status"] != "GAP_SKIP"


# ---------------------------------------------------------------------------
# T7 (F7): Earnings Blackout Filter (NY Date, Formats, Past Label, ETF Safety)
# ---------------------------------------------------------------------------
def test_earnings_blackout_ny_date_formats_and_config(monkeypatch):
    # (a) 2026-11-18 19:00 EST (already 2026-11-19 in UTC/TH) -> NY date is 2026-11-18
    dt_ny_evening = datetime(2026, 11, 18, 19, 0, tzinfo=time_utils.NY_TZ)
    monkeypatch.setattr(earnings_filter, "today_ny", lambda: dt_ny_evening.date())

    class _MockTickerDict:
        def __init__(self, tk):
            self.fast_info = {"quoteType": "EQUITY"}
            self.calendar = {
                "Earnings High": 3.13,  # EPS float must be ignored
                "Earnings Date": [date(2026, 11, 20)],
            }

    monkeypatch.setattr(earnings_filter.yf, "Ticker", _MockTickerDict)
    res_ny = earnings_filter.check_earnings_blackout("NVDA", days_before=2, days_after=1)
    assert res_ny["days_until_earnings"] == 2
    assert res_ny["is_blackout"] is True

    # (b) enabled=False & custom blackout_days_before/after
    res_off = earnings_filter.check_earnings_blackout("NVDA", enabled=False)
    assert res_off["is_blackout"] is False
    assert "Disabled" in res_off["status_label"]

    res_custom = earnings_filter.check_earnings_blackout(
        "NVDA", reference_date=date(2026, 11, 18), blackout_days_before=1, blackout_days_after=0
    )
    assert res_custom["is_blackout"] is False

    # (c) DataFrame calendar, ISO string, and get_earnings_dates fallback
    class _MockTickerDF:
        def __init__(self, tk):
            self.fast_info = {"quoteType": "EQUITY"}
            self.calendar = pd.DataFrame(
                {0: [pd.Timestamp("2026-11-19 16:00:00-05:00")]},
                index=["Earnings Date"],
            )

    monkeypatch.setattr(earnings_filter.yf, "Ticker", _MockTickerDF)
    res_df = earnings_filter.check_earnings_blackout("AMD", reference_date="2026-11-18")
    assert res_df["earnings_date"] == "2026-11-19"
    assert res_df["is_blackout"] is True

    class _MockTickerFallback:
        def __init__(self, tk):
            self.fast_info = {"quoteType": "EQUITY"}
            self.calendar = {}

        def get_earnings_dates(self, limit=4):
            return pd.DataFrame(
                {"EPS Estimate": [1.5]},
                index=pd.DatetimeIndex([pd.Timestamp("2026-10-30", tz="America/New_York")]),
            )

    monkeypatch.setattr(earnings_filter.yf, "Ticker", _MockTickerFallback)
    # (d) Past earnings label fix (no "งบถัดไป")
    res_past = earnings_filter.check_earnings_blackout("AMZN", reference_date=date(2026, 11, 18))
    assert res_past["is_blackout"] is False
    assert res_past["days_until_earnings"] == -19
    assert "งบล่าสุด" in res_past["status_label"]
    assert "งบถัดไป" not in res_past["status_label"]

    # (e) Generic ETF / 404 safety
    class _MockTickerETF404:
        def __init__(self, tk):
            self.fast_info = {"quoteType": "ETF"}

        @property
        def calendar(self):
            raise RuntimeError("HTTP Error 404: No fundamentals data found")

    monkeypatch.setattr(earnings_filter.yf, "Ticker", _MockTickerETF404)
    res_xlk = earnings_filter.check_earnings_blackout("XLK", reference_date=date(2026, 11, 18))
    assert res_xlk["is_blackout"] is False
    assert "ETF" in res_xlk["status_label"]


# ---------------------------------------------------------------------------
# T8 (F8): Real-Time BSM Pricing at spot_realtime, NY DTE & Liquidity Rules
# ---------------------------------------------------------------------------
def test_realtime_option_pricing_spot_vs_fibo_and_rules(monkeypatch):
    # 2026-11-18 19:00 EST -> NY date is 2026-11-18; 2026-11-19 expiry is 1 DTE (valid when min_dte=1)
    monkeypatch.setattr(options_pricing, "today_ny", lambda: date(2026, 11, 18))

    calls_df = pd.DataFrame(
        [
            {"strike": 100.0, "bid": 0.0, "ask": 0.0, "volume": 0, "openInterest": 0, "impliedVolatility": np.nan},
            {"strike": 98.0, "bid": 3.40, "ask": 3.50, "volume": 500, "openInterest": 1200, "impliedVolatility": 0.42},
            {"strike": 102.0, "bid": 1.10, "ask": 1.50, "volume": 100, "openInterest": 200, "impliedVolatility": 0.40},  # wide spread 30.8%
        ]
    )

    class _MockOptTicker:
        def __init__(self, tk):
            self.options = ("2026-11-18", "2026-11-19", "2026-11-27")

        def option_chain(self, exp):
            return SimpleNamespace(calls=calls_df, puts=calls_df)

    monkeypatch.setattr(options_pricing.yf, "Ticker", _MockOptTicker)

    # (a) spot_realtime (100.0) != fibo_entry_underlying (106.0)
    quote = options_pricing.select_and_price_realtime_option(
        ticker="AMD",
        direction="BUY CALL",
        spot_realtime=100.0,
        tp1_underlying=103.0,
        tp2_underlying=106.0,
        sl_underlying=97.0,
        fibo_entry_underlying=106.0,
        min_dte=1,
        max_dte=7,
        target_abs_delta=0.58,
        max_spread_pct=4.5,
        default_contracts_under_80=2,
        default_contracts_over_80=1,
    )
    assert quote.expiration == "2026-11-19"
    assert quote.dte == 1  # skipped 0-DTE 2026-11-18
    assert quote.strike == 98.0  # skipped illiquid 100.0 & wide-spread 102.0
    assert quote.spread_pct <= 4.5
    assert quote.recommended_contracts == 1  # spot >= 80

    expected_rt = round(
        options_pricing.black_scholes_price(100.0, 98.0, 1, 0.42, "CALL"), 2
    )
    expected_fibo = round(
        options_pricing.black_scholes_price(106.0, 98.0, 1, 0.42, "CALL"), 2
    )
    assert quote.realtime_theo_price == expected_rt
    assert quote.fibo_entry_theo_price == expected_fibo
    assert quote.est_bid <= quote.realtime_theo_price <= quote.est_ask
    assert quote.max_limit_buy == round(quote.est_ask + 0.02, 2)

    expected_tp1 = round(options_pricing.black_scholes_price(103.0, 98.0, 1, 0.42, "CALL"), 2)
    expected_tp2 = round(options_pricing.black_scholes_price(106.0, 98.0, 1, 0.42, "CALL"), 2)
    expected_sl = round(options_pricing.black_scholes_price(97.0, 98.0, 1, 0.42, "CALL"), 2)
    assert quote.tp1_option_price == expected_tp1
    assert quote.tp2_option_price == expected_tp2
    assert quote.sl_option_price == expected_sl

    # Under $80 contract sizing & option_type alias when direction=None
    quote_u80 = options_pricing.select_and_price_realtime_option(
        ticker="PLTR",
        option_type="PUT",
        spot_realtime=55.0,
        tp1_underlying=53.0,
        tp2_underlying=51.0,
        sl_underlying=57.0,
        default_contracts_under_80=2,
        default_contracts_over_80=1,
    )
    assert quote_u80.option_type == "PUT"
    assert quote_u80.recommended_contracts == 2
    assert math.isfinite(quote_u80.realtime_theo_price)
    assert math.isfinite(quote_u80.delta)

    # All 23 legacy keys present in __dict__
    legacy_keys = {
        "ticker", "option_type", "expiration", "dte", "strike", "spot_realtime", "iv",
        "realtime_theo_price", "est_bid", "est_ask", "max_limit_buy", "delta", "abs_delta",
        "gamma", "theta_daily", "tp1_option_price", "tp1_gain_pct", "tp2_option_price",
        "tp2_gain_pct", "sl_option_price", "sl_loss_pct", "recommended_contracts", "spread_pct",
    }
    assert legacy_keys.issubset(set(quote.__dict__.keys()))


# ---------------------------------------------------------------------------
# T9 (F9): time_utils DST/Holidays/Window & yfinance Retry / Normalization
# ---------------------------------------------------------------------------
def test_ny_tz_dst_window_and_yfinance_edge_cases(monkeypatch):
    # (a) EST (Jan = UTC-5) vs EDT (Jul = UTC-4)
    est_dt = datetime(2026, 1, 15, 10, 0, tzinfo=time_utils.NY_TZ)
    edt_dt = datetime(2026, 7, 15, 10, 0, tzinfo=time_utils.NY_TZ)
    assert est_dt.utcoffset().total_seconds() == -5 * 3600
    assert edt_dt.utcoffset().total_seconds() == -4 * 3600
    assert time_utils.format_ny_timestamp(est_dt) == "2026-01-15 10:00:00 ET"

    # (b) 2h window boundaries + weekends + NYSE holidays
    assert time_utils.is_within_2h_window_ny(datetime(2026, 11, 18, 9, 34, 59, tzinfo=time_utils.NY_TZ)) is False
    assert time_utils.is_within_2h_window_ny(datetime(2026, 11, 18, 9, 35, 0, tzinfo=time_utils.NY_TZ)) is True
    assert time_utils.is_within_2h_window_ny(datetime(2026, 11, 18, 11, 29, 59, tzinfo=time_utils.NY_TZ)) is True
    assert time_utils.is_within_2h_window_ny(datetime(2026, 11, 18, 11, 30, 0, tzinfo=time_utils.NY_TZ)) is False
    assert time_utils.is_within_2h_window_ny(datetime(2026, 11, 21, 10, 0, tzinfo=time_utils.NY_TZ)) is False  # Saturday
    assert time_utils.is_within_2h_window_ny(datetime(2026, 11, 26, 10, 0, tzinfo=time_utils.NY_TZ)) is False  # Thanksgiving 2026
    assert time_utils.is_within_2h_window_ny(datetime(2026, 4, 3, 10, 0, tzinfo=time_utils.NY_TZ)) is False   # Good Friday 2026

    # (c) ensure_ny_regular_hours: MultiIndex, UTC conversion, filtering, NaN/dup cleanup
    utc = ZoneInfo("UTC")
    raw_idx = pd.DatetimeIndex(
        [
            datetime(2026, 11, 18, 15, 0, tzinfo=utc),   # 10:00 ET (dup 1)
            datetime(2026, 11, 18, 14, 28, tzinfo=utc),  # 09:28 ET (pre-market -> drop)
            datetime(2026, 11, 18, 21, 0, tzinfo=utc),   # 16:00 ET (after-hours -> drop)
            datetime(2026, 11, 18, 15, 0, tzinfo=utc),   # 10:00 ET (dup 2 -> keep last)
            datetime(2026, 11, 18, 15, 2, tzinfo=utc),   # 10:02 ET (NaN close -> drop)
        ]
    )
    cols = pd.MultiIndex.from_tuples(
        [("Open", "SPY"), ("High", "SPY"), ("Low", "SPY"), ("Close", "SPY"), ("Volume", "SPY")]
    )
    raw_df = pd.DataFrame(
        [
            [500.0, 501.0, 499.0, 500.5, 100.0],
            [498.0, 499.0, 497.0, 498.5, 50.0],
            [505.0, 506.0, 504.0, 505.5, 50.0],
            [500.1, 501.2, 499.1, 500.8, 200.0],
            [501.0, 502.0, 500.0, np.nan, 150.0],
        ],
        index=raw_idx,
        columns=cols,
    )
    clean_df = time_utils.ensure_ny_regular_hours(raw_df)
    assert list(clean_df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(clean_df) == 1
    assert clean_df.iloc[0]["Close"] == 500.8
    assert str(clean_df.index.tz) == "America/New_York"

    # (d) fetch_intraday_2m retry succeeds on 2nd attempt
    attempts = {"n": 0}
    valid_df = make_2m_bars(n_prev=20, n_curr=5)

    class _FlakyTicker:
        def __init__(self, tk):
            pass

        def history(self, **kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise ConnectionError("429 Too Many Requests")
            return valid_df

    monkeypatch.setattr(fibo_scanner.time, "sleep", lambda s: None)
    monkeypatch.setattr(fibo_scanner.yf, "Ticker", _FlakyTicker)
    retried = fibo_scanner.fetch_intraday_2m("SPY", max_retries=3)
    assert attempts["n"] == 2
    assert len(retried) == 25


# ---------------------------------------------------------------------------
# T10 (F10): Paper Trader Gates, EOD Before TP/SL, Lock SL, File Lock & Flush
# ---------------------------------------------------------------------------
def _make_actionable_scan(
    ticker: str = "NVDA",
    option_action: str = "BUY PUT",
    spot: float = 106.0,
    fibo_entry: float = 107.0,
    tp1: float = 105.0,
    tp2: float = 103.82,
    sl: float = 107.86,
    lock_sl_after_tp1: float = 106.5,  # Intentionally bad (worse than entry 106.0 for PUT!)
    trading_date: str = "2026-11-18",
    is_current_session: bool = True,
) -> dict:
    opt_type = "PUT" if "PUT" in option_action else "CALL"
    theo = round(options_pricing.black_scholes_price(spot, spot, 3, 0.40, opt_type), 2)
    return {
        "timestamp": f"{trading_date} 10:15:00 ET",
        "results": [
            {
                "ticker": ticker,
                "is_etf": ticker in ("SPY", "QQQ"),
                "status": "ACTIONABLE",
                "current_trading_date": trading_date,
                "is_current_session": is_current_session,
                "spot_price": spot,
                "active_fibo_level": "0.618",
                "option_action": option_action,
                "trade_plan": {
                    "entry_underlying": spot,
                    "fibo_entry_level": "0.618",
                    "fibo_entry_price": fibo_entry,
                    "tp1_underlying": tp1,
                    "lock_sl_after_tp1": lock_sl_after_tp1,
                    "tp2_underlying": tp2,
                    "sl_underlying": sl,
                },
                "option_quote": {
                    "expiration": "2026-11-21",
                    "dte": 3,
                    "strike": spot,
                    "iv": 0.40,
                    "delta": -0.52 if opt_type == "PUT" else 0.52,
                    "est_ask": round(theo + 0.02, 2),
                    "realtime_theo_price": theo,
                    "recommended_contracts": 1,
                },
            }
        ],
    }


def test_paper_trader_2h_gate_eod_before_tpsl_lock_sl_persistence_and_flush(
    monkeypatch, tmp_path
):
    cfg = load_base_config()
    state_path = str(tmp_path / "paper_trades.json")

    # (a) Outside 2h window (12:00 NY): new ACTIONABLE signal is NOT entered
    _patch_clock(monkeypatch, datetime(2026, 11, 18, 12, 0, tzinfo=time_utils.NY_TZ))
    scan_entry = _make_actionable_scan()
    out_closed = paper_trader.update_paper_portfolio(scan_entry, cfg, enforce_window=True, path=state_path)
    assert out_closed["entry_window_open"] is False
    assert len(out_closed["newly_opened"]) == 0
    assert len(out_closed["state"]["open_positions"]) == 0

    # (b) Inside 2h window (10:15 NY): stale session (is_current_session=False) is blocked, live is opened
    _patch_clock(monkeypatch, datetime(2026, 11, 18, 10, 15, tzinfo=time_utils.NY_TZ))
    scan_stale = _make_actionable_scan(is_current_session=False)
    out_stale = paper_trader.update_paper_portfolio(scan_stale, cfg, enforce_window=True, path=state_path)
    assert len(out_stale["newly_opened"]) == 0

    out_open = paper_trader.update_paper_portfolio(scan_entry, cfg, enforce_window=True, path=state_path)
    assert out_open["entry_window_open"] is True
    assert len(out_open["newly_opened"]) == 1
    pos = out_open["newly_opened"][0]
    # (e) lock_sl_after_tp1 MUST be anchored to entry_underlying (106.0 -> 105.55), NOT 106.5
    assert pos["lock_sl_after_tp1"] < pos["entry_underlying"]
    assert pos["lock_sl_after_tp1"] == 105.55
    assert pos["opened_at"].endswith(" ET")

    # Duplicate same-day ticker+fibo prevented
    out_dup = paper_trader.update_paper_portfolio(scan_entry, cfg, enforce_window=True, path=state_path)
    assert len(out_dup["newly_opened"]) == 0

    # (c) Outside window, EOD_FORCE_FLUSH date-change runs BEFORE TP2/TP1/SL even if spot <= tp2!
    _patch_clock(monkeypatch, datetime(2026, 11, 19, 9, 40, tzinfo=time_utils.NY_TZ))
    scan_next_day = {
        "results": [
            {
                "ticker": "NVDA",
                "status": "WATCHING",
                "current_trading_date": "2026-11-19",
                "spot_price": 102.0,  # beyond tp2 (103.82), yet must trigger EOD_FORCE_FLUSH first
            }
        ]
    }
    out_eod = paper_trader.update_paper_portfolio(scan_next_day, cfg, enforce_window=True, path=state_path)
    assert len(out_eod["newly_closed"]) == 1
    assert out_eod["newly_closed"][0]["exit_reason"] == "EOD_FORCE_FLUSH"
    assert out_eod["newly_closed"][0]["closed_at"].endswith(" ET")

    # (d) TP1_TRAIL_WIN for PUT (profitable side of entry_underlying) and CALL TP2_FULL_WIN + SL_LOSS
    _patch_clock(monkeypatch, datetime(2026, 11, 20, 10, 0, tzinfo=time_utils.NY_TZ))
    state_tp1 = str(tmp_path / "tp1.json")
    paper_trader.update_paper_portfolio(
        _make_actionable_scan(ticker="AMD", trading_date="2026-11-20"),
        cfg,
        enforce_window=True,
        path=state_tp1,
    )
    # Hit TP1 (105.0)
    paper_trader.update_paper_portfolio(
        {"results": [{"ticker": "AMD", "current_trading_date": "2026-11-20", "spot_price": 104.9}]},
        cfg,
        path=state_tp1,
    )
    # Bounce back to lock_sl (105.55) -> TP1_TRAIL_WIN with exit < entry_underlying (106.0)
    res_trail = paper_trader.update_paper_portfolio(
        {"results": [{"ticker": "AMD", "current_trading_date": "2026-11-20", "spot_price": 105.7}]},
        cfg,
        path=state_tp1,
    )
    assert res_trail["newly_closed"][0]["exit_reason"] == "TP1_TRAIL_WIN"
    assert res_trail["newly_closed"][0]["exit_underlying"] < 106.0

    # CALL TP2_FULL_WIN & SL_LOSS
    state_call = str(tmp_path / "call.json")
    paper_trader.update_paper_portfolio(
        _make_actionable_scan(
            ticker="META",
            option_action="BUY CALL",
            spot=100.0,
            fibo_entry=99.0,
            tp1=102.0,
            tp2=104.0,
            sl=98.0,
            lock_sl_after_tp1=99.5,
            trading_date="2026-11-20",
        ),
        cfg,
        path=state_call,
    )
    res_tp2 = paper_trader.update_paper_portfolio(
        {"results": [{"ticker": "META", "current_trading_date": "2026-11-20", "spot_price": 104.5}]},
        cfg,
        path=state_call,
    )
    assert res_tp2["newly_closed"][0]["exit_reason"] == "TP2_FULL_WIN"

    state_sl = str(tmp_path / "sl.json")
    paper_trader.update_paper_portfolio(
        _make_actionable_scan(
            ticker="AAPL",
            option_action="BUY CALL",
            spot=100.0,
            fibo_entry=99.0,
            tp1=102.0,
            tp2=104.0,
            sl=98.0,
            lock_sl_after_tp1=100.9,
            trading_date="2026-11-20",
        ),
        cfg,
        path=state_sl,
    )
    res_sl = paper_trader.update_paper_portfolio(
        {"results": [{"ticker": "AAPL", "current_trading_date": "2026-11-20", "spot_price": 97.5}]},
        cfg,
        path=state_sl,
    )
    assert res_sl["newly_closed"][0]["exit_reason"] == "SL_LOSS"

    # (g) Corrupt JSON creates .corrupt.<ts>.bak backup without crashing
    corrupt_path = str(tmp_path / "corrupt.json")
    with open(corrupt_path, "w", encoding="utf-8") as f:
        f.write("{not valid json")
    recovered = paper_trader.load_paper_state(corrupt_path)
    assert recovered["open_positions"] == []
    bak_files = [n for n in os.listdir(tmp_path) if n.startswith("corrupt.json.corrupt.") and n.endswith(".bak")]
    assert len(bak_files) == 1

    # (h) Concurrent threads complete cleanly under _paper_state_lock
    conc_path = str(tmp_path / "concurrent.json")
    errors = []

    def _worker(idx: int):
        try:
            paper_trader.update_paper_portfolio(
                _make_actionable_scan(ticker=f"TK{idx}", trading_date="2026-11-20"),
                cfg,
                enforce_window=False,
                path=conc_path,
            )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    assert errors == []
    final_conc = paper_trader.load_paper_state(conc_path)
    assert len(final_conc["open_positions"]) == 6

    # (i) Session-end flush: fresh mark via latest_scan + custom starting_balance when empty
    flush_res = paper_trader.flush_open_paper_positions_at_session_end(
        cfg,
        latest_scan={"results": [{"ticker": "TK0", "spot_price": 103.0}]},
        path=conc_path,
    )
    assert len(flush_res["flushed"]) == 6
    tk0_closed = next(p for p in flush_res["flushed"] if p["ticker"] == "TK0")
    assert tk0_closed["exit_reason"] == "2H_WINDOW_CLOSE"
    assert tk0_closed["exit_underlying"] == 103.0

    cfg_10k = copy.deepcopy(cfg)
    cfg_10k["paper_trading"]["starting_balance_usd"] = 10000.0
    empty_flush_path = str(tmp_path / "empty_flush.json")
    empty_flush = paper_trader.flush_open_paper_positions_at_session_end(cfg_10k, path=empty_flush_path)
    assert empty_flush["flushed"] == []
    assert empty_flush["kpis"]["current_equity_usd"] == 10000.0


# ---------------------------------------------------------------------------
# Regression Tests (Challenger 1 Defects): Mixed-Offset DST & Gap-Across Touch
# ---------------------------------------------------------------------------
def _make_ohlcv_df(index_values):
    rows = [
        {"Open": 100.0 + i, "High": 101.0 + i, "Low": 99.0 + i, "Close": 100.5 + i, "Volume": 1000.0}
        for i in range(len(index_values))
    ]
    return pd.DataFrame(rows, index=pd.Index(index_values, dtype=object))


def test_ensure_ny_regular_hours_mixed_offsets_and_aware_naive_regression():
    # Case A: Mixed fixed-offset pd.Timestamp across fall-back ambiguous 01:30 + Monday regular hours
    df_a = _make_ohlcv_df(
        [
            pd.Timestamp("2026-11-01 01:30:00-04:00"),
            pd.Timestamp("2026-11-01 01:30:00-05:00"),
            pd.Timestamp("2026-11-02 09:30:00-05:00"),
        ]
    )
    out_a = time_utils.ensure_ny_regular_hours(df_a)
    assert len(out_a) == 1
    assert out_a.index[0] == pd.Timestamp("2026-11-02 09:30:00-05:00", tz="America/New_York")
    assert str(out_a.index.tz) == "America/New_York"

    # Case B: Friday EDT -> Monday EST pd.Timestamp object Index retains all 3 bars
    df_b = _make_ohlcv_df(
        [
            pd.Timestamp("2026-10-30 15:58:00-04:00"),
            pd.Timestamp("2026-11-02 09:30:00-05:00"),
            pd.Timestamp("2026-11-02 09:32:00-05:00"),
        ]
    )
    out_b = time_utils.ensure_ny_regular_hours(df_b)
    assert len(out_b) == 3
    assert str(out_b.index.tz) == "America/New_York"

    # Case C: Mixed datetime.timezone(timedelta) objects across DST retain all 3 bars
    tz_edt = timezone(timedelta(hours=-4))
    tz_est = timezone(timedelta(hours=-5))
    df_c = _make_ohlcv_df(
        [
            datetime(2026, 10, 30, 15, 58, tzinfo=tz_edt),
            datetime(2026, 11, 2, 9, 30, tzinfo=tz_est),
            datetime(2026, 11, 2, 9, 32, tzinfo=tz_est),
        ]
    )
    out_c = time_utils.ensure_ny_regular_hours(df_c)
    assert len(out_c) == 3
    assert str(out_c.index.tz) == "America/New_York"

    # Case D: Mixed tz-aware + naive strings retain both rows
    df_d = _make_ohlcv_df(
        [
            "2026-11-02 09:30:00-05:00",
            "2026-11-02 09:32:00",
        ]
    )
    out_d = time_utils.ensure_ny_regular_hours(df_d)
    assert len(out_d) == 2
    assert str(out_d.index.tz) == "America/New_York"

    # Case E: Pure-naive nonexistent/ambiguous DST boundaries + valid regular hours bar
    df_e = _make_ohlcv_df(
        [
            "2026-03-08 02:30:00",
            "2026-11-01 01:30:00",
            "2026-11-01 01:30:00",
            "2026-11-02 09:30:00",
        ]
    )
    out_e = time_utils.ensure_ny_regular_hours(df_e)
    assert len(out_e) == 1
    assert out_e.index[0] == pd.Timestamp("2026-11-02 09:30:00-05:00", tz="America/New_York")


def test_detect_reversal_candle_gap_across_level_regression():
    ny = time_utils.NY_TZ
    idx = pd.DatetimeIndex(
        [
            datetime(2026, 11, 18, 10, 0, tzinfo=ny),
            datetime(2026, 11, 18, 10, 2, tzinfo=ny),
        ]
    )

    # DOWNTREND: prev_bar [109, 111] -> curr_bar [89, 91] gaps completely across level=100.0
    df_gap_down = pd.DataFrame(
        [
            {"Open": 110.0, "High": 111.0, "Low": 109.0, "Close": 109.5},
            {"Open": 90.5, "High": 91.0, "Low": 89.0, "Close": 89.5},
        ],
        index=idx,
    )
    res_down = fibo_scanner.detect_reversal_candle(df_gap_down, "DOWNTREND", 100.0, 10.0)
    assert res_down["touched"] is False
    assert res_down["confirmed"] is False
    assert "Reversal Candle" not in res_down["pattern_name"]

    # UPTREND: prev_bar [89, 91] -> curr_bar [109, 111] gaps completely across level=100.0
    df_gap_up = pd.DataFrame(
        [
            {"Open": 90.0, "High": 91.0, "Low": 89.0, "Close": 90.5},
            {"Open": 109.5, "High": 111.0, "Low": 109.0, "Close": 110.5},
        ],
        index=idx,
    )
    res_up = fibo_scanner.detect_reversal_candle(df_gap_up, "UPTREND", 100.0, 10.0)
    assert res_up["touched"] is False
    assert res_up["confirmed"] is False
    assert "Reversal Candle" not in res_up["pattern_name"]

    # Control: prev_bar genuinely touches tolerance zone of 100.0, curr_bar confirms below
    df_real_touch_down = pd.DataFrame(
        [
            {"Open": 99.7, "High": 100.1, "Low": 99.6, "Close": 100.0},
            {"Open": 99.9, "High": 100.0, "Low": 99.3, "Close": 99.4},
        ],
        index=idx,
    )
    res_ctrl = fibo_scanner.detect_reversal_candle(df_real_touch_down, "DOWNTREND", 100.0, 10.0)
    assert res_ctrl["touched"] is True
    assert res_ctrl["confirmed"] is True
