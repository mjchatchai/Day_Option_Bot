"""
Unit tests for Day Option Bot:
- Previous-Day Asymmetric Fibonacci calculation (Downtrend Wick Low -> Body High vs Uptrend Wick High -> Body Low)
- Real-Time Black-Scholes Option Pricing & Greeks (Zero 15-min Delay)
- Paper Trade Two-Stage Exit Engine (TP1 Lock Profit -> TP2 Runner -> SL -> Win Rate KPIs)
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from day_option_bot.src.fibo_scanner import build_fibo_levels, extract_prev_day_fibo_anchors
from day_option_bot.src.options_pricing import black_scholes_price, calculate_greeks
from day_option_bot.src.paper_trader import compute_paper_kpis


def test_asymmetric_fibo_downtrend_and_uptrend():
    # Previous day bars:
    # Bar 0 (lowest): Open=102, High=103, Low=100 (Wick Low=100), Close=101 (BodyLow=101)
    # Bar 1 (highest): Open=109, High=112 (Wick High=112), Low=108, Close=110 (BodyHigh=110)
    df = pd.DataFrame({
        "Open": [102.0, 109.0],
        "High": [103.0, 112.0],
        "Low": [100.0, 108.0],
        "Close": [101.0, 110.0],
        "BodyHigh": [102.0, 110.0],
        "BodyLow": [101.0, 109.0],
    })
    anchors = extract_prev_day_fibo_anchors(df)
    assert anchors["wick_low"] == 100.0
    assert anchors["body_high"] == 110.0
    assert anchors["wick_high"] == 112.0
    assert anchors["body_low"] == 101.0

    # Downtrend: Point 1 = Wick Low (100), Point 2 = Body High (110), Range = 10
    down_fibo = build_fibo_levels(anchors, "DOWNTREND")
    assert down_fibo["0.000"] == 100.0
    assert down_fibo["0.500"] == 105.0
    assert down_fibo["0.618"] == 106.18
    assert down_fibo["0.786"] == 107.86
    assert down_fibo["1.000"] == 110.0

    # Uptrend: Point 1 = Wick High (112), Point 2 = Body Low (101), Range = 11
    up_fibo = build_fibo_levels(anchors, "UPTREND")
    assert up_fibo["0.000"] == 112.0
    assert up_fibo["0.500"] == 106.5
    assert up_fibo["0.618"] == round(112.0 - 0.618 * 11.0, 2)
    assert up_fibo["1.000"] == 101.0


def test_realtime_black_scholes_pricing():
    call_p = black_scholes_price(spot=150.0, strike=150.0, dte_days=5, iv=0.40, option_type="CALL")
    put_p = black_scholes_price(spot=150.0, strike=150.0, dte_days=5, iv=0.40, option_type="PUT")
    assert call_p > 1.0
    assert put_p > 1.0

    greeks_put = calculate_greeks(spot=150.0, strike=152.5, dte_days=5, iv=0.40, option_type="PUT")
    assert -0.85 <= greeks_put["delta"] <= -0.45
    assert greeks_put["abs_delta"] >= 0.45


def test_paper_kpis_winrate_calculation():
    closed = [
        {"ticker": "AMD", "fibo_level": "0.618", "net_pnl_usd": 120.0},
        {"ticker": "AMD", "fibo_level": "0.618", "net_pnl_usd": 85.0},
        {"ticker": "SPY", "fibo_level": "0.500", "net_pnl_usd": 65.0},
        {"ticker": "META", "fibo_level": "0.618", "net_pnl_usd": -45.0},
    ]
    kpis = compute_paper_kpis(closed, [], starting_balance=5000.0)
    assert kpis["total_closed"] == 4
    assert kpis["wins"] == 3
    assert kpis["losses"] == 1
    assert kpis["win_rate_pct"] == 75.0
    assert kpis["net_pnl_usd"] == 225.0
    assert kpis["fibo_0618"]["win_rate_pct"] == 66.7
    assert kpis["fibo_0500"]["win_rate_pct"] == 100.0
