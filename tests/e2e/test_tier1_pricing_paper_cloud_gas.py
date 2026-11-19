"""
Tier 1 Feature Coverage — Pricing, Paper Trading, Telegram, Main Loop,
Cloud Workflow, and GAS Webhook (F8–F14): 35 tests (5 per feature).
Opaque-box tests derived from ORIGINAL_REQUEST.md R1–R4 and PROJECT.md F8–F14.
"""

import base64
from datetime import date, datetime, timedelta
import json
import os
from types import SimpleNamespace
import pandas as pd
import pytest

from day_option_bot.src import (
    command_handler,
    dashboard_builder,
    fibo_scanner,
    main as bot_main,
    options_pricing,
    paper_trader,
    telegram_notifier,
)
from day_option_bot.tests.e2e.conftest import (
    NY_TZ,
    call_flush_paper,
    call_select_option,
    call_update_paper,
    cli_has_flag,
    get_execute_step_run_script,
    get_has_active_telegram_webhook,
    import_time_utils,
    load_cloud_workflow,
    make_intraday_2m_df,
    make_option_chain_df,
    run_cloud_shell_simulation,
    run_gas_function,
)


def _make_actionable_scan_item(
    ticker: str = "AMD",
    is_etf: bool = False,
    spot: float = 105.20,
    action: str = "BUY CALL",
    fibo_level: str = "0.618",
    tp1: float = 106.50,
    tp2: float = 107.80,
    sl: float = 103.35,
    lock_sl: float = 105.79,
    curr_date: str = "2026-11-18",
    status: str = "ACTIONABLE",
    contracts: int = 1,
):
    opt_type = "PUT" if "PUT" in action else "CALL"
    strike = 104.0 if opt_type == "CALL" else 106.0
    iv = 0.38
    dte = 3
    theo = round(options_pricing.black_scholes_price(spot, strike, dte, iv, opt_type), 2)
    est_bid = round(max(0.01, theo * 0.99), 2)
    est_ask = round(theo * 1.01, 2)
    tp1_opt = round(options_pricing.black_scholes_price(tp1, strike, dte, iv, opt_type), 2)
    tp2_opt = round(options_pricing.black_scholes_price(tp2, strike, dte, iv, opt_type), 2)
    sl_opt = round(options_pricing.black_scholes_price(sl, strike, dte, iv, opt_type), 2)
    greeks = options_pricing.calculate_greeks(spot, strike, dte, iv, opt_type)
    return {
        "ticker": ticker,
        "is_etf": is_etf,
        "status": status,
        "status_thai": f"🔥 {ticker}",
        "score": 95,
        "prev_trading_date": "2026-11-17",
        "current_trading_date": curr_date,
        "spot_price": spot,
        "ema200": 102.0 if opt_type == "CALL" else 109.0,
        "vwap": spot,
        "rvol": 1.35,
        "rvol_passed": True,
        "morning_gap_pct": 0.2,
        "trend_direction": "UPTREND" if opt_type == "CALL" else "DOWNTREND",
        "option_action": action,
        "trend_thai": "trend",
        "primary_fibo_level": "0.500" if is_etf else "0.618",
        "active_fibo_level": fibo_level,
        "active_fibo_price": spot,
        "distance_to_active_fibo_pct": 0.05,
        "fibo_levels": {
            "anchor_point_1_label": "Wick High ($112.00)",
            "anchor_point_2_label": "Body Low ($101.00)",
            "range_usd": 11.0,
            "0.000": 112.0,
            "0.236": 109.40,
            "0.382": 107.80,
            "0.500": 106.50,
            "0.618": 105.20,
            "0.786": 103.35,
            "1.000": 101.0,
        },
        "candle_reversal": {"touched": True, "confirmed": True, "pattern_name": "Bullish Hammer"},
        "trade_plan": {
            "entry_underlying": spot,
            "fibo_entry_level": fibo_level,
            "fibo_entry_price": spot,
            "tp1_underlying": tp1,
            "tp1_label": "TP1",
            "lock_sl_after_tp1": lock_sl,
            "tp2_underlying": tp2,
            "tp2_label": "TP2",
            "sl_underlying": sl,
            "sl_label": "SL",
        },
        "option_quote": {
            "ticker": ticker,
            "option_type": opt_type,
            "expiration": "2026-11-21",
            "dte": dte,
            "strike": strike,
            "spot_realtime": spot,
            "iv": iv,
            "realtime_theo_price": theo,
            "est_bid": est_bid,
            "est_ask": est_ask,
            "max_limit_buy": round(est_ask + 0.02, 2),
            "delta": greeks["delta"],
            "abs_delta": greeks["abs_delta"],
            "gamma": greeks["gamma"],
            "theta_daily": greeks["theta_daily"],
            "tp1_option_price": tp1_opt,
            "tp1_gain_pct": round((tp1_opt - theo) / max(0.05, theo) * 100.0, 1),
            "tp2_option_price": tp2_opt,
            "tp2_gain_pct": round((tp2_opt - theo) / max(0.05, theo) * 100.0, 1),
            "sl_option_price": sl_opt,
            "sl_loss_pct": round((sl_opt - theo) / max(0.05, theo) * 100.0, 1),
            "recommended_contracts": contracts,
            "spread_pct": 2.0,
        },
        "earnings": {"ticker": ticker, "is_blackout": False, "earnings_date": None, "days_until_earnings": None, "status_label": "SAFE"},
    }


# ===========================================================================
# F8: Real-Time BSM Option Pricing (5 tests)
# ===========================================================================

def test_t1_f08_prices_entry_bid_ask_limit_at_spot_realtime(mock_yfinance_factory):
    ref = date(2026, 11, 18)
    exp = "2026-11-21"
    # Deliberately separate spot_realtime (108.00) from fibo_entry_underlying (106.00)
    chain_calls = make_option_chain_df(spot=108.00, option_type="CALL", strikes=[105.0, 107.5, 110.0], dte_days=3, iv=0.40)
    mock_yfinance_factory({
        "AMD": {
            "options": (exp,),
            "option_chains": {exp: SimpleNamespace(calls=chain_calls, puts=pd.DataFrame())},
        }
    })

    q = call_select_option(
        ticker="AMD",
        option_type="CALL",
        spot_realtime=108.00,
        fibo_entry_underlying=106.00,
        tp1_underlying=109.50,
        tp2_underlying=111.00,
        sl_underlying=104.50,
        reference_date=ref,
    )
    assert q is not None
    expected_at_spot = round(options_pricing.black_scholes_price(108.00, q["strike"], q["dte"], q["iv"], "CALL"), 2)
    wrong_at_fibo = round(options_pricing.black_scholes_price(106.00, q["strike"], q["dte"], q["iv"], "CALL"), 2)
    assert expected_at_spot != wrong_at_fibo
    assert q["realtime_theo_price"] == pytest.approx(expected_at_spot, abs=0.02)
    assert q["est_bid"] <= q["realtime_theo_price"] <= q["est_ask"]
    assert q["max_limit_buy"] >= q["est_ask"]


def test_t1_f08_projects_tp1_tp2_sl_monotonically(mock_yfinance_factory):
    mock_yfinance_factory({"AMD": {"options": ()}, "SPY": {"options": ()}})
    ref = date(2026, 11, 18)

    call_q = call_select_option("AMD", "CALL", 105.20, 105.20, 106.50, 107.80, 103.35, reference_date=ref)
    assert call_q["tp2_option_price"] >= call_q["tp1_option_price"] >= call_q["realtime_theo_price"] >= call_q["sl_option_price"]

    put_q = call_select_option("SPY", "PUT", 105.00, 105.00, 103.82, 102.36, 106.18, reference_date=ref)
    assert put_q["tp2_option_price"] >= put_q["tp1_option_price"] >= put_q["realtime_theo_price"] >= put_q["sl_option_price"]


def test_t1_f08_selects_dte_within_min_max_dte_ny(mock_yfinance_factory, freeze_ny_time):
    ref = date(2026, 11, 18)
    freeze_ny_time(datetime(2026, 11, 18, 10, 0))
    exps = ("2026-11-18", "2026-11-21", "2026-12-02")  # 0 DTE, 3 DTE, 14 DTE
    chain_3 = make_option_chain_df(spot=105.20, option_type="CALL", strikes=[104.0, 105.0, 106.0], dte_days=3)
    mock_yfinance_factory({
        "AMD": {
            "options": exps,
            "option_chains": {exp: SimpleNamespace(calls=chain_3, puts=pd.DataFrame()) for exp in exps},
        }
    })

    q = call_select_option(
        "AMD", "CALL", 105.20, 105.20, 106.50, 107.80, 103.35,
        min_dte=1, max_dte=7, reference_date=ref,
    )
    assert q["expiration"] == "2026-11-21"
    assert 1 <= q["dte"] <= 7


def test_t1_f08_selects_liquid_contract_near_target_delta(mock_yfinance_factory, freeze_ny_time):
    ref = date(2026, 11, 18)
    freeze_ny_time(datetime(2026, 11, 18, 10, 0))
    exp = "2026-11-21"
    # Strike 105.0 is ATM (closer dist to spot) but completely illiquid (bid=0, vol=0, OI=0)
    # Strike 104.0 is liquid (bid>0, vol=250, OI=800) with |delta| ~ 0.60 near target 0.58
    df_calls = pd.DataFrame([
        {"contractSymbol": "ILLIQ", "strike": 105.0, "lastPrice": 2.0, "bid": 0.0, "ask": 0.0, "volume": 0, "openInterest": 0, "impliedVolatility": 0.40},
        {"contractSymbol": "LIQ", "strike": 104.0, "lastPrice": 2.6, "bid": 2.55, "ask": 2.62, "volume": 250, "openInterest": 800, "impliedVolatility": 0.40},
        {"contractSymbol": "DEEP", "strike": 95.0, "lastPrice": 10.5, "bid": 10.30, "ask": 10.50, "volume": 100, "openInterest": 200, "impliedVolatility": 0.40},
    ])
    mock_yfinance_factory({
        "AMD": {
            "options": (exp,),
            "option_chains": {exp: SimpleNamespace(calls=df_calls, puts=pd.DataFrame())},
        }
    })

    q = call_select_option(
        "AMD", "CALL", 105.20, 105.20, 106.50, 107.80, 103.35,
        target_abs_delta=0.58, max_spread_pct=4.5, reference_date=ref,
    )
    assert q["strike"] == 104.0


def test_t1_f08_contract_sizing_under_vs_over_80(mock_yfinance_factory):
    mock_yfinance_factory({"PLTR": {"options": ()}, "AMD": {"options": ()}})
    ref = date(2026, 11, 18)

    q_under = call_select_option("PLTR", "CALL", 50.0, 50.0, 51.0, 52.0, 49.0, reference_date=ref)
    q_over = call_select_option("AMD", "CALL", 150.0, 150.0, 152.0, 154.0, 148.0, reference_date=ref)
    assert q_under["recommended_contracts"] == 2
    assert q_over["recommended_contracts"] == 1


# ===========================================================================
# F9: yfinance & NY TZ/DST Helper (5 tests)
# ===========================================================================

def test_t1_f09_now_ny_and_today_ny_return_america_new_york():
    tu = import_time_utils()
    now_dt = tu.now_ny()
    today_dt = tu.today_ny()
    assert str(now_dt.tzinfo) == "America/New_York"
    assert today_dt == now_dt.date()
    formatted = tu.format_ny_timestamp(datetime(2026, 11, 18, 9, 35, 12, tzinfo=NY_TZ))
    assert formatted == "2026-11-18 09:35:12 ET"


def test_t1_f09_is_within_2h_window_ny_true_and_false():
    tu = import_time_utils()
    assert tu.is_within_2h_window_ny(datetime(2026, 11, 18, 10, 15, tzinfo=NY_TZ)) is True
    assert tu.is_within_2h_window_ny(datetime(2026, 11, 18, 9, 30, tzinfo=NY_TZ)) is False
    assert tu.is_within_2h_window_ny(datetime(2026, 11, 18, 11, 30, tzinfo=NY_TZ)) is False
    # Saturday 10:15 NY -> False
    assert tu.is_within_2h_window_ny(datetime(2026, 11, 21, 10, 15, tzinfo=NY_TZ)) is False


def test_t1_f09_ensure_ny_regular_hours_converts_utc_and_filters_0930_1559():
    tu = import_time_utils()
    df_utc = make_intraday_2m_df(
        trend="DOWNTREND",
        active_level=106.18,
        tz_mode="UTC",
        include_extended_hours=True,
    )
    cleaned = tu.ensure_ny_regular_hours(df_utc)
    assert str(cleaned.index.tz) == "America/New_York"
    times_hm = cleaned.index.hour * 60 + cleaned.index.minute
    assert int(times_hm.min()) >= 9 * 60 + 30
    assert int(times_hm.max()) <= 15 * 60 + 59
    assert float(cleaned["Low"].min()) >= 99.0


def test_t1_f09_flattens_multiindex_and_dedups_sorts():
    tu = import_time_utils()
    df_mi = make_intraday_2m_df(
        trend="DOWNTREND",
        active_level=106.18,
        multiindex=True,
        ticker="AMD",
        inject_duplicates_and_nans=True,
    )
    cleaned = tu.ensure_ny_regular_hours(df_mi)
    assert not isinstance(cleaned.columns, pd.MultiIndex)
    assert set(["Open", "High", "Low", "Close", "Volume"]).issubset(cleaned.columns)
    assert cleaned.index.is_monotonic_increasing
    assert not cleaned.index.has_duplicates
    assert not cleaned[["Open", "High", "Low", "Close"]].isna().any().any()


def test_t1_f09_rate_limit_retry_succeeds_on_second_attempt(sample_config, mock_yfinance_factory):
    valid_df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, rvol_target=1.30)
    attempts = {"n": 0}

    def _flaky_history(period="5d", interval="2m"):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("429 Too Many Requests")
        return valid_df

    mock_yfinance_factory({"AMD": {"history_2m": _flaky_history, "calendar": {}}})
    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert attempts["n"] >= 2
    assert res["status"] == "ACTIONABLE"


# ===========================================================================
# F10: Paper Trading & 2h Flush (5 tests)
# ===========================================================================

def test_t1_f10_auto_enter_actionable_inside_2h_window(sample_config, freeze_ny_time):
    freeze_ny_time(datetime(2026, 11, 18, 9, 45))
    item = _make_actionable_scan_item(ticker="AMD", is_etf=False, spot=105.20)
    res = call_update_paper({"results": [item]}, sample_config, enforce_window=True)

    assert len(res["newly_opened"]) == 1
    opened = res["newly_opened"][0]
    assert opened["entry_option_price"] == item["option_quote"]["est_ask"]
    assert os.path.exists(paper_trader.PAPER_STATE_PATH)
    loaded = paper_trader.load_paper_state(paper_trader.PAPER_STATE_PATH)
    assert len(loaded["open_positions"]) == 1


def test_t1_f10_tp1_hit_locks_trailing_stop_and_closes_tp1_trail_win(sample_config, freeze_ny_time):
    freeze_ny_time(datetime(2026, 11, 18, 9, 45))
    entry_item = _make_actionable_scan_item(
        ticker="AMD", spot=105.20, action="BUY CALL",
        tp1=106.50, tp2=107.80, sl=103.35, lock_sl=105.79,
    )
    call_update_paper({"results": [entry_item]}, sample_config, enforce_window=True)

    # Tick 2: spot hits TP1 (106.55) -> tp1_hit=True, sl_underlying becomes lock_sl_after_tp1 (105.79)
    tp1_item = _make_actionable_scan_item(ticker="AMD", spot=106.55, status="WATCHING")
    mid = call_update_paper({"results": [tp1_item]}, sample_config, enforce_window=True)
    assert len(mid["newly_closed"]) == 0
    assert mid["state"]["open_positions"][0]["tp1_hit"] is True

    # Tick 3: spot retraces to lock_sl_after_tp1 (105.75 <= 105.79) -> closes TP1_TRAIL_WIN
    trail_item = _make_actionable_scan_item(ticker="AMD", spot=105.75, status="WATCHING")
    done = call_update_paper({"results": [trail_item]}, sample_config, enforce_window=True)
    assert len(done["newly_closed"]) == 1
    closed = done["newly_closed"][0]
    assert closed["exit_reason"] == "TP1_TRAIL_WIN"
    assert closed["net_pnl_usd"] > 0
    assert closed["is_win"] is True


def test_t1_f10_tp2_full_win_and_sl_loss(sample_config, freeze_ny_time):
    freeze_ny_time(datetime(2026, 11, 18, 9, 45))
    amd = _make_actionable_scan_item(ticker="AMD", spot=105.20, tp1=106.50, tp2=107.80, sl=103.35)
    nvda = _make_actionable_scan_item(ticker="NVDA", spot=105.20, tp1=106.50, tp2=107.80, sl=103.35)
    call_update_paper({"results": [amd, nvda]}, sample_config, enforce_window=True)

    # AMD hits TP2 (108.00), NVDA hits SL (103.00)
    tick2 = {
        "results": [
            _make_actionable_scan_item(ticker="AMD", spot=108.00, status="WATCHING"),
            _make_actionable_scan_item(ticker="NVDA", spot=103.00, status="WATCHING"),
        ]
    }
    res = call_update_paper(tick2, sample_config, enforce_window=True)
    reasons = {c["ticker"]: c["exit_reason"] for c in res["newly_closed"]}
    assert reasons["AMD"] == "TP2_FULL_WIN"
    assert reasons["NVDA"] == "SL_LOSS"


def test_t1_f10_flush_at_session_end_closes_all_with_2h_window_close(sample_config, freeze_ny_time):
    freeze_ny_time(datetime(2026, 11, 18, 9, 45))
    amd = _make_actionable_scan_item(ticker="AMD", spot=105.20, tp1=106.50, tp2=107.80, sl=103.35)
    call_update_paper({"results": [amd]}, sample_config, enforce_window=True)

    # Re-mark with latest_scan at higher spot (106.10) during session-end flush
    latest_scan = {"results": [_make_actionable_scan_item(ticker="AMD", spot=106.10, status="WATCHING")]}
    res = call_flush_paper(sample_config, latest_scan=latest_scan)
    assert len(res["flushed"]) == 1
    closed = res["flushed"][0]
    assert closed["exit_reason"] == "2H_WINDOW_CLOSE"
    assert closed["exit_underlying"] == pytest.approx(106.10, abs=0.01)
    assert paper_trader.load_paper_state(paper_trader.PAPER_STATE_PATH)["open_positions"] == []


def test_t1_f10_kpis_fibo_and_per_ticker_breakdown():
    closed = [
        {"ticker": "AMD", "fibo_level": "0.618", "net_pnl_usd": 120.0},
        {"ticker": "AMD", "fibo_level": "0.618", "net_pnl_usd": -60.0},
        {"ticker": "SPY", "fibo_level": "0.500", "net_pnl_usd": 90.0},
    ]
    kpis = paper_trader.compute_paper_kpis(closed, [], starting_balance=5000.0)
    assert kpis["total_closed"] == 3
    assert kpis["wins"] == 2
    assert kpis["losses"] == 1
    assert kpis["win_rate_pct"] == 66.7
    assert kpis["net_pnl_usd"] == 150.0
    assert kpis["current_equity_usd"] == 5150.0
    assert kpis["profit_factor"] == pytest.approx(210.0 / 60.0, abs=0.01)
    assert kpis["realized_rr"] == pytest.approx(105.0 / 60.0, abs=0.01)
    assert kpis["fibo_0500"]["win_rate_pct"] == 100.0
    assert kpis["fibo_0618"]["win_rate_pct"] == 50.0
    assert kpis["per_ticker"]["AMD"]["net_pnl_usd"] == 60.0
    assert kpis["per_ticker"]["SPY"]["wins"] == 1


# ===========================================================================
# F11: Telegram 5-Line Cards & Delivery (5 tests)
# ===========================================================================

def test_t1_f11_format_signal_card_exact_5_lines():
    item = _make_actionable_scan_item(ticker="AMD", spot=105.20)
    card = telegram_notifier.format_signal_card(item)
    lines = [ln for ln in card.splitlines() if ln.strip()]
    assert len(lines) == 5
    assert "AMD" in card
    assert "BUY CALL" in card
    assert "2026-11-21" in card
    assert "TP1" in card and "TP2" in card and "SL" in card


def test_t1_f11_html_escapes_dynamic_fields():
    item = _make_actionable_scan_item(ticker="A<M>&D", spot=105.20)
    item["option_quote"]["expiration"] = "2026<11>&21"
    card = telegram_notifier.format_signal_card(item)
    assert "A&lt;M&gt;&amp;D" in card
    assert "2026&lt;11&gt;&amp;21" in card
    assert "<b>" in card and "</b>" in card


def test_t1_f11_build_scan_report_includes_actionable_and_near_sorted():
    act = _make_actionable_scan_item(ticker="AMD", status="ACTIONABLE")
    near = _make_actionable_scan_item(ticker="SPY", is_etf=True, status="NEAR_SETUP")
    scan_out = {"timestamp": "2026-11-18 09:45:00 ET", "results": [act, near]}
    paper_summary = {"kpis": {"win_rate_pct": 75.0, "wins": 3, "losses": 1, "net_pnl_usd": 225.0}}

    report = telegram_notifier.build_scan_telegram_report(scan_out, paper_summary)
    assert "AMD" in report and "SPY" in report
    assert "75.0%" in report and "3W/1L" in report


def test_t1_f11_send_telegram_message_posts_html_and_returns_true_on_200(monkeypatch, telegram_mock):
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "bot_tok_123")
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_CHAT_ID", "8964251574")
    tg = telegram_mock(post_statuses=[200])

    ok = telegram_notifier.send_telegram_message("<b>Hello</b>")
    assert ok is True
    posts = [c for c in tg["calls"] if c["method"] == "POST"]
    assert len(posts) == 1
    assert posts[0]["json"]["parse_mode"] == "HTML"
    assert posts[0]["json"]["chat_id"] == "8964251574"


def test_t1_f11_send_telegram_message_retries_on_429_500(monkeypatch, telegram_mock):
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "bot_tok_123")
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_CHAT_ID", "8964251574")
    monkeypatch.setattr(telegram_notifier.time, "sleep", lambda s: None) if hasattr(telegram_notifier, "time") else None
    import time as py_time
    monkeypatch.setattr(py_time, "sleep", lambda s: None)

    tg1 = telegram_mock(post_statuses=[429, 200])
    assert telegram_notifier.send_telegram_message("retry ok") is True
    assert len([c for c in tg1["calls"] if c["method"] == "POST"]) == 2

    tg2 = telegram_mock(post_statuses=[500, 500, 500, 500])
    assert telegram_notifier.send_telegram_message("retry fail") is False
    assert len([c for c in tg2["calls"] if c["method"] == "POST"]) >= 2


# ===========================================================================
# F12: Webhook vs Polling & main.py Loop (5 tests)
# ===========================================================================

def test_t1_f12_execute_single_cycle_writes_scan_paper_dashboard_and_notifies(
    sample_config, monkeypatch, telegram_mock, freeze_ny_time
):
    freeze_ny_time(datetime(2026, 11, 18, 9, 45))
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "bot_tok_123")
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_CHAT_ID", "8964251574")
    tg = telegram_mock(post_statuses=[200])

    item = _make_actionable_scan_item(ticker="AMD", spot=105.20)
    monkeypatch.setattr(
        bot_main,
        "run_full_day_option_scan",
        lambda cfg: {
            "timestamp": "2026-11-18 09:45:00 ET",
            "total_scanned": 1,
            "actionable_count": 1,
            "near_setup_count": 0,
            "blackout_count": 0,
            "results": [item],
        },
    )

    out = bot_main.execute_single_cycle(sample_config, notify_telegram=True, ny_session_date="2026-11-18")
    assert os.path.exists(bot_main.LATEST_SCAN_PATH)
    saved_scan = json.loads(open(bot_main.LATEST_SCAN_PATH, encoding="utf-8").read())
    assert saved_scan["ny_session_date"] == "2026-11-18"
    assert os.path.exists(paper_trader.PAPER_STATE_PATH)
    assert os.path.exists(out["dashboard"])
    assert len([c for c in tg["calls"] if c["method"] == "POST"]) == 1


def test_t1_f12_skips_getupdates_when_webhook_active(
    sample_config, monkeypatch, telegram_mock, freeze_ny_time
):
    clock = freeze_ny_time(datetime(2026, 11, 18, 9, 35))
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "bot_tok_f12_active")
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_CHAT_ID", "8964251574")
    tg = telegram_mock(webhook_url="https://script.google.com/macros/s/AKfycb/exec?market=DAY_OPTION")

    has_wh = get_has_active_telegram_webhook()
    assert has_wh("bot_tok_f12_active") is True

    fake_ts = {"t": 1000.0}
    monkeypatch.setattr(bot_main.time, "time", lambda: fake_ts["t"])

    def _sleep(sec):
        fake_ts["t"] += sec
        clock["now"] += timedelta(seconds=sec)

    monkeypatch.setattr(bot_main.time, "sleep", _sleep)
    monkeypatch.setattr(
        bot_main,
        "run_full_day_option_scan",
        lambda cfg: {
            "timestamp": "2026-11-18 09:35:00 ET",
            "total_scanned": 1,
            "actionable_count": 0,
            "near_setup_count": 0,
            "blackout_count": 0,
            "results": [_make_actionable_scan_item(ticker="AMD", status="WATCHING")],
        },
    )

    bot_main.run_continuous_30s_loop(sample_config, interval_sec=30, max_minutes=1, ny_session_date="2026-11-18")
    get_updates_calls = [c for c in tg["calls"] if c["method"] == "GET" and c["url"].endswith("/getUpdates")]
    assert len(get_updates_calls) == 0


def test_t1_f12_polls_commands_when_no_webhook_active(
    sample_config, monkeypatch, telegram_mock, isolated_bot_paths
):
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "bot_tok_f12_none")
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_CHAT_ID", "8964251574")
    with open(isolated_bot_paths["backtest_path"], "w", encoding="utf-8") as f:
        json.dump({
            "overall_original": {"win_rate_pct": 55.0, "net_pnl": 400.0},
            "overall_golden": {"win_rate_pct": 66.7, "net_pnl": 1480.05, "profit_factor": 1.58},
            "per_ticker": {},
        }, f)

    item = _make_actionable_scan_item(ticker="AMD", status="WATCHING")
    monkeypatch.setattr(
        command_handler,
        "run_full_day_option_scan",
        lambda cfg: {"timestamp": "2026-11-18 10:00:00 ET", "results": [item]},
    )
    monkeypatch.setattr(command_handler, "analyze_single_ticker", lambda tk, cfg: item)

    updates = [
        {"update_id": 101, "message": {"text": "/scan"}},
        {"update_id": 102, "message": {"text": "/fibo AMD"}},
        {"update_id": 103, "message": {"text": "/winrate"}},
        {"update_id": 104, "message": {"text": "/backtest"}},
        {"update_id": 105, "message": {"text": "/help"}},
    ]
    tg = telegram_mock(webhook_url="", updates_batches=[updates])

    has_wh = get_has_active_telegram_webhook()
    assert has_wh("bot_tok_f12_none") is False

    command_handler.poll_telegram_commands(sample_config)
    posts = [c for c in tg["calls"] if c["method"] == "POST"]
    assert len(posts) == 5
    assert open(command_handler.OFFSET_FILE, encoding="utf-8").read().strip() == "105"


def test_t1_f12_loop_deduplicates_actionable_alerts_and_sends_close_alerts(
    sample_config, monkeypatch, telegram_mock, freeze_ny_time
):
    clock = freeze_ny_time(datetime(2026, 11, 18, 9, 40))
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "bot_tok_f12_dedup")
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_CHAT_ID", "8964251574")
    tg = telegram_mock(webhook_url="https://script.google.com/macros/s/AKfycb/exec?market=DAY_OPTION")

    fake_ts = {"t": 1000.0}
    monkeypatch.setattr(bot_main.time, "time", lambda: fake_ts["t"])

    def _sleep(sec):
        fake_ts["t"] += sec
        clock["now"] += timedelta(seconds=sec)

    monkeypatch.setattr(bot_main.time, "sleep", _sleep)

    cycle = {"n": 0}

    def _scan(cfg):
        cycle["n"] += 1
        if cycle["n"] > 5:
            raise RuntimeError("Loop exceeded safety iterations")
        if cycle["n"] == 1:
            return {
                "timestamp": "2026-11-18 09:40:00 ET",
                "total_scanned": 1, "actionable_count": 1, "near_setup_count": 0, "blackout_count": 0,
                "results": [_make_actionable_scan_item(ticker="AMD", spot=105.20, status="ACTIONABLE")],
            }
        # Cycle 2: same ACTIONABLE key + spot hits TP2 (108.00)
        return {
            "timestamp": "2026-11-18 09:40:30 ET",
            "total_scanned": 1, "actionable_count": 1, "near_setup_count": 0, "blackout_count": 0,
            "results": [_make_actionable_scan_item(ticker="AMD", spot=108.00, status="ACTIONABLE")],
        }

    monkeypatch.setattr(bot_main, "run_full_day_option_scan", _scan)
    bot_main.run_continuous_30s_loop(
        sample_config, interval_sec=30, max_minutes=1, send_opening_report=True, ny_session_date="2026-11-18"
    )

    texts = [c["json"]["text"] for c in tg["calls"] if c["method"] == "POST"]
    signal_cards = [t for t in texts if t.startswith("🎫 <b>AMD")]
    close_alerts = [t for t in texts if "[PAPER TRADE CLOSED]" in t]
    assert len(signal_cards) == 1
    assert len(close_alerts) == 1


def test_t1_f12_loop_session_end_runs_final_mtm_flush_and_rebuilds_dashboard(
    sample_config, monkeypatch, telegram_mock, freeze_ny_time
):
    clock = freeze_ny_time(datetime(2026, 11, 18, 11, 29))
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "bot_tok_f12_flush")
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_CHAT_ID", "8964251574")
    tg = telegram_mock(webhook_url="https://script.google.com/macros/s/AKfycb/exec?market=DAY_OPTION")

    fake_ts = {"t": 1000.0}
    monkeypatch.setattr(bot_main.time, "time", lambda: fake_ts["t"])

    def _sleep(sec):
        fake_ts["t"] += sec
        clock["now"] += timedelta(seconds=sec)

    monkeypatch.setattr(bot_main.time, "sleep", _sleep)

    cycle = {"n": 0}

    def _scan(cfg):
        cycle["n"] += 1
        if cycle["n"] > 5:
            raise RuntimeError("Loop exceeded safety iterations")
        spot = 105.20 if cycle["n"] == 1 else 106.10
        st = "ACTIONABLE" if cycle["n"] == 1 else "WATCHING"
        return {
            "timestamp": "2026-11-18 11:29:00 ET",
            "total_scanned": 1, "actionable_count": 1 if st == "ACTIONABLE" else 0,
            "near_setup_count": 0, "blackout_count": 0,
            "results": [_make_actionable_scan_item(ticker="AMD", spot=spot, status=st)],
        }

    monkeypatch.setattr(bot_main, "run_full_day_option_scan", _scan)
    bot_main.run_continuous_30s_loop(
        sample_config, interval_sec=30, max_minutes=1, ny_session_date="2026-11-18"
    )

    state = paper_trader.load_paper_state(paper_trader.PAPER_STATE_PATH)
    assert state["open_positions"] == []
    assert len(state["closed_positions"]) == 1
    assert state["closed_positions"][0]["exit_reason"] == "2H_WINDOW_CLOSE"
    texts = [c["json"]["text"] for c in tg["calls"] if c["method"] == "POST"]
    assert any("[ปิดจบ 2 ชม. แรก]" in t for t in texts)
    assert os.path.exists(dashboard_builder.DASHBOARD_PATH)
    html_text = open(dashboard_builder.DASHBOARD_PATH, encoding="utf-8").read()
    assert "AMD" in html_text
    assert "100.0%" in html_text


# ===========================================================================
# F13: GitHub Actions Cloud Scheduler (5 tests)
# ===========================================================================

def test_t1_f13_workflow_yaml_valid_and_permissions_concurrency():
    wf = load_cloud_workflow()
    on_cfg = wf.get("on") or wf.get(True)
    assert wf["permissions"]["contents"] == "write"
    assert wf["concurrency"]["cancel-in-progress"] is False
    job = wf["jobs"]["run-day-option-bot"]
    assert int(job["timeout-minutes"]) >= 200
    mode_cfg = on_cfg["workflow_dispatch"]["inputs"]["mode"]
    assert mode_cfg["default"] == "scan_once"
    assert mode_cfg["options"] == ["scan_once", "live_session_2h", "backtest_90d"]


def test_t1_f13_workflow_dispatch_passes_mark_session_date_all_modes(tmp_path):
    for mode in ("scan_once", "live_session_2h", "backtest_90d"):
        sim = run_cloud_shell_simulation(
            tmp_path / mode,
            event_name="workflow_dispatch",
            inputs_mode=mode,
            ny_date="2026-11-18",
        )
        assert sim["returncode"] == 0
        assert len(sim["python_cmds"]) == 1
        assert cli_has_flag(sim["python_cmds"][0], "--mark-session-date", "2026-11-18")


def test_t1_f13_cron_schedule_covers_est_and_edt_prewarm():
    wf = load_cloud_workflow()
    on_cfg = wf.get("on") or wf.get(True)
    crons = [item["cron"] for item in on_cfg["schedule"]]
    assert len(crons) >= 3
    for c in crons:
        assert c.endswith("* * 1-5")
    # Every cron minute/hour in EDT (UTC-4) or EST (UTC-5) must fall >= 08:30 NY (or pre-warm covers it)
    utc_hours_mins = [(int(c.split()[1]), int(c.split()[0])) for c in crons]
    assert any(h == 13 for h, _ in utc_hours_mins)
    assert any(h == 14 for h, _ in utc_hours_mins)


def test_t1_f13_active_window_caps_rem_mins_to_1130_ny(tmp_path):
    # 11:22 NY = 682 mins -> REM_MINS = 690 - 682 = 8 (must NOT clamp upward to 15)
    sim = run_cloud_shell_simulation(
        tmp_path,
        event_name="schedule",
        ny_date="2026-11-18",
        ny_h=11,
        ny_m=22,
    )
    assert sim["returncode"] == 0
    assert len(sim["python_cmds"]) == 1
    assert cli_has_flag(sim["python_cmds"][0], "--max-minutes", "8")


def test_t1_f13_commit_step_stages_config_and_docs():
    wf = load_cloud_workflow()
    steps = wf["jobs"]["run-day-option-bot"]["steps"]
    commit_steps = [s for s in steps if "Commit" in str(s.get("name", ""))]
    assert len(commit_steps) == 1
    run_cmd = commit_steps[0]["run"]
    assert "git add config/ docs/" in run_cmd
    assert "git diff --staged --quiet" in run_cmd


# ===========================================================================
# F14: GAS Webhook & 09:35 NY Trigger (5 tests)
# ===========================================================================

def test_t1_f14_trigger_fires_workflow_dispatch_at_0935_ny_weekday():
    # Wednesday 2026-11-18 09:35:00 EST (-05:00)
    res = run_gas_function(
        "triggerDayOptionMarketOpen()",
        frozen_iso="2026-11-18T09:35:00-05:00",
        properties={"GH_TOKEN": "ghp_secret_abc"},
    )
    assert len(res["githubDispatches"]) == 1
    disp = res["githubDispatches"][0]
    assert "mjchatchai/Day_Option_Bot/actions/workflows/day_option_cloud.yml/dispatches" in disp["url"]
    assert disp["body"] == {"ref": "main", "inputs": {"mode": "live_session_2h"}}
    assert res["properties"]["LAST_DAY_OPTION_OPEN_DATE"] == "2026-11-18"


def test_t1_f14_webhook_scan_and_fibo_commands():
    # 1. /scan command
    upd_scan = {
        "update_id": 5001,
        "message": {"chat": {"id": 8964251574}, "text": "/scan"},
    }
    res_scan = run_gas_function(
        f"doPost({json.dumps({'parameter': {'market': 'DAY_OPTION'}, 'postData': {'contents': json.dumps(upd_scan)}})})"
    )
    assert len(res_scan["telegramSends"]) >= 1
    assert len(res_scan["githubDispatches"]) == 1
    assert res_scan["githubDispatches"][0]["body"]["inputs"]["mode"] == "scan_once"

    # 2. /fibo AMD command reading GitHub latest_scan.json
    scan_json = {"results": [_make_actionable_scan_item(ticker="AMD", spot=105.20)]}
    b64_scan = base64.b64encode(json.dumps(scan_json).encode("utf-8")).decode("ascii")
    upd_fibo = {
        "update_id": 5002,
        "message": {"chat": {"id": 8964251574}, "text": "/fibo AMD"},
    }
    res_fibo = run_gas_function(
        f"doPost({json.dumps({'parameter': {'market': 'DAY_OPTION'}, 'postData': {'contents': json.dumps(upd_fibo)}})})",
        mock_responses=[
            {"match": "config/latest_scan.json", "status": 200, "body": {"sha": "abc", "content": b64_scan}}
        ],
    )
    assert len(res_fibo["telegramSends"]) == 1
    card_txt = res_fibo["telegramSends"][0]["body"]["text"]
    assert "AMD" in card_txt and "TP1" in card_txt and "TP2" in card_txt and "SL" in card_txt


def test_t1_f14_webhook_winrate_and_backtest_commands():
    paper_json = {
        "open_positions": [],
        "closed_positions": [{"ticker": "AMD", "net_pnl_usd": 120.0}, {"ticker": "SPY", "net_pnl_usd": 80.0}],
    }
    b64_paper = base64.b64encode(json.dumps(paper_json).encode("utf-8")).decode("ascii")
    upd_wr = {"update_id": 6001, "message": {"chat": {"id": 8964251574}, "text": "/winrate"}}
    res_wr = run_gas_function(
        f"doPost({json.dumps({'parameter': {'market': 'DAY_OPTION'}, 'postData': {'contents': json.dumps(upd_wr)}})})",
        mock_responses=[
            {"match": "config/paper_trades.json", "status": 200, "body": {"sha": "p1", "content": b64_paper}}
        ],
    )
    wr_txt = res_wr["telegramSends"][0]["body"]["text"]
    assert "100.0%" in wr_txt and "2W / 0L" in wr_txt

    # /backtest with missing profit_factor tests fallback 1.58 (not 1.28)
    bt_json = {"overall_golden": {"win_rate_pct": 66.7, "wins": 80, "losses": 40, "net_pnl": 1480.05}}
    b64_bt = base64.b64encode(json.dumps(bt_json).encode("utf-8")).decode("ascii")
    upd_bt = {"update_id": 6002, "message": {"chat": {"id": 8964251574}, "text": "/backtest"}}
    res_bt = run_gas_function(
        f"doPost({json.dumps({'parameter': {'market': 'DAY_OPTION'}, 'postData': {'contents': json.dumps(upd_bt)}})})",
        mock_responses=[
            {"match": "config/backtest_90d.json", "status": 200, "body": {"sha": "b1", "content": b64_bt}}
        ],
    )
    bt_txt = res_bt["telegramSends"][0]["body"]["text"]
    assert "1.58" in bt_txt


def test_t1_f14_doget_day_option_health_non_secret_json():
    secret = "ghp_super_secret_do_not_leak_999"
    res = run_gas_function(
        "doGet({parameter: {action: 'day_option_health'}})",
        properties={"GH_TOKEN": secret, "LAST_DAY_OPTION_OPEN_DATE": "2026-11-18"},
        triggers=[{"handlerFunction": "triggerDayOptionMarketOpen", "everyMinutes": 5}],
    )
    assert res["result"]["mime"] == "JSON"
    raw = res["result"]["content"]
    assert secret not in raw
    parsed = json.loads(raw)
    assert parsed.get("ok") is True
    assert any(
        parsed.get(k) is True
        for k in ("has_gh_token", "gh_token_configured", "token_configured", "hasGhToken")
    )
    assert len(res["githubDispatches"]) == 0


def test_t1_f14_doget_setup_day_option_creates_5m_trigger():
    res = run_gas_function(
        "doGet({parameter: {action: 'setup_day_option'}})",
        triggers=[{"handlerFunction": "triggerDayOptionMarketOpen", "everyMinutes": 15}],
    )
    assert res["result"]["mime"] == "JSON"
    assert json.loads(res["result"]["content"])["ok"] is True
    day_trigs = [t for t in res["triggers"] if t["handlerFunction"] == "triggerDayOptionMarketOpen"]
    assert len(day_trigs) == 1
    assert day_trigs[0]["everyMinutes"] == 5
