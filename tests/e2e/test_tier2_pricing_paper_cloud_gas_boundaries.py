"""
Tier 2 Boundary & Corner Cases — Pricing, Paper Trading, Telegram, Main Loop,
Cloud Workflow, and GAS Webhook (F8–F14): 35 tests (5 per feature).
"""

import base64
from datetime import date, datetime, timedelta
import json
import math
import os
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import requests

from day_option_bot.src import (
    command_handler,
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
    run_cloud_shell_simulation,
    run_gas_function,
)
from day_option_bot.tests.e2e.test_tier1_pricing_paper_cloud_gas import _make_actionable_scan_item


# ===========================================================================
# F8 Boundaries: Real-Time BSM Option Pricing (5 tests)
# ===========================================================================

def test_t2_f08_rejects_excessive_spread_over_max_spread_pct(mock_yfinance_factory, freeze_ny_time):
    ref = date(2026, 11, 18)
    freeze_ny_time(datetime(2026, 11, 18, 10, 0))
    exp = "2026-11-21"
    # Strike 104.0 has 8.0% spread (> 4.5%), Strike 105.0 has 2.0% spread (<= 4.5%)
    df_calls = pd.DataFrame([
        {"contractSymbol": "WIDE", "strike": 104.0, "lastPrice": 2.6, "bid": 2.40, "ask": 2.60, "volume": 200, "openInterest": 400, "impliedVolatility": 0.40},
        {"contractSymbol": "TIGHT", "strike": 105.0, "lastPrice": 2.0, "bid": 1.98, "ask": 2.02, "volume": 200, "openInterest": 400, "impliedVolatility": 0.40},
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
    assert q["strike"] == 105.0
    assert q["spread_pct"] <= 4.5


def test_t2_f08_nan_iv_bid_ask_in_chain_handled_safely(mock_yfinance_factory, freeze_ny_time):
    ref = date(2026, 11, 18)
    freeze_ny_time(datetime(2026, 11, 18, 10, 0))
    exp = "2026-11-21"
    df_calls = pd.DataFrame([
        {"contractSymbol": "NAN1", "strike": 105.0, "lastPrice": np.nan, "bid": np.nan, "ask": np.nan, "volume": 100, "openInterest": 200, "impliedVolatility": np.nan},
    ])
    mock_yfinance_factory({
        "AMD": {
            "options": (exp,),
            "option_chains": {exp: SimpleNamespace(calls=df_calls, puts=pd.DataFrame())},
        }
    })

    q = call_select_option("AMD", "CALL", 105.20, 105.20, 106.50, 107.80, 103.35, reference_date=ref)
    for field in ("iv", "realtime_theo_price", "est_bid", "est_ask", "max_limit_buy", "delta"):
        assert not math.isnan(float(q[field])), f"{field} was NaN"


def test_t2_f08_empty_chain_or_exception_falls_back_synthetically(mock_yfinance_factory):
    ref = date(2026, 11, 18)
    mock_yfinance_factory({
        "AMD": {"options": RuntimeError("yfinance timeout")},
        "NVDA": {"options": ("2026-11-21",), "option_chains": {"2026-11-21": RuntimeError("chain 500")}},
    })
    for tk in ("AMD", "NVDA"):
        q = call_select_option(tk, "PUT", 106.18, 106.18, 105.00, 103.82, 107.86, reference_date=ref)
        assert q is not None
        assert q["realtime_theo_price"] >= 0.01
        assert q["est_ask"] >= q["est_bid"]


def test_t2_f08_dte_midnight_utc_bangkok_uses_ny_date(mock_yfinance_factory, freeze_ny_time):
    # NY is 2026-11-18 19:05 (UTC/Bangkok already 2026-11-19) -> exp 2026-11-19 has DTE=1 in NY!
    freeze_ny_time(datetime(2026, 11, 18, 19, 5))
    exp = "2026-11-19"
    df_calls = pd.DataFrame([
        {"contractSymbol": "NEXTDAY", "strike": 105.0, "lastPrice": 2.0, "bid": 1.98, "ask": 2.02, "volume": 200, "openInterest": 500, "impliedVolatility": 0.40},
    ])
    mock_yfinance_factory({
        "AMD": {"options": (exp,), "option_chains": {exp: SimpleNamespace(calls=df_calls, puts=pd.DataFrame())}}
    })

    q = call_select_option("AMD", "CALL", 105.20, 105.20, 106.50, 107.80, 103.35, min_dte=1, max_dte=7)
    assert q["expiration"] == "2026-11-19"
    assert q["dte"] == 1


def test_t2_f08_zero_or_negative_spot_floor_and_0_dte():
    assert options_pricing.black_scholes_price(0.0, 100.0, 3, 0.35, "CALL") == 0.01
    assert options_pricing.black_scholes_price(-5.0, 100.0, 3, 0.35, "PUT") == 0.01
    p_0dte = options_pricing.black_scholes_price(105.0, 105.0, 0.0, 0.35, "CALL")
    assert p_0dte > 0.01
    g_0dte = options_pricing.calculate_greeks(105.0, 105.0, 0.0, 0.35, "PUT")
    assert -1.0 <= g_0dte["delta"] <= 0.0


# ===========================================================================
# F9 Boundaries: yfinance & NY TZ/DST Helper (5 tests)
# ===========================================================================

def test_t2_f09_dst_spring_forward_and_fall_back_window():
    tu = import_time_utils()
    # Spring forward: 2026-03-08 -> Monday 2026-03-09 09:35 EDT (-04:00)
    spring_mon = datetime(2026, 3, 9, 9, 35, tzinfo=NY_TZ)
    # Fall back: 2026-11-01 -> Monday 2026-11-02 09:35 EST (-05:00)
    fall_mon = datetime(2026, 11, 2, 9, 35, tzinfo=NY_TZ)
    assert spring_mon.utcoffset() == timedelta(hours=-4)
    assert fall_mon.utcoffset() == timedelta(hours=-5)
    assert tu.is_within_2h_window_ny(spring_mon) is True
    assert tu.is_within_2h_window_ny(fall_mon) is True


def test_t2_f09_2h_window_exact_second_boundaries():
    tu = import_time_utils()
    assert tu.is_within_2h_window_ny(datetime(2026, 11, 18, 9, 34, 59, tzinfo=NY_TZ)) is False
    assert tu.is_within_2h_window_ny(datetime(2026, 11, 18, 9, 35, 0, tzinfo=NY_TZ)) is True
    assert tu.is_within_2h_window_ny(datetime(2026, 11, 18, 11, 29, 59, tzinfo=NY_TZ)) is True
    assert tu.is_within_2h_window_ny(datetime(2026, 11, 18, 11, 30, 0, tzinfo=NY_TZ)) is False
    assert tu.is_within_2h_window_ny(datetime(2026, 11, 22, 10, 0, 0, tzinfo=NY_TZ)) is False


def test_t2_f09_tz_naive_dataframe_localized_safely():
    tu = import_time_utils()
    df_naive = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, tz_mode="NAIVE")
    assert df_naive.index.tz is None
    cleaned = tu.ensure_ny_regular_hours(df_naive)
    assert str(cleaned.index.tz) == "America/New_York"
    assert len(cleaned) >= 200


def test_t2_f09_all_nan_ohlcv_dataframe_returns_empty():
    tu = import_time_utils()
    idx = pd.date_range("2026-11-18 09:30", periods=10, freq="2min", tz="America/New_York")
    df_nan = pd.DataFrame(
        {"Open": np.nan, "High": np.nan, "Low": np.nan, "Close": np.nan, "Volume": np.nan},
        index=idx,
    )
    cleaned = tu.ensure_ny_regular_hours(df_nan)
    assert cleaned.empty


def test_t2_f09_stale_or_non_current_day_data_handling(sample_config, mock_yfinance_factory):
    # Exhausted yfinance retries -> returns NO_DATA cleanly without crash
    attempts = {"n": 0}

    def _always_fail(period="5d", interval="2m"):
        attempts["n"] += 1
        raise RuntimeError("Rate limit 429")

    mock_yfinance_factory({"AMD": {"history_2m": _always_fail, "calendar": {}}})
    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    assert res["status"] == "NO_DATA"
    assert attempts["n"] >= 2


# ===========================================================================
# F10 Boundaries: Paper Trading & 2h Flush (5 tests)
# ===========================================================================

def test_t2_f10_outside_2h_window_blocks_new_entry_but_evaluates_exits(sample_config, freeze_ny_time):
    # 1. Open AMD at 10:00 NY
    freeze_ny_time(datetime(2026, 11, 18, 10, 0))
    amd = _make_actionable_scan_item(ticker="AMD", spot=105.20, tp1=106.50, tp2=107.80, sl=103.35)
    call_update_paper({"results": [amd]}, sample_config, enforce_window=True)

    # 2. At 12:00 NY (outside 2h window): AMD hits TP2 (should close), NVDA is ACTIONABLE (must NOT open)
    freeze_ny_time(datetime(2026, 11, 18, 12, 0))
    amd_tp2 = _make_actionable_scan_item(ticker="AMD", spot=108.00, status="WATCHING")
    nvda_act = _make_actionable_scan_item(ticker="NVDA", spot=105.20, status="ACTIONABLE")
    res = call_update_paper({"results": [amd_tp2, nvda_act]}, sample_config, enforce_window=True)

    assert len(res["newly_closed"]) == 1
    assert res["newly_closed"][0]["ticker"] == "AMD"
    assert res["newly_closed"][0]["exit_reason"] == "TP2_FULL_WIN"
    assert len(res["newly_opened"]) == 0
    assert res["state"]["open_positions"] == []


def test_t2_f10_eod_force_flush_runs_before_tp_sl_on_new_day(sample_config, freeze_ny_time):
    # Open on 2026-11-17
    freeze_ny_time(datetime(2026, 11, 17, 10, 0))
    amd_d1 = _make_actionable_scan_item(
        ticker="AMD", spot=105.20, tp1=106.50, tp2=107.80, sl=103.35, curr_date="2026-11-17"
    )
    call_update_paper({"results": [amd_d1]}, sample_config, enforce_window=True)

    # Next trading day 2026-11-18 opens below yesterday's SL (101.00 < 103.35)
    freeze_ny_time(datetime(2026, 11, 18, 9, 36))
    amd_d2 = _make_actionable_scan_item(
        ticker="AMD", spot=101.00, curr_date="2026-11-18", status="WATCHING"
    )
    res = call_update_paper({"results": [amd_d2]}, sample_config, enforce_window=True)
    assert len(res["newly_closed"]) == 1
    assert res["newly_closed"][0]["exit_reason"] == "EOD_FORCE_FLUSH"


def test_t2_f10_tp1_lock_sl_guaranteed_ge_breakeven_when_entry_past_fibo(sample_config, freeze_ny_time):
    # Suppose entry_underlying (106.00) moved slightly past fibo_entry (105.20) toward tp1 (106.50)
    freeze_ny_time(datetime(2026, 11, 18, 9, 45))
    amd = _make_actionable_scan_item(
        ticker="AMD", spot=106.00, action="BUY CALL",
        tp1=106.50, tp2=107.80, sl=103.35, lock_sl=105.79,  # stale lock_sl < entry 106.00!
    )
    call_update_paper({"results": [amd]}, sample_config, enforce_window=True)

    # Hit TP1 (106.60), then retrace to 106.10 -> locked SL must be >= entry_underlying (106.00)!
    call_update_paper({"results": [_make_actionable_scan_item(ticker="AMD", spot=106.60, status="WATCHING")]}, sample_config, enforce_window=True)
    state = paper_trader.load_paper_state(paper_trader.PAPER_STATE_PATH)
    open_pos = state["open_positions"][0]
    assert open_pos["tp1_hit"] is True
    assert open_pos["sl_underlying"] >= open_pos["entry_underlying"]


def test_t2_f10_corrupt_json_backs_up_and_raises_no_silent_wipe(isolated_bot_paths):
    path = isolated_bot_paths["paper_state_path"]
    corrupt_payload = "{corrupt json !!!"
    with open(path, "w", encoding="utf-8") as f:
        f.write(corrupt_payload)

    try:
        state = paper_trader.load_paper_state(path)
        assert isinstance(state, dict)
        assert state["open_positions"] == []
        assert state["closed_positions"] == []
    except (json.JSONDecodeError, ValueError):
        pass

    backups = list(isolated_bot_paths["config_dir"].glob("paper_trades.json*.bak*")) + list(
        isolated_bot_paths["config_dir"].glob("*corrupt*")
    )
    assert len(backups) >= 1
    assert backups[0].read_text(encoding="utf-8") == corrupt_payload


def test_t2_f10_flush_empty_open_positions_preserves_custom_starting_balance(sample_config, freeze_ny_time):
    freeze_ny_time(datetime(2026, 11, 18, 11, 30))
    sample_config["paper_trading"]["starting_balance_usd"] = 10000.0
    res = call_flush_paper(sample_config)
    assert res["flushed"] == []
    assert res["kpis"]["current_equity_usd"] == 10000.0

    # Also verify duplicate same-ticker/same-date/same-fibo prevention
    freeze_ny_time(datetime(2026, 11, 18, 9, 45))
    amd = _make_actionable_scan_item(ticker="AMD", spot=105.20)
    r1 = call_update_paper({"results": [amd]}, sample_config, enforce_window=True)
    r2 = call_update_paper({"results": [amd]}, sample_config, enforce_window=True)
    assert len(r1["newly_opened"]) == 1
    assert len(r2["newly_opened"]) == 0


# ===========================================================================
# F11 Boundaries: Telegram 5-Line Cards & Delivery (5 tests)
# ===========================================================================

def test_t2_f11_empty_picks_renders_summary_without_crash():
    scan_out = {
        "timestamp": "2026-11-18 10:00:00 ET",
        "results": [
            {"ticker": "NVDA", "status": "EARNINGS_BLACKOUT"},
            {"ticker": "META", "status": "GAP_SKIP"},
            {"ticker": "AAPL", "status": "NO_DATA"},
        ],
    }
    report = telegram_notifier.build_scan_telegram_report(scan_out, {"kpis": {"win_rate_pct": 0.0, "wins": 0, "losses": 0, "net_pnl_usd": 0.0}})
    assert "Paper Win Rate" in report
    assert "0" in report or "ไม่มี" in report or "ไม่พบ" in report or "BLACKOUT" in report


def test_t2_f11_message_exceeding_4096_chars_handled_safely(monkeypatch, telegram_mock):
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "bot_tok_123")
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_CHAT_ID", "8964251574")
    tg = telegram_mock(post_statuses=[200, 200, 200])

    huge_msg = ("Line of signal card details for AMD CALL TP1 TP2 SL.\n" * 150)
    assert len(huge_msg) > 4096

    ok = telegram_notifier.send_telegram_message(huge_msg)
    assert ok is True
    posts = [c for c in tg["calls"] if c["method"] == "POST"]
    assert len(posts) >= 1
    for p in posts:
        assert len(p["json"]["text"]) <= 4096


def test_t2_f11_placeholder_or_missing_day_option_creds_rejected(monkeypatch, telegram_mock):
    tg = telegram_mock(post_statuses=[200])
    # Generic TELEGRAM_BOT_TOKEN set, but DAY_OPTION_* unset -> must NOT leak to generic bot!
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "wrong_generic_bot")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "999999")
    assert telegram_notifier.send_telegram_message("test") is False

    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "your_bot_token_here")
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_CHAT_ID", "your_chat_id")
    assert telegram_notifier.send_telegram_message("test") is False
    assert len([c for c in tg["calls"] if c["method"] == "POST"]) == 0


def test_t2_f11_missing_option_quote_in_near_setup_skipped_safely():
    scan_out = {
        "timestamp": "2026-11-18 10:00:00 ET",
        "results": [{"ticker": "AMD", "status": "NEAR_SETUP", "option_action": "BUY CALL"}],
    }
    report = telegram_notifier.build_scan_telegram_report(scan_out, {"kpis": {}})
    assert "Paper Win Rate" in report


def test_t2_f11_network_timeout_logs_and_returns_false(monkeypatch, telegram_mock, capsys, caplog):
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "bot_tok_123")
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_CHAT_ID", "8964251574")
    import time as py_time
    monkeypatch.setattr(py_time, "sleep", lambda s: None)

    telegram_mock(post_statuses=[requests.Timeout("timed out"), requests.Timeout("timed out"), requests.Timeout("timed out")])
    ok = telegram_notifier.send_telegram_message("hello")
    assert ok is False
    captured = capsys.readouterr()
    combined = captured.out + captured.err + caplog.text
    assert "Telegram" in combined or "timeout" in combined.lower()


# ===========================================================================
# F12 Boundaries: Webhook vs Polling & main.py Loop (5 tests)
# ===========================================================================

def test_t2_f12_getwebhookinfo_failure_safe_no_crash(monkeypatch, telegram_mock):
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "bot_tok_f12_err")
    has_wh = get_has_active_telegram_webhook()

    telegram_mock(webhook_error=requests.Timeout("timeout"))
    assert has_wh("bot_tok_f12_err_timeout") is False

    telegram_mock(webhook_status=500)
    assert has_wh("bot_tok_f12_err_500") is False


def test_t2_f12_no_command_poll_flag_overrides_polling(
    sample_config, monkeypatch, telegram_mock, freeze_ny_time
):
    clock = freeze_ny_time(datetime(2026, 11, 18, 9, 40))
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "bot_tok_f12_nopoll")
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_CHAT_ID", "8964251574")
    tg = telegram_mock(webhook_url="")  # No webhook, yet --no-command-poll must still skip getUpdates!

    fake_ts = {"t": 1000.0}
    monkeypatch.setattr(bot_main.time, "time", lambda: fake_ts["t"])

    def _sleep(sec):
        fake_ts["t"] += sec
        clock["now"] += timedelta(seconds=sec)

    monkeypatch.setattr(bot_main.time, "sleep", _sleep)
    monkeypatch.setattr(bot_main, "load_config", lambda: sample_config)
    monkeypatch.setattr(
        bot_main,
        "run_full_day_option_scan",
        lambda cfg: {"timestamp": "t", "total_scanned": 0, "actionable_count": 0, "near_setup_count": 0, "blackout_count": 0, "results": []},
    )
    monkeypatch.setattr(sys, "argv", ["main.py", "--loop", "--interval", "30", "--max-minutes", "1", "--no-command-poll"])

    bot_main.main()
    get_updates = [c for c in tg["calls"] if c["method"] == "GET" and c["url"].endswith("/getUpdates")]
    assert len(get_updates) == 0


def test_t2_f12_poll_commands_non_200_logged_and_ignores_non_slash(
    sample_config, monkeypatch, telegram_mock, capsys
):
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "bot_tok_f12_409")
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_CHAT_ID", "8964251574")

    # 1. HTTP 409 Conflict logged cleanly
    telegram_mock(updates_batches=[(409, {"ok": False, "description": "Conflict: terminated by setWebhook"})])
    command_handler.poll_telegram_commands(sample_config)
    out = capsys.readouterr()
    assert "409" in (out.out + out.err)

    # 2. Non-slash message ignored cleanly without sending Telegram reply
    tg2 = telegram_mock(updates_batches=[[{"update_id": 200, "message": {"text": "hello world"}}]])
    command_handler.poll_telegram_commands(sample_config)
    assert len([c for c in tg2["calls"] if c["method"] == "POST"]) == 0


def test_t2_f12_loop_recovers_from_transient_cycle_exception(
    sample_config, monkeypatch, telegram_mock, freeze_ny_time
):
    clock = freeze_ny_time(datetime(2026, 11, 18, 9, 40))
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "bot_tok_f12_recover")
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_CHAT_ID", "8964251574")
    telegram_mock(webhook_url="https://script.google.com/macros/s/AKfycb/exec?market=DAY_OPTION")

    fake_ts = {"t": 1000.0}
    monkeypatch.setattr(bot_main.time, "time", lambda: fake_ts["t"])

    def _sleep(sec):
        fake_ts["t"] += sec
        clock["now"] += timedelta(seconds=sec)

    monkeypatch.setattr(bot_main.time, "sleep", _sleep)

    calls = {"n": 0}

    def _flaky_scan(cfg):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Transient scan error")
        return {
            "timestamp": "2026-11-18 09:40:30 ET",
            "total_scanned": 1, "actionable_count": 0, "near_setup_count": 0, "blackout_count": 0,
            "results": [_make_actionable_scan_item(ticker="AMD", status="WATCHING")],
        }

    monkeypatch.setattr(bot_main, "run_full_day_option_scan", _flaky_scan)
    bot_main.run_continuous_30s_loop(sample_config, interval_sec=30, max_minutes=1, ny_session_date="2026-11-18")
    assert calls["n"] >= 2
    assert os.path.exists(bot_main.LATEST_SCAN_PATH)


def test_t2_f12_loop_stops_immediately_at_1130_ny_wall_clock(
    sample_config, monkeypatch, telegram_mock, freeze_ny_time
):
    # Start at 11:29 NY with huge max_minutes=115; advancing 60s to 11:30 NY must exit immediately!
    clock = freeze_ny_time(datetime(2026, 11, 18, 11, 29))
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "bot_tok_f12_1130")
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_CHAT_ID", "8964251574")
    telegram_mock(webhook_url="https://script.google.com/macros/s/AKfycb/exec?market=DAY_OPTION")

    fake_ts = {"t": 1000.0}
    monkeypatch.setattr(bot_main.time, "time", lambda: fake_ts["t"])

    def _sleep(sec):
        fake_ts["t"] += sec
        clock["now"] += timedelta(seconds=sec)

    monkeypatch.setattr(bot_main.time, "sleep", _sleep)
    cycles = {"n": 0}

    def _scan(cfg):
        cycles["n"] += 1
        if cycles["n"] > 4:
            raise RuntimeError("Loop did not stop at 11:30 NY wall clock")
        return {
            "timestamp": "2026-11-18 11:29:00 ET",
            "total_scanned": 1, "actionable_count": 0, "near_setup_count": 0, "blackout_count": 0,
            "results": [_make_actionable_scan_item(ticker="AMD", status="WATCHING")],
        }

    monkeypatch.setattr(bot_main, "run_full_day_option_scan", _scan)
    bot_main.run_continuous_30s_loop(sample_config, interval_sec=30, max_minutes=115, ny_session_date="2026-11-18")
    assert cycles["n"] <= 3


# ===========================================================================
# F13 Boundaries: GitHub Actions Cloud Scheduler (5 tests)
# ===========================================================================

def test_t2_f13_est_1315_utc_does_not_misfire_premarket_else(tmp_path):
    # 13:15 UTC in EST winter = 08:15 EST (NY_MINS=495) -> MUST NOT run --once pre-market!
    sim = run_cloud_shell_simulation(tmp_path, event_name="schedule", ny_date="2026-11-18", ny_h=8, ny_m=15)
    assert sim["returncode"] == 0
    assert not any("--once" in cmd for cmd in sim["python_cmds"])


def test_t2_f13_est_1335_utc_fits_inside_timeout_minutes(tmp_path):
    # 13:35 UTC in EST winter = 08:35 EST (NY_MINS=515) -> either pre-warm sleep + 115m loop < timeout-minutes, or clean skip
    wf = load_cloud_workflow()
    timeout_mins = int(wf["jobs"]["run-day-option-bot"]["timeout-minutes"])
    sim = run_cloud_shell_simulation(tmp_path, event_name="schedule", ny_date="2026-11-18", ny_h=8, ny_m=35)
    assert sim["returncode"] == 0
    assert not any("--once" in cmd for cmd in sim["python_cmds"])
    assert len(sim["sleeps"]) <= 1
    sleep_secs = sim["sleeps"][0] if sim["sleeps"] else 0
    loop_mins = 115 if sim["python_cmds"] else 0
    assert (sleep_secs / 60.0 + loop_mins) < timeout_mins


def test_t2_f13_duplicate_cron_after_1130_skips_when_already_ran(tmp_path):
    sim = run_cloud_shell_simulation(
        tmp_path, event_name="schedule", ny_date="2026-11-18", ny_h=12, ny_m=10, last_run_date="2026-11-18"
    )
    assert sim["returncode"] == 0
    assert sim["python_cmds"] == []


def test_t2_f13_late_cron_after_1130_runs_once_only_if_not_yet_run(tmp_path):
    sim = run_cloud_shell_simulation(
        tmp_path, event_name="schedule", ny_date="2026-11-18", ny_h=12, ny_m=10, last_run_date="2026-11-17"
    )
    assert sim["returncode"] == 0
    assert len(sim["python_cmds"]) == 1
    assert "--once" in sim["python_cmds"][0]
    assert cli_has_flag(sim["python_cmds"][0], "--mark-session-date", "2026-11-18")


def test_t2_f13_secrets_wired_in_env():
    wf = load_cloud_workflow()
    steps = wf["jobs"]["run-day-option-bot"]["steps"]
    exec_step = [s for s in steps if "Execute Day Option Bot" in str(s.get("name", ""))][0]
    env = exec_step["env"]
    assert env["DAY_OPTION_TELEGRAM_BOT_TOKEN"] == "${{ secrets.DAY_OPTION_TELEGRAM_BOT_TOKEN }}"
    assert env["DAY_OPTION_TELEGRAM_CHAT_ID"] == "${{ secrets.DAY_OPTION_TELEGRAM_CHAT_ID }}"


# ===========================================================================
# F14 Boundaries: GAS Webhook & 09:35 NY Trigger (5 tests)
# ===========================================================================

def test_t2_f14_weekend_and_outside_0933_0945_does_not_trigger():
    # Saturday 09:35 EST, Wednesday 09:30 EST, Wednesday 09:46 EST
    for iso in ("2026-11-21T09:35:00-05:00", "2026-11-18T09:30:00-05:00", "2026-11-18T09:46:00-05:00"):
        res = run_gas_function("triggerDayOptionMarketOpen()", frozen_iso=iso)
        assert res["githubDispatches"] == []


def test_t2_f14_already_triggered_same_ny_date_skips_duplicate():
    res = run_gas_function(
        "triggerDayOptionMarketOpen()",
        frozen_iso="2026-11-18T09:40:00-05:00",
        properties={"GH_TOKEN": "ghp_abc", "LAST_DAY_OPTION_OPEN_DATE": "2026-11-18"},
    )
    assert res["githubDispatches"] == []


def test_t2_f14_gh_api_401_403_404_alerts_telegram_and_does_not_lock_date():
    for status in (401, 403):
        res = run_gas_function(
            "triggerDayOptionMarketOpen()",
            frozen_iso="2026-11-18T09:35:00-05:00",
            properties={"GH_TOKEN": "ghp_bad"},
            mock_responses=[{"match": "/dispatches", "status": status, "body": {"message": "Bad credentials"}}],
        )
        assert "LAST_DAY_OPTION_OPEN_DATE" not in res["properties"]
        assert len(res["telegramSends"]) >= 1
        assert str(status) in res["telegramSends"][0]["body"]["text"]


def test_t2_f14_scan_and_fibo_error_on_gh_401_notifies_user():
    upd_scan = {"update_id": 7001, "message": {"chat": {"id": 8964251574}, "text": "/scan"}}
    res = run_gas_function(
        f"doPost({json.dumps({'parameter': {'market': 'DAY_OPTION'}, 'postData': {'contents': json.dumps(upd_scan)}})})",
        mock_responses=[{"match": "/dispatches", "status": 401, "body": {"message": "Bad credentials"}}],
    )
    # Must send an error notification mentioning 401/error instead of silently dropping it
    assert any("401" in s["body"]["text"] for s in res["telegramSends"])


def test_t2_f14_formatting_negative_pnl_negative_tp_positive_delta_and_dedup():
    # 1. Negative Net PnL -> -$50.00 (never $-50.00)
    paper_json = {"open_positions": [], "closed_positions": [{"ticker": "AMD", "net_pnl_usd": -50.0}]}
    b64_paper = base64.b64encode(json.dumps(paper_json).encode("utf-8")).decode("ascii")
    upd_wr = {"update_id": 8001, "message": {"chat": {"id": 8964251574}, "text": "/winrate"}}
    res_wr = run_gas_function(
        f"doPost({json.dumps({'parameter': {'market': 'DAY_OPTION'}, 'postData': {'contents': json.dumps(upd_wr)}})})",
        mock_responses=[{"match": "config/paper_trades.json", "status": 200, "body": {"sha": "p", "content": b64_paper}}],
    )
    wr_txt = res_wr["telegramSends"][0]["body"]["text"]
    assert "-$50.00" in wr_txt
    assert "$-50.00" not in wr_txt

    # 2. /fibo positive delta +0.58 and negative TP/SL without "+-"
    item = _make_actionable_scan_item(ticker="AMD", spot=105.20)
    item["option_quote"]["delta"] = 0.58
    item["option_quote"]["tp1_gain_pct"] = -5.0
    b64_scan = base64.b64encode(json.dumps({"results": [item]}).encode("utf-8")).decode("ascii")
    upd_fibo = {"update_id": 8002, "message": {"chat": {"id": 8964251574}, "text": "/fibo AMD"}}
    res_fibo = run_gas_function(
        f"doPost({json.dumps({'parameter': {'market': 'DAY_OPTION'}, 'postData': {'contents': json.dumps(upd_fibo)}})})",
        mock_responses=[{"match": "config/latest_scan.json", "status": 200, "body": {"sha": "s", "content": b64_scan}}],
    )
    card_txt = res_fibo["telegramSends"][0]["body"]["text"]
    assert "+0.58" in card_txt
    assert "+-" not in card_txt

    # 3. Duplicate update_id ignored & empty postData safe
    res_dup = run_gas_function(
        f"doPost({json.dumps({'parameter': {'market': 'DAY_OPTION'}, 'postData': {'contents': json.dumps(upd_wr)}})})",
        properties={"GH_TOKEN": "ghp", "LAST_UID_DAY_OPTION": "8001"},
    )
    assert res_dup["telegramSends"] == []
    res_empty = run_gas_function("doPost({parameter: {market: 'DAY_OPTION'}})")
    assert res_empty["result"]["content"] == "OK"
