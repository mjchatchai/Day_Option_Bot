"""
Tier 4 Real-World 2-Hour NY Session Scenarios (7 scenarios).
End-to-end simulations from 09:35 NY -> 11:30 NY exercising scanner, earnings,
BSM pricing, paper trader, Telegram notifier, dashboard builder, cloud YAML, and GAS VM.
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
    paper_trader,
)
from day_option_bot.tests.e2e.conftest import (
    NY_TZ,
    call_update_paper,
    cli_has_flag,
    load_cloud_workflow,
    make_intraday_2m_df,
    make_option_chain_df,
    run_cloud_shell_simulation,
    run_gas_function,
)


def _wire_session_clock(monkeypatch, clock):
    fake_ts = {"t": 1000.0}
    monkeypatch.setattr(bot_main.time, "time", lambda: fake_ts["t"])

    def _sleep(sec):
        fake_ts["t"] += sec
        clock["now"] += timedelta(seconds=sec)

    monkeypatch.setattr(bot_main.time, "sleep", _sleep)
    return fake_ts


def test_t4_01_bullish_stock_tp1_lock_to_tp2_runner_session(
    sample_config, mock_yfinance_factory, monkeypatch, telegram_mock, freeze_ny_time
):
    clock = freeze_ny_time(datetime(2026, 11, 18, 9, 35))
    _wire_session_clock(monkeypatch, clock)
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "bot_tok_t4_01")
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_CHAT_ID", "8964251574")
    tg = telegram_mock(webhook_url="https://script.google.com/macros/s/AKfycb/exec?market=DAY_OPTION")

    sample_config["watchlist"] = {"etf_index": [], "golden_stocks": ["AMD"]}
    cycle = {"n": 0}

    def _amd_hist(period="5d", interval="2m"):
        idx = cycle["n"]
        if idx == 0:
            # 09:35 NY: watching away from 0.618 (105.20)
            return make_intraday_2m_df(trend="UPTREND", active_level=105.20, candle_mode="NO_TOUCH", spot_override=107.00)
        if idx == 1:
            # 10:05 NY: pullback to 0.618 with hammer + RVOL 1.35 -> ACTIONABLE
            return make_intraday_2m_df(trend="UPTREND", active_level=105.20, candle_mode="CONFIRMED_REVERSAL", rvol_target=1.35)
        if idx == 2:
            # 10:35 NY: hits TP1 (106.50)
            return make_intraday_2m_df(trend="UPTREND", active_level=105.20, candle_mode="NO_TOUCH", spot_override=106.55)
        # 11:05 NY: hits TP2 (107.80)
        return make_intraday_2m_df(trend="UPTREND", active_level=105.20, candle_mode="NO_TOUCH", spot_override=108.00)

    exp = "2026-11-21"
    chain = make_option_chain_df(spot=105.20, option_type="CALL", strikes=[103.0, 104.0, 105.0, 106.0], dte_days=3)
    mock_yfinance_factory({
        "AMD": {
            "history_2m": _amd_hist,
            "calendar": {},
            "options": (exp,),
            "option_chains": {exp: SimpleNamespace(calls=chain, puts=pd.DataFrame())},
        }
    })

    orig_scan = bot_main.run_full_day_option_scan

    def _counted_scan(cfg):
        if cycle["n"] > 6:
            raise RuntimeError("Session loop safety stop")
        out = orig_scan(cfg)
        cycle["n"] += 1
        return out

    monkeypatch.setattr(bot_main, "run_full_day_option_scan", _counted_scan)
    bot_main.run_continuous_30s_loop(
        sample_config, interval_sec=1800, max_minutes=115, send_opening_report=True, ny_session_date="2026-11-18"
    )

    state = paper_trader.load_paper_state(paper_trader.PAPER_STATE_PATH)
    assert state["open_positions"] == []
    assert len(state["closed_positions"]) == 1
    closed = state["closed_positions"][0]
    assert closed["exit_reason"] == "TP2_FULL_WIN"
    assert closed["net_pnl_usd"] > 0
    assert closed["gross_pnl_usd"] - closed["net_pnl_usd"] == pytest.approx(5.0 * closed["contracts"], abs=0.01)

    texts = [c["json"]["text"] for c in tg["calls"] if c["method"] == "POST"]
    assert len([t for t in texts if t.startswith("🎫 <b>AMD")]) == 1
    assert any("[PAPER TRADE CLOSED]" in t and "✅ WIN" in t for t in texts)
    html_text = open(dashboard_builder.DASHBOARD_PATH, encoding="utf-8").read()
    assert "100.0%" in html_text


def test_t4_02_bearish_etf_tp1_trailing_profit_lock_session(
    sample_config, mock_yfinance_factory, monkeypatch, telegram_mock, freeze_ny_time
):
    clock = freeze_ny_time(datetime(2026, 11, 18, 9, 45))
    _wire_session_clock(monkeypatch, clock)
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "bot_tok_t4_02")
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_CHAT_ID", "8964251574")
    telegram_mock(webhook_url="https://script.google.com/macros/s/AKfycb/exec?market=DAY_OPTION")

    sample_config["watchlist"] = {"etf_index": ["SPY"], "golden_stocks": []}
    cycle = {"n": 0}

    def _spy_hist(period="5d", interval="2m"):
        idx = cycle["n"]
        if idx == 0:
            # 09:45 NY: Downtrend bounce to 0.500 (105.00) with shooting star + RVOL 1.25
            return make_intraday_2m_df(trend="DOWNTREND", active_level=105.00, candle_mode="CONFIRMED_REVERSAL", rvol_target=1.25)
        if idx == 1:
            # 10:15 NY: drops to TP1 (0.382 = 103.82)
            return make_intraday_2m_df(trend="DOWNTREND", active_level=105.00, candle_mode="NO_TOUCH", spot_override=103.80)
        # 10:45 NY: bounces back up to lock_sl_after_tp1 (~104.47) -> TP1_TRAIL_WIN
        return make_intraday_2m_df(trend="DOWNTREND", active_level=105.00, candle_mode="NO_TOUCH", spot_override=104.60)

    exp = "2026-11-21"
    chain = make_option_chain_df(spot=105.00, option_type="PUT", strikes=[104.0, 105.0, 106.0], dte_days=3)
    mock_yfinance_factory({
        "SPY": {
            "history_2m": _spy_hist,
            "options": (exp,),
            "option_chains": {exp: SimpleNamespace(calls=pd.DataFrame(), puts=chain)},
        }
    })

    orig_scan = bot_main.run_full_day_option_scan

    def _counted_scan(cfg):
        if cycle["n"] > 6:
            raise RuntimeError("Session loop safety stop")
        out = orig_scan(cfg)
        cycle["n"] += 1
        return out

    monkeypatch.setattr(bot_main, "run_full_day_option_scan", _counted_scan)
    bot_main.run_continuous_30s_loop(
        sample_config, interval_sec=1800, max_minutes=105, ny_session_date="2026-11-18"
    )

    state = paper_trader.load_paper_state(paper_trader.PAPER_STATE_PATH)
    assert len(state["closed_positions"]) == 1
    closed = state["closed_positions"][0]
    assert closed["exit_reason"] == "TP1_TRAIL_WIN"
    assert closed["is_win"] is True
    assert closed["net_pnl_usd"] > 0
    assert closed["gross_pnl_usd"] - closed["net_pnl_usd"] == pytest.approx(3.0 * closed["contracts"], abs=0.01)
    kpis = paper_trader.compute_paper_kpis(state["closed_positions"], [])
    assert kpis["fibo_0500"]["win_rate_pct"] == 100.0


def test_t4_03_multi_ticker_mixed_filters_sl_and_1130_flush(
    sample_config, mock_yfinance_factory, monkeypatch, telegram_mock, freeze_ny_time
):
    clock = freeze_ny_time(datetime(2026, 11, 18, 9, 35))
    _wire_session_clock(monkeypatch, clock)
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "bot_tok_t4_03")
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_CHAT_ID", "8964251574")
    tg = telegram_mock(webhook_url="https://script.google.com/macros/s/AKfycb/exec?market=DAY_OPTION")

    cycle = {"n": 0}

    def _spy_hist(period="5d", interval="2m"):
        if cycle["n"] == 0:
            # Secondary 0.618 allowed with RVOL 1.65 >= 1.50
            return make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="CONFIRMED_REVERSAL", rvol_target=1.65)
        # Stays mid-trade between entry (106.18) and TP1 (105.00), SL (107.86) until 11:30 flush
        return make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="NO_TOUCH", spot_override=105.60)

    def _amzn_hist(period="5d", interval="2m"):
        if cycle["n"] == 0:
            return make_intraday_2m_df(trend="UPTREND", active_level=105.20, candle_mode="CONFIRMED_REVERSAL", rvol_target=1.30)
        # Drops past SL (103.35)
        return make_intraday_2m_df(trend="UPTREND", active_level=105.20, candle_mode="NO_TOUCH", spot_override=103.00)

    mock_yfinance_factory({
        "SPY": {"history_2m": _spy_hist, "options": ()},
        "QQQ": {"history_2m": make_intraday_2m_df(trend="DOWNTREND", active_level=105.00, candle_mode="NO_TOUCH", spot_override=102.00)},
        "AMD": {"history_2m": make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="NO_TOUCH", spot_override=102.00), "calendar": {}},
        "PLTR": {"history_2m": make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="NO_TOUCH", spot_override=102.00), "calendar": {}},
        "META": {"history_2m": make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="NO_TOUCH", morning_gap_pct=3.2, spot_override=101.00), "calendar": {}},
        "NVDA": {"history_2m": make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="CONFIRMED_REVERSAL", rvol_target=1.40), "calendar": {"Earnings Date": [datetime(2026, 11, 18)]}},
        "AAPL": {"history_2m": make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="CONFIRMED_REVERSAL", rvol_target=0.85), "calendar": {}},
        "AMZN": {"history_2m": _amzn_hist, "calendar": {}, "options": ()},
    })

    orig_scan = bot_main.run_full_day_option_scan

    def _counted_scan(cfg):
        if cycle["n"] > 6:
            raise RuntimeError("Session loop safety stop")
        out = orig_scan(cfg)
        cycle["n"] += 1
        return out

    monkeypatch.setattr(bot_main, "run_full_day_option_scan", _counted_scan)
    bot_main.run_continuous_30s_loop(
        sample_config, interval_sec=3600, max_minutes=115, ny_session_date="2026-11-18"
    )

    state = paper_trader.load_paper_state(paper_trader.PAPER_STATE_PATH)
    assert state["open_positions"] == []
    by_tk = {c["ticker"]: c for c in state["closed_positions"]}
    assert "NVDA" not in by_tk and "META" not in by_tk and "AAPL" not in by_tk
    assert by_tk["AMZN"]["exit_reason"] == "SL_LOSS"
    assert by_tk["SPY"]["exit_reason"] == "2H_WINDOW_CLOSE"
    assert by_tk["SPY"]["fibo_level"] == "0.618"

    texts = [c["json"]["text"] for c in tg["calls"] if c["method"] == "POST"]
    assert any("[ปิดจบ 2 ชม. แรก]" in t for t in texts)
    assert os.path.exists(dashboard_builder.DASHBOARD_PATH)


def test_t4_04_dst_transition_utc_midnight_skew_and_overnight_flush(
    sample_config, mock_yfinance_factory, freeze_ny_time, tmp_path
):
    # Prior trading day 2026-11-06 left 1 stale open position; session runs Monday 2026-11-09 (EST)
    freeze_ny_time(datetime(2026, 11, 6, 10, 0))
    d1_df = make_intraday_2m_df(prev_date="2026-11-05", curr_date="2026-11-06", trend="UPTREND", active_level=105.20, rvol_target=1.35)
    mock_yfinance_factory({"AMD": {"history_2m": d1_df, "calendar": {}, "options": ()}})
    sample_config["watchlist"] = {"etf_index": [], "golden_stocks": ["AMD"]}
    scan_d1 = fibo_scanner.run_full_day_option_scan(sample_config)
    call_update_paper(scan_d1, sample_config, enforce_window=True)

    # Monday 2026-11-09 09:40 EST (post-DST fallback) with gap across SL and UTC/TH midnight exp check
    freeze_ny_time(datetime(2026, 11, 9, 9, 40))
    d2_df = make_intraday_2m_df(
        prev_date="2026-11-06", curr_date="2026-11-09", trend="DOWNTREND",
        active_level=106.18, candle_mode="NO_TOUCH", spot_override=101.00,
    )
    exp = "2026-11-10"
    chain = make_option_chain_df(spot=101.00, option_type="PUT", strikes=[100.0, 101.0, 102.0], dte_days=1)
    mock_yfinance_factory({
        "AMD": {
            "history_2m": d2_df,
            "calendar": {"Earnings Date": [datetime(2026, 11, 12)]},  # 3 days from 2026-11-09 -> safe
            "options": (exp,),
            "option_chains": {exp: SimpleNamespace(calls=pd.DataFrame(), puts=chain)},
        }
    })
    scan_d2 = fibo_scanner.run_full_day_option_scan(sample_config)
    assert scan_d2["results"][0]["earnings"]["days_until_earnings"] == 3
    assert scan_d2["results"][0]["option_quote"]["dte"] == 1

    paper_d2 = call_update_paper(scan_d2, sample_config, enforce_window=True)
    assert len(paper_d2["newly_closed"]) == 1
    assert paper_d2["newly_closed"][0]["exit_reason"] == "EOD_FORCE_FLUSH"

    # Cloud workflow dedup check for ny_session_date == "2026-11-09"
    sim = run_cloud_shell_simulation(
        tmp_path, event_name="schedule", ny_date="2026-11-09", ny_h=12, ny_m=5, last_run_date="2026-11-09"
    )
    assert sim["python_cmds"] == []


def test_t4_05_cloud_webhook_resilience_and_gas_parity_session(
    sample_config, mock_yfinance_factory, monkeypatch, telegram_mock, freeze_ny_time
):
    clock = freeze_ny_time(datetime(2026, 11, 18, 9, 35))
    _wire_session_clock(monkeypatch, clock)
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "bot_tok_t4_05")
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_CHAT_ID", "8964251574")
    # Simulate 429 on first Telegram POST, then 200s
    tg = telegram_mock(
        webhook_url="https://script.google.com/macros/s/AKfycb/exec?market=DAY_OPTION",
        post_statuses=[429, 200, 200, 200, 200],
    )

    sample_config["watchlist"] = {"etf_index": [], "golden_stocks": ["AMD"]}
    attempts = {"n": 0}

    def _flaky_amd(period="5d", interval="2m"):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("429 yf rate limit")
        return make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="NO_TOUCH", spot_override=102.00)

    mock_yfinance_factory({"AMD": {"history_2m": _flaky_amd, "calendar": {}, "options": ()}})
    bot_main.run_continuous_30s_loop(
        sample_config, interval_sec=60, max_minutes=1, send_opening_report=True, ny_session_date="2026-11-18"
    )

    assert len([c for c in tg["calls"] if c["method"] == "GET" and c["url"].endswith("/getUpdates")]) == 0
    assert os.path.exists(bot_main.LATEST_SCAN_PATH)
    scan_json = json.loads(open(bot_main.LATEST_SCAN_PATH, encoding="utf-8").read())
    paper_json = json.loads(open(paper_trader.PAPER_STATE_PATH, encoding="utf-8").read())

    # Feed session JSON outputs into GAS /fibo and /winrate via Node VM
    b64_scan = base64.b64encode(json.dumps(scan_json).encode("utf-8")).decode("ascii")
    upd_fibo = {"update_id": 9901, "message": {"chat": {"id": 8964251574}, "text": "/fibo AMD"}}
    res_fibo = run_gas_function(
        f"doPost({json.dumps({'parameter': {'market': 'DAY_OPTION'}, 'postData': {'contents': json.dumps(upd_fibo)}})})",
        mock_responses=[{"match": "config/latest_scan.json", "status": 200, "body": {"sha": "s1", "content": b64_scan}}],
    )
    assert "AMD" in res_fibo["telegramSends"][0]["body"]["text"]

    b64_paper = base64.b64encode(json.dumps(paper_json).encode("utf-8")).decode("ascii")
    upd_wr = {"update_id": 9902, "message": {"chat": {"id": 8964251574}, "text": "/winrate"}}
    res_wr = run_gas_function(
        f"doPost({json.dumps({'parameter': {'market': 'DAY_OPTION'}, 'postData': {'contents': json.dumps(upd_wr)}})})",
        mock_responses=[{"match": "config/paper_trades.json", "status": 200, "body": {"sha": "p1", "content": b64_paper}}],
    )
    assert "0.0%" in res_wr["telegramSends"][0]["body"]["text"]


def test_t4_06_corrupt_state_protection_and_post_1130_entry_block(
    sample_config, mock_yfinance_factory, monkeypatch, telegram_mock, isolated_bot_paths, freeze_ny_time
):
    # 1. Corrupt paper_trades.json must back up evidence safely without losing data
    corrupt_raw = "{corrupt"
    with open(isolated_bot_paths["paper_state_path"], "w", encoding="utf-8") as f:
        f.write(corrupt_raw)
    try:
        state = paper_trader.load_paper_state(isolated_bot_paths["paper_state_path"])
        assert isinstance(state, dict)
        assert state["open_positions"] == []
        assert state["closed_positions"] == []
    except (json.JSONDecodeError, ValueError):
        pass
    backups = list(isolated_bot_paths["config_dir"].glob("paper_trades.json*.bak*")) + list(
        isolated_bot_paths["config_dir"].glob("*corrupt*")
    )
    assert len(backups) >= 1
    assert backups[0].read_text(encoding="utf-8") == corrupt_raw

    # Clean file for morning session
    if os.path.exists(isolated_bot_paths["paper_state_path"]):
        os.remove(isolated_bot_paths["paper_state_path"])

    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "bot_tok_t4_06")
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_CHAT_ID", "8964251574")
    tg = telegram_mock(post_statuses=[200, 200])

    sample_config["watchlist"] = {"etf_index": [], "golden_stocks": ["AMD"]}
    df = make_intraday_2m_df(trend="UPTREND", active_level=105.20, candle_mode="CONFIRMED_REVERSAL", rvol_target=1.35)
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}, "options": ()}})

    # 2. Outside 2h window at 13:00 NY -> scan & Telegram report execute, but 0 new paper positions open!
    freeze_ny_time(datetime(2026, 11, 18, 13, 0))
    out = bot_main.execute_single_cycle(sample_config, notify_telegram=True, ny_session_date="2026-11-18")
    assert out["scan"]["actionable_count"] == 1
    assert out["paper"]["newly_opened"] == []
    assert out["paper"]["state"]["open_positions"] == []
    assert len([c for c in tg["calls"] if c["method"] == "POST"]) == 1


def test_t4_07_gas_trigger_to_cloud_dispatch_to_telegram_e2e_session(
    sample_config, mock_yfinance_factory, monkeypatch, telegram_mock, isolated_bot_paths, freeze_ny_time, tmp_path
):
    # 1. GAS triggerDayOptionMarketOpen fires workflow_dispatch at 09:35 NY (EDT & EST)
    for iso in ("2026-07-15T09:35:00-04:00", "2026-11-18T09:35:00-05:00"):
        gas_res = run_gas_function("triggerDayOptionMarketOpen()", frozen_iso=iso)
        assert len(gas_res["githubDispatches"]) == 1
        assert gas_res["githubDispatches"][0]["body"] == {"ref": "main", "inputs": {"mode": "live_session_2h"}}

    # 2. Cloud workflow executes workflow_dispatch live_session_2h with --mark-session-date
    sim = run_cloud_shell_simulation(
        tmp_path, event_name="workflow_dispatch", inputs_mode="live_session_2h", ny_date="2026-11-18"
    )
    assert sim["returncode"] == 0
    assert cli_has_flag(sim["python_cmds"][0], "--mark-session-date", "2026-11-18")

    # 3. main.py session runs & generates dashboard + backtest PF 1.58 parity in GAS /backtest
    clock = freeze_ny_time(datetime(2026, 11, 18, 9, 35))
    _wire_session_clock(monkeypatch, clock)
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "bot_tok_t4_07")
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_CHAT_ID", "8964251574")
    tg = telegram_mock(webhook_url="https://script.google.com/macros/s/AKfycb/exec?market=DAY_OPTION")

    bt_data = {
        "overall_original": {"win_rate_pct": 55.0, "net_pnl": 400.0},
        "overall_golden": {"win_rate_pct": 66.7, "wins": 80, "losses": 40, "net_pnl": 1480.05, "profit_factor": 1.58},
        "per_ticker": {
            "AMD": {
                "days_tested": 60,
                "bars_tested": 8190,
                "golden": {"win_rate_pct": 70.0, "wins": 14, "losses": 6, "net_pnl": 1480.05, "profit_factor": 1.58},
                "upgraded": {"win_rate_pct": 60.0, "net_pnl": 500.0},
            }
        },
    }
    with open(isolated_bot_paths["backtest_path"], "w", encoding="utf-8") as f:
        json.dump(bt_data, f)

    sample_config["watchlist"] = {"etf_index": [], "golden_stocks": ["AMD"]}
    mock_yfinance_factory({
        "AMD": {"history_2m": make_intraday_2m_df(trend="UPTREND", active_level=105.20, candle_mode="CONFIRMED_REVERSAL", rvol_target=1.35), "calendar": {}, "options": ()}
    })
    bot_main.run_continuous_30s_loop(
        sample_config, interval_sec=1800, max_minutes=115, send_opening_report=True, ny_session_date="2026-11-18"
    )

    assert os.path.exists(dashboard_builder.DASHBOARD_PATH)
    assert any("1,480.05" in open(dashboard_builder.DASHBOARD_PATH, encoding="utf-8").read() for _ in [1])
    assert len([c for c in tg["calls"] if c["method"] == "POST"]) >= 2

    b64_bt = base64.b64encode(json.dumps(bt_data).encode("utf-8")).decode("ascii")
    upd_bt = {"update_id": 9950, "message": {"chat": {"id": 8964251574}, "text": "/backtest"}}
    gas_bt = run_gas_function(
        f"doPost({json.dumps({'parameter': {'market': 'DAY_OPTION'}, 'postData': {'contents': json.dumps(upd_bt)}})})",
        mock_responses=[{"match": "config/backtest_90d.json", "status": 200, "body": {"sha": "b1", "content": b64_bt}}],
    )
    assert "1.58" in gas_bt["telegramSends"][0]["body"]["text"]
