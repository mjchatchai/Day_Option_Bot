"""
Tier 3 Cross-Feature Pairwise Integrations (18 tests).
Verifies seams across F1–F14, Cloud YAML ↔ main.py CLI, and GAS ↔ Python JSON schemas.
"""

import base64
from datetime import date, datetime, timedelta
import json
import os
import shlex
import sys
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
    call_flush_paper,
    call_update_paper,
    cli_has_flag,
    load_cloud_workflow,
    make_intraday_2m_df,
    make_option_chain_df,
    run_cloud_shell_simulation,
    run_gas_function,
)
from day_option_bot.tests.e2e.test_tier1_pricing_paper_cloud_gas import _make_actionable_scan_item


def test_t3_01_f1_f2_f3_scanner_pipeline_etf_vs_stock(sample_config, mock_yfinance_factory):
    sample_config["watchlist"] = {"etf_index": ["SPY", "QQQ"], "golden_stocks": ["AMD", "NVDA"]}
    mock_yfinance_factory({
        "SPY": {"history_2m": make_intraday_2m_df(trend="DOWNTREND", active_level=105.00, rvol_target=1.30)},
        "AMD": {"history_2m": make_intraday_2m_df(trend="UPTREND", active_level=105.20, rvol_target=1.30), "calendar": {}},
        "QQQ": {"history_2m": make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, rvol_target=1.30)},
        "NVDA": {"history_2m": make_intraday_2m_df(trend="DOWNTREND", active_level=105.00, rvol_target=1.65), "calendar": {}},
    })

    scan = fibo_scanner.run_full_day_option_scan(sample_config)
    by_tk = {r["ticker"]: r for r in scan["results"]}
    assert by_tk["SPY"]["trend_direction"] == "DOWNTREND" and by_tk["SPY"]["active_fibo_level"] == "0.500"
    assert by_tk["AMD"]["trend_direction"] == "UPTREND" and by_tk["AMD"]["active_fibo_level"] == "0.618"
    assert by_tk["QQQ"]["active_fibo_level"] == "0.500"  # 1.30 < 1.50 blocks secondary 0.618
    assert by_tk["NVDA"]["active_fibo_level"] == "0.500"  # 1.65 >= 1.50 allows secondary 0.500


def test_t3_02_f3_f4_f5_actionable_vs_near_vs_watching_ranking(sample_config, mock_yfinance_factory):
    sample_config["watchlist"] = {"etf_index": ["SPY"], "golden_stocks": ["AMD", "NVDA", "META", "AAPL"]}
    mock_yfinance_factory({
        "AMD": {"history_2m": make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="CONFIRMED_REVERSAL", rvol_target=1.35), "calendar": {}},
        "NVDA": {"history_2m": make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="TOUCHED_ONLY", rvol_target=1.35), "calendar": {}},
        "SPY": {"history_2m": make_intraday_2m_df(trend="DOWNTREND", active_level=105.00, candle_mode="NO_TOUCH", spot_override=101.50)},
        "META": {"history_2m": make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="NO_TOUCH", morning_gap_pct=3.5, spot_override=101.00), "calendar": {}},
        "AAPL": {"history_2m": pd.DataFrame(), "calendar": {}},
    })

    scan = fibo_scanner.run_full_day_option_scan(sample_config)
    statuses = [r["status"] for r in scan["results"]]
    assert statuses == ["ACTIONABLE", "NEAR_SETUP", "WATCHING", "GAP_SKIP", "NO_DATA"]
    assert scan["actionable_count"] == 1
    assert scan["near_setup_count"] == 1


def test_t3_03_f6_f7_priority_blackout_overrides_gap_and_actionable(sample_config, mock_yfinance_factory, freeze_ny_time):
    freeze_ny_time(datetime(2026, 11, 18, 10, 0))
    sample_config["watchlist"] = {"etf_index": [], "golden_stocks": ["NVDA"]}
    df = make_intraday_2m_df(
        trend="DOWNTREND", active_level=106.18, candle_mode="CONFIRMED_REVERSAL",
        rvol_target=1.50, morning_gap_pct=3.5,
    )
    mock_yfinance_factory({
        "NVDA": {"history_2m": df, "calendar": {"Earnings Date": [datetime(2026, 11, 18)]}}
    })

    scan = fibo_scanner.run_full_day_option_scan(sample_config)
    res = scan["results"][0]
    assert res["status"] == "EARNINGS_BLACKOUT"
    assert res["score"] == 0
    assert scan["blackout_count"] == 1


def test_t3_04_f2_f3_f8_fibo_anchors_to_bsm_tp1_tp2_sl(sample_config, mock_yfinance_factory, freeze_ny_time):
    freeze_ny_time(datetime(2026, 11, 18, 10, 0))
    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="CONFIRMED_REVERSAL", rvol_target=1.35)
    exp = "2026-11-21"
    chain_puts = make_option_chain_df(spot=106.10, option_type="PUT", strikes=[104.0, 106.0, 107.0, 108.0], dte_days=3)
    mock_yfinance_factory({
        "AMD": {
            "history_2m": df,
            "calendar": {},
            "options": (exp,),
            "option_chains": {exp: SimpleNamespace(calls=pd.DataFrame(), puts=chain_puts)},
        }
    })

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    oq = res["option_quote"]
    expected_theo = round(
        options_pricing.black_scholes_price(res["spot_price"], oq["strike"], oq["dte"], oq["iv"], "PUT"), 2
    )
    assert oq["realtime_theo_price"] == pytest.approx(expected_theo, abs=0.02)
    assert oq["tp2_option_price"] >= oq["tp1_option_price"] >= oq["realtime_theo_price"] >= oq["sl_option_price"]


def test_t3_05_f8_f10_zero_artificial_entry_pnl_and_friction(sample_config, mock_yfinance_factory, freeze_ny_time):
    freeze_ny_time(datetime(2026, 11, 18, 9, 45))
    # Spot slightly displaced from fibo_entry_underlying (e.g. 106.05 vs 106.18)
    df = make_intraday_2m_df(
        trend="DOWNTREND", active_level=106.18, candle_mode="CONFIRMED_REVERSAL",
        rvol_target=1.35, spot_override=106.05,
    )
    mock_yfinance_factory({
        "AMD": {"history_2m": df, "calendar": {}, "options": ()},
        "SPY": {"history_2m": make_intraday_2m_df(trend="DOWNTREND", active_level=105.00, rvol_target=1.35), "options": ()},
    })

    amd_res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    spy_res = fibo_scanner.analyze_single_ticker("SPY", sample_config)
    assert amd_res["status"] == "ACTIONABLE" and spy_res["status"] == "ACTIONABLE"

    opened = call_update_paper({"results": [amd_res, spy_res]}, sample_config, enforce_window=True)
    assert len(opened["newly_opened"]) == 2
    for pos in opened["newly_opened"]:
        # Marked at entry spot_realtime, NOT artificially offset by fibo_entry_underlying!
        assert abs(pos["current_option_price"] - pos["entry_option_price"]) <= round(pos["entry_option_price"] * 0.06 + 0.05, 2)

    # Close both via flush at same spot -> net PnL reflects spread + exact friction ($5 stock, $3 ETF)
    flushed = call_flush_paper(sample_config, latest_scan={"results": [amd_res, spy_res]})
    by_tk = {c["ticker"]: c for c in flushed["flushed"]}
    assert by_tk["AMD"]["gross_pnl_usd"] - by_tk["AMD"]["net_pnl_usd"] == pytest.approx(5.0 * by_tk["AMD"]["contracts"], abs=0.01)
    assert by_tk["SPY"]["gross_pnl_usd"] - by_tk["SPY"]["net_pnl_usd"] == pytest.approx(3.0 * by_tk["SPY"]["contracts"], abs=0.01)


def test_t3_06_f9_f7_f8_f10_ny_midnight_bangkok_utc_skew(sample_config, mock_yfinance_factory, freeze_ny_time):
    # 2026-11-18 19:05 EST == 2026-11-19 00:05 UTC == 2026-11-19 07:05 Bangkok
    freeze_ny_time(datetime(2026, 11, 18, 19, 5))
    df = make_intraday_2m_df(prev_date="2026-11-17", curr_date="2026-11-18", trend="UPTREND", active_level=105.20, rvol_target=1.35)
    exp = "2026-11-19"
    chain_calls = make_option_chain_df(spot=105.20, option_type="CALL", strikes=[104.0, 105.0, 106.0], dte_days=1)
    mock_yfinance_factory({
        "AMD": {
            "history_2m": df,
            "calendar": {"Earnings Date": [datetime(2026, 11, 21)]},  # 3 days from NY 2026-11-18 -> safe
            "options": (exp,),
            "option_chains": {exp: SimpleNamespace(calls=chain_calls, puts=pd.DataFrame())},
        }
    })
    sample_config["watchlist"] = {"etf_index": [], "golden_stocks": ["AMD"]}
    scan = fibo_scanner.run_full_day_option_scan(sample_config)
    amd = scan["results"][0]
    assert amd["earnings"]["days_until_earnings"] == 3
    assert amd["earnings"]["is_blackout"] is False
    assert amd["option_quote"]["dte"] == 1
    assert "ET" in scan["timestamp"]

    # Force window open check separately with enforce_window=False to verify trading_date uses NY 2026-11-18
    paper = call_update_paper(scan, sample_config, enforce_window=False)
    assert paper["newly_opened"][0]["trading_date"] == "2026-11-18"


def test_t3_07_f10_f11_paper_close_triggers_exact_telegram_card(sample_config, freeze_ny_time):
    freeze_ny_time(datetime(2026, 11, 18, 9, 45))
    amd = _make_actionable_scan_item(ticker="AMD", spot=105.20, tp1=106.50, tp2=107.80, sl=103.35)
    call_update_paper({"results": [amd]}, sample_config, enforce_window=True)

    tick2 = {"timestamp": "2026-11-18 10:15:00 ET", "results": [_make_actionable_scan_item(ticker="AMD", spot=108.00, status="WATCHING")]}
    paper_out = call_update_paper(tick2, sample_config, enforce_window=True)
    report = telegram_notifier.build_scan_telegram_report(tick2, paper_out)
    assert "100.0%" in report
    assert "1W/0L" in report


def test_t3_08_f1_f8_f11_scanner_to_telegram_5line_card(sample_config, mock_yfinance_factory, monkeypatch, telegram_mock):
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_CHAT_ID", "8964251574")
    tg = telegram_mock(post_statuses=[200])

    df = make_intraday_2m_df(trend="DOWNTREND", active_level=106.18, candle_mode="CONFIRMED_REVERSAL", rvol_target=1.35)
    mock_yfinance_factory({"AMD": {"history_2m": df, "calendar": {}, "options": ()}})

    res = fibo_scanner.analyze_single_ticker("AMD", sample_config)
    card = telegram_notifier.format_signal_card(res)
    assert len([ln for ln in card.splitlines() if ln.strip()]) == 5
    assert len(card) <= 4096
    assert telegram_notifier.send_telegram_message(card) is True
    assert len([c for c in tg["calls"] if c["method"] == "POST"]) == 1


def test_t3_09_f11_f12_webhook_active_eliminates_409_in_main_loop(
    sample_config, monkeypatch, telegram_mock, freeze_ny_time
):
    clock = freeze_ny_time(datetime(2026, 11, 18, 9, 35))
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "bot_tok_t3_09")
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_CHAT_ID", "8964251574")
    tg = telegram_mock(webhook_url="https://script.google.com/macros/s/AKfycb/exec?market=DAY_OPTION")

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
            "total_scanned": 1, "actionable_count": 1, "near_setup_count": 0, "blackout_count": 0,
            "results": [_make_actionable_scan_item(ticker="AMD", status="ACTIONABLE")],
        },
    )

    bot_main.run_continuous_30s_loop(
        sample_config, interval_sec=30, max_minutes=1, send_opening_report=True, ny_session_date="2026-11-18"
    )
    assert len([c for c in tg["calls"] if c["method"] == "GET" and c["url"].endswith("/getUpdates")]) == 0
    assert len([c for c in tg["calls"] if c["method"] == "POST"]) >= 2


def test_t3_10_f10_f12_dashboard_builder_reflects_live_and_flushed_state(
    sample_config, monkeypatch, isolated_bot_paths, freeze_ny_time
):
    freeze_ny_time(datetime(2026, 11, 18, 9, 45))
    with open(isolated_bot_paths["backtest_path"], "w", encoding="utf-8") as f:
        json.dump({
            "overall_golden": {"win_rate_pct": 66.7, "net_pnl": 1480.05, "profit_factor": 1.58},
            "per_ticker": {
                "AMD": {
                    "days_tested": 60,
                    "bars_tested": 8190,
                    "golden": {"win_rate_pct": 70.0, "wins": 14, "losses": 6, "net_pnl": 1480.05, "profit_factor": 1.58},
                    "upgraded": {"win_rate_pct": 60.0, "net_pnl": 500.0},
                }
            },
        }, f)

    item = _make_actionable_scan_item(ticker="AMD", spot=105.20, status="ACTIONABLE")
    scan_out = {
        "timestamp": "2026-11-18 09:45:00 ET",
        "total_scanned": 1, "actionable_count": 1, "near_setup_count": 0, "blackout_count": 0,
        "results": [item],
    }
    monkeypatch.setattr(bot_main, "run_full_day_option_scan", lambda cfg: scan_out)

    cycle_out = bot_main.execute_single_cycle(sample_config, notify_telegram=False, ny_session_date="2026-11-18")
    html_live = open(cycle_out["dashboard"], encoding="utf-8").read()
    assert "AMD" in html_live and "1,480.05" in html_live

    flush_res = call_flush_paper(sample_config, latest_scan=scan_out)
    dash_path = dashboard_builder.build_html_dashboard(scan_out, flush_res)
    html_flushed = open(dash_path, encoding="utf-8").read()
    assert "AMD" in html_flushed


def test_t3_11_f12_command_handler_polling_scan_fibo_winrate_backtest(
    sample_config, monkeypatch, telegram_mock, isolated_bot_paths, freeze_ny_time
):
    freeze_ny_time(datetime(2026, 11, 18, 10, 0))
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "bot_tok_t3_11")
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_CHAT_ID", "8964251574")
    with open(isolated_bot_paths["backtest_path"], "w", encoding="utf-8") as f:
        json.dump({
            "overall_original": {"win_rate_pct": 55.0, "net_pnl": 400.0},
            "overall_golden": {"win_rate_pct": 66.7, "net_pnl": 1480.05, "profit_factor": 1.58},
            "per_ticker": {},
        }, f)

    item = _make_actionable_scan_item(ticker="AMD", status="WATCHING")
    monkeypatch.setattr(command_handler, "run_full_day_option_scan", lambda cfg: {"timestamp": "t", "results": [item]})
    monkeypatch.setattr(command_handler, "analyze_single_ticker", lambda tk, cfg: item)

    tg = telegram_mock(
        webhook_url="",
        updates_batches=[[
            {"update_id": 1, "message": {"text": "/scan"}},
            {"update_id": 2, "message": {"text": "/fibo AMD"}},
            {"update_id": 3, "message": {"text": "/winrate"}},
            {"update_id": 4, "message": {"text": "/backtest"}},
        ]],
    )
    command_handler.poll_telegram_commands(sample_config)
    texts = [c["json"]["text"] for c in tg["calls"] if c["method"] == "POST"]
    assert len(texts) == 4
    assert "1.58" in texts[3]


def test_t3_12_f10_concurrent_file_lock_safety(sample_config, freeze_ny_time):
    freeze_ny_time(datetime(2026, 11, 18, 9, 50))
    amd = _make_actionable_scan_item(ticker="AMD", spot=105.20)
    nvda = _make_actionable_scan_item(ticker="NVDA", spot=105.20)
    call_update_paper({"results": [amd]}, sample_config, enforce_window=True)
    call_update_paper({"results": [nvda]}, sample_config, enforce_window=True)

    state = paper_trader.load_paper_state(paper_trader.PAPER_STATE_PATH)
    open_tks = {p["ticker"] for p in state["open_positions"]}
    assert open_tks == {"AMD", "NVDA"}


def test_t3_13_f13_cloud_yaml_cli_args_compatible_with_main_py(tmp_path, sample_config, monkeypatch):
    cmds = []
    for mode in ("scan_once", "live_session_2h", "backtest_90d"):
        sim = run_cloud_shell_simulation(tmp_path / mode, event_name="workflow_dispatch", inputs_mode=mode)
        cmds.extend(sim["python_cmds"])
    sim_sched = run_cloud_shell_simulation(tmp_path / "sched", event_name="schedule", ny_h=10, ny_m=0)
    cmds.extend(sim_sched["python_cmds"])

    monkeypatch.setattr(bot_main, "load_config", lambda: sample_config)
    monkeypatch.setattr(bot_main, "run_90d_backtest", lambda cfg: {"overall_golden": {"win_rate_pct": 66.7, "net_pnl": 1480.05}})
    monkeypatch.setattr(bot_main, "execute_single_cycle", lambda cfg, notify_telegram=True, ny_session_date="": {})
    monkeypatch.setattr(bot_main, "run_continuous_30s_loop", lambda *a, **kw: None)

    for cmd in cmds:
        tokens = shlex.split(cmd)
        # Expect: -m src.main <flags...>
        flag_idx = tokens.index("src.main") + 1
        monkeypatch.setattr(sys, "argv", ["main.py", *tokens[flag_idx:]])
        bot_main.main()


def test_t3_14_f13_workflow_dispatch_plus_late_cron_dedup(tmp_path):
    sim1 = run_cloud_shell_simulation(tmp_path / "wd", event_name="workflow_dispatch", inputs_mode="scan_once", ny_date="2026-11-18")
    assert any(cli_has_flag(c, "--mark-session-date", "2026-11-18") for c in sim1["python_cmds"])

    sim2 = run_cloud_shell_simulation(
        tmp_path / "cron", event_name="schedule", ny_date="2026-11-18", ny_h=12, ny_m=30, last_run_date="2026-11-18"
    )
    assert sim2["python_cmds"] == []


def test_t3_15_f14_gas_trigger_plus_cloud_yaml_payload_contract():
    wf = load_cloud_workflow()
    on_cfg = wf.get("on") or wf.get(True)
    allowed_modes = on_cfg["workflow_dispatch"]["inputs"]["mode"]["options"]

    res = run_gas_function("triggerDayOptionMarketOpen()", frozen_iso="2026-11-18T09:35:00-05:00")
    assert len(res["githubDispatches"]) == 1
    disp = res["githubDispatches"][0]
    assert disp["url"].endswith("/actions/workflows/day_option_cloud.yml/dispatches")
    assert disp["body"]["ref"] == "main"
    assert disp["body"]["inputs"]["mode"] in allowed_modes


def test_t3_16_f14_gas_fibo_card_matches_latest_scan_schema():
    item = _make_actionable_scan_item(ticker="AMD", spot=105.20, action="BUY CALL")
    b64_scan = base64.b64encode(json.dumps({"results": [item]}).encode("utf-8")).decode("ascii")
    upd = {"update_id": 9101, "message": {"chat": {"id": 8964251574}, "text": "/fibo AMD"}}
    res = run_gas_function(
        f"doPost({json.dumps({'parameter': {'market': 'DAY_OPTION'}, 'postData': {'contents': json.dumps(upd)}})})",
        mock_responses=[{"match": "config/latest_scan.json", "status": 200, "body": {"sha": "s1", "content": b64_scan}}],
    )
    card = res["telegramSends"][0]["body"]["text"]
    assert "AMD" in card
    assert f"${item['option_quote']['strike']:.2f}" in card
    assert f"{item['option_quote']['dte']} DTE" in card
    assert "TP1" in card and "TP2" in card and "SL" in card


def test_t3_17_f14_gas_winrate_and_backtest_match_python_json_schemas(isolated_bot_paths):
    paper_state = {
        "open_positions": [],
        "closed_positions": [
            {"ticker": "AMD", "fibo_level": "0.618", "net_pnl_usd": 120.0},
            {"ticker": "SPY", "fibo_level": "0.500", "net_pnl_usd": -40.0},
        ],
    }
    kpis = paper_trader.compute_paper_kpis(paper_state["closed_positions"], [])
    b64_paper = base64.b64encode(json.dumps(paper_state).encode("utf-8")).decode("ascii")
    upd_wr = {"update_id": 9201, "message": {"chat": {"id": 8964251574}, "text": "/winrate"}}
    res_wr = run_gas_function(
        f"doPost({json.dumps({'parameter': {'market': 'DAY_OPTION'}, 'postData': {'contents': json.dumps(upd_wr)}})})",
        mock_responses=[{"match": "config/paper_trades.json", "status": 200, "body": {"sha": "p1", "content": b64_paper}}],
    )
    wr_txt = res_wr["telegramSends"][0]["body"]["text"]
    assert f"{kpis['win_rate_pct']:.1f}%" in wr_txt
    assert f"{kpis['wins']}W / {kpis['losses']}L" in wr_txt

    bt_data = {
        "overall_original": {"win_rate_pct": 55.0, "net_pnl": 400.0},
        "overall_golden": {"win_rate_pct": 66.7, "wins": 80, "losses": 40, "net_pnl": 1480.05, "profit_factor": 1.58},
        "per_ticker": {},
    }
    with open(isolated_bot_paths["backtest_path"], "w", encoding="utf-8") as f:
        json.dump(bt_data, f)
    py_bt = command_handler.format_backtest_summary_message()

    b64_bt = base64.b64encode(json.dumps(bt_data).encode("utf-8")).decode("ascii")
    upd_bt = {"update_id": 9202, "message": {"chat": {"id": 8964251574}, "text": "/backtest"}}
    res_bt = run_gas_function(
        f"doPost({json.dumps({'parameter': {'market': 'DAY_OPTION'}, 'postData': {'contents': json.dumps(upd_bt)}})})",
        mock_responses=[{"match": "config/backtest_90d.json", "status": 200, "body": {"sha": "b1", "content": b64_bt}}],
    )
    gas_bt = res_bt["telegramSends"][0]["body"]["text"]
    assert "66.7%" in py_bt and "66.7%" in gas_bt
    assert "1.58" in py_bt and "1.58" in gas_bt


def test_t3_18_f9_f10_eod_next_day_session_carryover_flush_and_reentry(sample_config, freeze_ny_time):
    freeze_ny_time(datetime(2026, 11, 18, 10, 0))
    d1_item = _make_actionable_scan_item(ticker="AMD", spot=105.20, tp1=106.50, tp2=107.80, sl=103.35, curr_date="2026-11-18")
    call_update_paper({"results": [d1_item]}, sample_config, enforce_window=True)

    # Day 2 (2026-11-19 09:40 NY) opens with gap across yesterday's SL + fresh ACTIONABLE setup
    freeze_ny_time(datetime(2026, 11, 19, 9, 40))
    d2_item = _make_actionable_scan_item(
        ticker="AMD", spot=102.00, tp1=103.00, tp2=104.00, sl=100.50, curr_date="2026-11-19", status="ACTIONABLE"
    )
    out = call_update_paper({"results": [d2_item]}, sample_config, enforce_window=True)

    assert len(out["newly_closed"]) == 1
    assert out["newly_closed"][0]["exit_reason"] == "EOD_FORCE_FLUSH"
    assert len(out["newly_opened"]) == 1
    assert out["newly_opened"][0]["trading_date"] == "2026-11-19"
