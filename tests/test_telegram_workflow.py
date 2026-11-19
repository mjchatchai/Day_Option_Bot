"""
Offline deterministic unit and mock tests for Day Option Bot M2 (Features F11-F14):
- F11: telegram_notifier.py (5-line cards, html.escape, guards, empty-picks summary,
       HTML-safe 4096-char splitting, strict DAY_OPTION credential guard, non-200 logging & retry)
- F12: command_handler.py, main.py, dashboard_builder.py (getWebhookInfo skip + --no-command-poll,
       non-200 polling log, /fibo NEUTRAL/NO_DATA guard, 11:30 NY wall-clock cutoff, final MTM
       scan + flush + dashboard rebuild, dynamic DASHBOARD_PATH, failure tracebacks/exit)
- F13: .github/workflows/day_option_cloud.yml (timeout-minutes: 200, NY_DATE + --mark-session-date
       in workflow_dispatch, EST/EDT pre-warm [490, 575), REM_MINS >= 1 cap, no pre-market --once)
- F14: gas_webhook/Code.gs (Node vm mock-GAS tests for triggerGitHubWorkflowWithInputs,
       triggerDayOptionMarketOpen, ?action=day_option_health, /scan, /fibo, /winrate, /backtest)
"""

from datetime import datetime
from html.parser import HTMLParser
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import pytest
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from day_option_bot.src import command_handler, dashboard_builder, main as bot_main, telegram_notifier


NY_TZ = ZoneInfo("America/New_York")
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
WORKFLOW_PATH = os.path.join(REPO_ROOT, "day_option_bot", ".github", "workflows", "day_option_cloud.yml")
GAS_CODE_PATH = os.path.join(REPO_ROOT, "gas_webhook", "Code.gs")


class _TagBalanceChecker(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.stack: List[str] = []
        self.mismatch = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        self.stack.append(tag.lower())

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if self.stack and self.stack[-1] == t:
            self.stack.pop()
        else:
            self.mismatch = True


def _sample_scan_item(
    ticker: str = "AMD",
    status: str = "ACTIONABLE",
    action: str = "BUY CALL",
    delta: float = 0.63,
    tp1_gain: float = 28.5,
) -> Dict:
    return {
        "ticker": ticker,
        "status": status,
        "status_thai": "พร้อมเข้าเทรด",
        "option_action": action,
        "spot_price": 152.35,
        "ema200": 150.10,
        "rvol": 1.45,
        "prev_trading_date": "2026-11-18",
        "current_trading_date": "2026-11-19",
        "active_fibo_level": "0.618",
        "distance_to_active_fibo_pct": 0.12,
        "fibo_levels": {
            "0.236": 150.2,
            "0.382": 151.0,
            "0.500": 151.7,
            "0.618": 152.3,
            "0.786": 153.2,
            "anchor_point_1_label": "Low $149.00",
            "anchor_point_2_label": "High $154.00",
        },
        "candle_reversal": {"pattern_name": "Bullish Hammer"},
        "earnings": {"status_label": "Safe (45d)"},
        "trade_plan": {
            "fibo_entry_level": "0.618",
            "fibo_entry_price": 152.30,
            "tp1_label": "Fibo 0.382",
            "tp1_underlying": 154.00,
            "lock_sl_after_tp1": 152.30,
            "tp2_label": "Fibo 0.236",
            "tp2_underlying": 155.50,
            "sl_label": "Fibo 0.786",
            "sl_underlying": 150.80,
        },
        "option_quote": {
            "expiration": "2026-11-20",
            "dte": 1,
            "strike": 152.50,
            "option_type": "CALL" if "CALL" in action else "PUT",
            "delta": delta,
            "realtime_theo_price": 2.45,
            "est_bid": 2.40,
            "est_ask": 2.50,
            "max_limit_buy": 2.52,
            "recommended_contracts": 2,
            "tp1_option_price": 3.15,
            "tp1_gain_pct": tp1_gain,
            "tp2_option_price": 4.10,
            "tp2_gain_pct": 67.3,
            "sl_option_price": 1.65,
            "sl_loss_pct": -32.7,
        },
    }


def _dummy_config() -> Dict:
    return {
        "watchlist": {"etf_index": ["SPY"], "golden_stocks": ["AMD"]},
        "paper_trading": {"starting_balance_usd": 5000.0},
    }


class _FakeResponse:
    def __init__(self, status_code: int, json_data: Any = None, text: str = "", headers: Dict = None) -> None:
        self.status_code = status_code
        self._json = json_data
        self.text = text or (json.dumps(json_data) if json_data is not None else "")
        self.headers = headers or {}

    def json(self) -> Any:
        if isinstance(self._json, Exception):
            raise self._json
        return self._json


# ==============================================================================
# F11: telegram_notifier.py
# ==============================================================================


def test_f11_format_signal_card_5lines_escape_and_guards():
    assert "5-line" in (telegram_notifier.format_signal_card.__doc__ or "")

    card = telegram_notifier.format_signal_card(_sample_scan_item("AMD", "ACTIONABLE", "BUY CALL"))
    assert len(card.splitlines()) == 5
    assert "เข้าทันที" in card
    assert "+0.63" in card

    # HTML escaping of dynamic strings
    xss_item = _sample_scan_item("<script>&>", "NEAR_SETUP", "BUY PUT<bad>", delta=-0.55)
    xss_item["trade_plan"]["fibo_entry_level"] = "0.618<x>"
    xss_item["option_quote"]["expiration"] = "2026-11-20&exp"
    xss_item["option_quote"]["option_type"] = "PUT>"
    escaped = telegram_notifier.format_signal_card(xss_item)
    assert "<script>" not in escaped
    assert "&lt;script&gt;&amp;&gt;" in escaped
    assert "0.618&lt;x&gt;" in escaped
    assert "2026-11-20&amp;exp" in escaped

    # NO_DATA, NEUTRAL, and missing/empty option_quote / trade_plan guards
    no_data = telegram_notifier.format_signal_card({"ticker": "NVDA<1>", "status": "NO_DATA", "reason": "bad<data>"})
    assert "NO_DATA" in no_data and "&lt;data&gt;" in no_data and "<data>" not in no_data

    neutral = telegram_notifier.format_signal_card(
        {"ticker": "SPY", "status": "NEUTRAL", "spot_price": 500.0, "ema200": 500.0}
    )
    assert "NEUTRAL" in neutral and "$500.00" in neutral

    empty_oq = telegram_notifier.format_signal_card(
        {"ticker": "AAPL", "status": "WATCHING", "trade_plan": {"fibo_entry_level": "0.618"}, "option_quote": {}}
    )
    assert "AAPL" in empty_oq and "WATCHING" in empty_oq


def test_f11_build_scan_telegram_report_empty_picks_and_missing_quotes():
    scan_out = {
        "timestamp": "2026-11-19 09:35:00 ET <ok>",
        "results": [
            {"ticker": "SPY", "status": "GAP_SKIP", "spot_price": 590.0, "reason": "Gap > 2.5%"},
            {"ticker": "NVDA", "status": "EARNINGS_BLACKOUT", "spot_price": 140.0, "reason": "Earnings <2d>"},
            {"ticker": "AMD", "status": "ACTIONABLE", "trade_plan": {"fibo_entry_level": "0.618"}, "option_quote": {}},
        ],
    }
    paper = {"kpis": {"win_rate_pct": 60.0, "wins": 3, "losses": 2, "net_pnl_usd": -45.5}}
    report = telegram_notifier.build_scan_telegram_report(scan_out, paper)
    assert "&lt;ok&gt;" in report
    assert "&lt;2d&gt;" in report
    assert "ไม่มีสัญญาณพร้อมเทรดในรอบนี้" in report
    assert "SPY" in report and "GAP_SKIP" in report
    assert "NVDA" in report and "EARNINGS_BLACKOUT" in report
    assert "$-45.50" in report or "-$45.50" in report or "-45.50" in report


def test_f11_split_telegram_html_preserves_tags_and_entities():
    block = "🎫 <b>AMD — BUY CALL &amp; HOLD</b>\n• <code>Line with value 1234567890 &lt;tag&gt;</code>\n\n"
    payload = "<b>" + (block * 20) + "</b>"
    chunks = telegram_notifier.split_telegram_html(payload, max_len=160)
    assert len(chunks) > 1
    for ch in chunks:
        assert len(ch) <= 160
        # No broken tag boundaries
        assert ch.count("<") == ch.count(">")
        parser = _TagBalanceChecker()
        parser.feed(ch)
        assert not parser.mismatch
        assert parser.stack == []
        # No cut HTML entity at tail
        assert not re.search(r"&[a-zA-Z0-9#]*$", ch)


def test_f11_credentials_guard_and_send_retry(monkeypatch, capsys):
    # Strict credential guard: generic TELEGRAM_* must NOT be used by default
    monkeypatch.delenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("DAY_OPTION_TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("DAY_OPTION_ALLOW_TELEGRAM_FALLBACK", raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "other_bot_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "other_chat_id")

    assert telegram_notifier.get_telegram_credentials() == ("", "")
    assert telegram_notifier.get_telegram_credentials(allow_fallback=True) == ("other_bot_token", "other_chat_id")
    assert "WARNING" in capsys.readouterr().out

    # Placeholder rejection on both token and chat_id (case-insensitive)
    assert telegram_notifier.send_telegram_message("hi", token="YOUR_token", chat_id="123") is False
    assert telegram_notifier.send_telegram_message("hi", token="real_token", chat_id="your_chat") is False

    # Independent resolution when only one is None
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "env_tk")
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_CHAT_ID", "env_cid")
    posted = []

    def _fake_post_ok(url, json=None, timeout=None):
        posted.append((url, json))
        return _FakeResponse(200, {"ok": True})

    monkeypatch.setattr(telegram_notifier.requests, "post", _fake_post_ok)
    assert telegram_notifier.send_telegram_message("hello", token="override_tk", chat_id=None) is True
    assert posted[-1][0].endswith("/botoverride_tk/sendMessage")
    assert posted[-1][1]["chat_id"] == "env_cid"

    # 400 Bad Request -> log status+body and NO retry
    calls = {"n": 0}
    monkeypatch.setattr(telegram_notifier.time, "sleep", lambda _: None)

    def _fake_post_400(url, json=None, timeout=None):
        calls["n"] += 1
        return _FakeResponse(400, {"ok": False}, text="Bad Request: can't parse entities")

    monkeypatch.setattr(telegram_notifier.requests, "post", _fake_post_400)
    assert telegram_notifier.send_telegram_message("bad", token="tk", chat_id="cid") is False
    assert calls["n"] == 1
    assert "HTTP 400" in capsys.readouterr().out

    # 429 then 500 then 200 -> bounded retry succeeds
    seq = [
        _FakeResponse(429, {"ok": False, "parameters": {"retry_after": 1}}, text="Too Many Requests"),
        _FakeResponse(500, text="Internal Server Error"),
        _FakeResponse(200, {"ok": True}),
    ]
    idx = {"i": 0}
    slept = []
    monkeypatch.setattr(telegram_notifier.time, "sleep", lambda s: slept.append(s))

    def _fake_post_retry(url, json=None, timeout=None):
        r = seq[idx["i"]]
        idx["i"] += 1
        return r

    monkeypatch.setattr(telegram_notifier.requests, "post", _fake_post_retry)
    assert telegram_notifier.send_telegram_message("retry", token="tk", chat_id="cid", max_retries=3) is True
    assert idx["i"] == 3
    assert len(slept) == 2

    # RequestException exhausts retries -> False
    ex_calls = {"n": 0}

    def _fake_post_exc(url, json=None, timeout=None):
        ex_calls["n"] += 1
        raise requests.RequestException("network down")

    monkeypatch.setattr(telegram_notifier.requests, "post", _fake_post_exc)
    assert telegram_notifier.send_telegram_message("exc", token="tk", chat_id="cid", max_retries=2) is False
    assert ex_calls["n"] == 2


# ==============================================================================
# F12: command_handler.py, main.py, dashboard_builder.py
# ==============================================================================


def test_f12_webhook_check_and_command_handler(monkeypatch, tmp_path, capsys):
    command_handler._WEBHOOK_CACHE.clear()
    calls = {"getWebhookInfo": 0}

    def _fake_get(url, params=None, timeout=None):
        if "getWebhookInfo" in url:
            calls["getWebhookInfo"] += 1
            return _FakeResponse(200, {"ok": True, "result": {"url": "https://script.google.com/macros/s/abc/exec"}})
        raise AssertionError(f"Unexpected GET {url}")

    monkeypatch.setattr(command_handler.requests, "get", _fake_get)
    assert command_handler.has_active_telegram_webhook(token="tok1") is True
    assert command_handler.has_active_telegram_webhook(token="tok1") is True
    assert calls["getWebhookInfo"] == 1  # cached

    # Non-200 getUpdates logs error instead of silent return
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_BOT_TOKEN", "tok_poll")
    monkeypatch.setenv("DAY_OPTION_TELEGRAM_CHAT_ID", "chat_1")
    monkeypatch.setattr(command_handler, "OFFSET_FILE", str(tmp_path / "offset.txt"))

    def _fake_get_409(url, params=None, timeout=None):
        return _FakeResponse(409, {"ok": False}, text="Conflict: terminated by other getUpdates request")

    monkeypatch.setattr(command_handler.requests, "get", _fake_get_409)
    command_handler.poll_telegram_commands(_dummy_config())
    assert "HTTP 409" in capsys.readouterr().out

    # /fibo handles NEUTRAL and NO_DATA without KeyError
    sent_msgs: List[str] = []
    monkeypatch.setattr(
        command_handler.requests,
        "get",
        lambda url, params=None, timeout=None: _FakeResponse(
            200,
            {
                "ok": True,
                "result": [
                    {"update_id": 10, "message": {"text": "/fibo NVDA"}},
                    {"update_id": 11, "message": {"text": "/fibo BAD<TK>"}},
                ],
            },
        ),
    )
    monkeypatch.setattr(
        command_handler,
        "analyze_single_ticker",
        lambda tk, cfg: (
            {"ticker": tk, "status": "NEUTRAL", "spot_price": 140.0, "ema200": 140.0}
            if tk == "NVDA"
            else {"ticker": tk, "status": "NO_DATA", "reason": "missing"}
        ),
    )
    monkeypatch.setattr(
        command_handler,
        "send_telegram_message",
        lambda txt, tk=None, cid=None: sent_msgs.append(txt) or True,
    )
    command_handler.poll_telegram_commands(_dummy_config())
    assert len(sent_msgs) == 2
    assert "NEUTRAL" in sent_msgs[0]
    assert "BAD&lt;TK&gt;" in sent_msgs[1]


def test_f12_main_loop_webhook_skip_cutoff_flush_and_dynamic_dashboard(monkeypatch, tmp_path, capsys):
    # Ensure live docs/index.html is NEVER touched by using dynamic DASHBOARD_PATH
    test_dash = tmp_path / "docs" / "index.html"
    test_scan = tmp_path / "config" / "latest_scan.json"
    monkeypatch.setattr(dashboard_builder, "DASHBOARD_PATH", str(test_dash))
    monkeypatch.setattr(bot_main, "LATEST_SCAN_PATH", str(test_scan))
    monkeypatch.setattr(bot_main.time, "sleep", lambda _: None)

    poll_calls = {"n": 0}
    monkeypatch.setattr(bot_main, "has_active_telegram_webhook", lambda: True)
    monkeypatch.setattr(bot_main, "poll_telegram_commands", lambda cfg: poll_calls.__setitem__("n", poll_calls["n"] + 1))

    # Clock starts at 11:29:30 NY, advances past 11:30:00 NY after cycle 1
    clock_seq = [
        datetime(2026, 11, 19, 11, 29, 30, tzinfo=NY_TZ),  # start_ny
        datetime(2026, 11, 19, 11, 29, 35, tzinfo=NY_TZ),  # _stop_reason cycle 1
        datetime(2026, 11, 19, 11, 30, 5, tzinfo=NY_TZ),   # _stop_reason in sleep loop -> cutoff!
        datetime(2026, 11, 19, 11, 30, 5, tzinfo=NY_TZ),   # top of loop -> cutoff flush!
    ]
    clk_idx = {"i": 0}

    def _fake_now_ny():
        i = min(clk_idx["i"], len(clock_seq) - 1)
        clk_idx["i"] += 1
        return clock_seq[i]

    monkeypatch.setattr(bot_main, "now_ny", _fake_now_ny)

    scan_calls = {"n": 0}

    def _fake_scan(cfg):
        scan_calls["n"] += 1
        return {
            "timestamp": "2026-11-19 11:30:00 ET",
            "total_scanned": 1,
            "actionable_count": 0,
            "near_setup_count": 0,
            "results": [_sample_scan_item("AMD", "WATCHING", "BUY CALL")],
        }

    monkeypatch.setattr(bot_main, "run_full_day_option_scan", _fake_scan)
    monkeypatch.setattr(
        bot_main,
        "update_paper_portfolio",
        lambda scan, cfg: {
            "newly_closed": [],
            "kpis": {"win_rate_pct": 50.0, "wins": 1, "losses": 1, "net_pnl_usd": 20.0, "fibo_0618": {"win_rate_pct": 50.0}},
        },
    )

    flush_args: List[Dict] = []

    def _fake_flush(cfg, latest_scan=None):
        flush_args.append({"latest_scan": latest_scan})
        return {
            "flushed": [
                {
                    "ticker": "AMD<X>",
                    "option_action": "BUY CALL",
                    "entry_option_price": 2.0,
                    "exit_option_price": 2.5,
                    "net_pnl_usd": 45.0,
                    "is_win": True,
                }
            ],
            "kpis": {"win_rate_pct": 66.7, "wins": 2, "losses": 1, "net_pnl_usd": 65.0, "fibo_0618": {"win_rate_pct": 66.7}},
        }

    monkeypatch.setattr(bot_main, "flush_open_paper_positions_at_session_end", _fake_flush)

    dash_calls: List[Dict] = []
    orig_build = dashboard_builder.build_html_dashboard

    def _spy_build(scan_out, paper_out, out_path=None):
        dash_calls.append(paper_out)
        return orig_build(scan_out, paper_out, out_path=out_path)

    monkeypatch.setattr(bot_main, "build_html_dashboard", _spy_build)

    sent_alerts: List[str] = []
    monkeypatch.setattr(bot_main, "send_telegram_message", lambda txt: sent_alerts.append(txt) or False)

    bot_main.run_continuous_30s_loop(
        _dummy_config(),
        interval_sec=30,
        max_minutes=115,
        ny_session_date="2026-11-19",
    )

    assert poll_calls["n"] == 0  # skipped because webhook active
    assert scan_calls["n"] == 2  # 1 regular cycle + 1 final MTM scan on cutoff
    assert len(flush_args) == 1 and flush_args[0]["latest_scan"] is not None
    assert flush_args[0]["latest_scan"]["ny_session_date"] == "2026-11-19"
    assert len(dash_calls) == 2  # second call is after flush!
    assert dash_calls[-1]["kpis"]["win_rate_pct"] == 66.7
    assert test_dash.exists()
    assert json.loads(test_scan.read_text(encoding="utf-8"))["ny_session_date"] == "2026-11-19"
    assert any("AMD&lt;X&gt;" in a for a in sent_alerts)
    assert "WARNING: Telegram notification failed" in capsys.readouterr().out

    # Test --no-command-poll & all-cycles-fail raising RuntimeError
    monkeypatch.setattr(bot_main, "has_active_telegram_webhook", lambda: False)
    monkeypatch.setattr(bot_main, "run_full_day_option_scan", lambda cfg: (_ for _ in ()).throw(RuntimeError("boom")))
    clk_fail = [
        datetime(2026, 11, 19, 9, 40, 0, tzinfo=NY_TZ),
        datetime(2026, 11, 19, 9, 40, 5, tzinfo=NY_TZ),
        datetime(2026, 11, 19, 11, 30, 5, tzinfo=NY_TZ),
        datetime(2026, 11, 19, 11, 30, 5, tzinfo=NY_TZ),
    ]
    fail_i = {"i": 0}
    monkeypatch.setattr(bot_main, "now_ny", lambda: clk_fail[min(fail_i.__setitem__("i", fail_i["i"] + 1) or fail_i["i"] - 1, len(clk_fail) - 1)])

    with pytest.raises(RuntimeError, match="All 1 scan cycle"):
        bot_main.run_continuous_30s_loop(_dummy_config(), interval_sec=5, no_command_poll=True)
    assert poll_calls["n"] == 0


# ==============================================================================
# F13: day_option_cloud.yml
# ==============================================================================


def _simulate_cloud_schedule(ny_mins: int, last_run_date: str, ny_date: str = "2026-11-19") -> Dict[str, Any]:
    """Mirrors the fixed schedule logic in day_option_cloud.yml."""
    if last_run_date == ny_date and ny_mins >= 690:
        return {"action": "skip_duplicate"}
    if 490 <= ny_mins < 575:
        wait_sec = 575 * 60 - (ny_mins * 60)
        return {"action": "prewarm_loop", "wait_sec": wait_sec, "max_minutes": 115}
    if 575 <= ny_mins < 690:
        rem_mins = max(1, 690 - ny_mins)
        return {"action": "active_loop", "max_minutes": rem_mins}
    if ny_mins < 575:
        return {"action": "skip_premarket"}
    if last_run_date != ny_date:
        return {"action": "late_once"}
    return {"action": "already_scanned"}


def test_f13_cloud_workflow_yaml_and_dst_schedule():
    raw = open(WORKFLOW_PATH, "r", encoding="utf-8").read()

    # Optional PyYAML validation (not in requirements.txt, installed in some envs)
    try:
        import yaml  # type: ignore

        parsed = yaml.safe_load(raw)
        job = parsed["jobs"]["run-day-option-bot"]
        assert job["timeout-minutes"] == 200
        exec_step = next(s for s in job["steps"] if "Execute Day Option Bot" in s.get("name", ""))
        run_script = exec_step["run"]
    except ImportError:
        assert "timeout-minutes: 200" in raw
        run_script = raw

    # NY_DATE defined before workflow_dispatch and passed in all 3 modes
    assert run_script.index('NY_DATE=$(TZ="America/New_York" date +%Y-%m-%d)') < run_script.index("workflow_dispatch")
    for mode_cmd in [
        'python -m src.main --backtest --once --mark-session-date "$NY_DATE"',
        'python -m src.main --loop --interval 30 --max-minutes 115 --send-opening-report --mark-session-date "$NY_DATE"',
        'python -m src.main --once --mark-session-date "$NY_DATE"',
    ]:
        assert mode_cmd in run_script

    # Pre-warm EST 13:15 UTC (495) & 13:35 UTC (515), exact >= 690 cutoff, REM_MINS cap, pre-market skip
    assert '"$NY_MINS" -ge 690' in run_script
    assert '"$NY_MINS" -ge 490' in run_script
    assert 'if [ "$REM_MINS" -lt 1 ]; then REM_MINS=1; fi' in run_script
    assert 'if [ "$REM_MINS" -lt 15 ]' not in run_script
    assert 'elif [ "$NY_MINS" -lt 575 ]; then' in run_script

    # DST & boundary matrix verification
    assert _simulate_cloud_schedule(470, "")["action"] == "skip_premarket"
    assert _simulate_cloud_schedule(495, "") == {"action": "prewarm_loop", "wait_sec": 4800, "max_minutes": 115}
    assert _simulate_cloud_schedule(515, "") == {"action": "prewarm_loop", "wait_sec": 3600, "max_minutes": 115}
    assert _simulate_cloud_schedule(575, "") == {"action": "active_loop", "max_minutes": 115}
    assert _simulate_cloud_schedule(685, "") == {"action": "active_loop", "max_minutes": 5}
    assert _simulate_cloud_schedule(690, "2026-11-19")["action"] == "skip_duplicate"
    assert _simulate_cloud_schedule(690, "")["action"] == "late_once"


# ==============================================================================
# F14: gas_webhook/Code.gs (Node vm.Script + mock-GAS execution)
# ==============================================================================


def test_f14_gas_webhook_vm_execution():
    backtest_json_path = os.path.join(REPO_ROOT, "day_option_bot", "config", "backtest_90d.json")
    latest_scan_path = os.path.join(REPO_ROOT, "day_option_bot", "config", "latest_scan.json")

    node_test_script = r"""
const fs = require("fs");
const vm = require("vm");
const assert = require("assert");

const gasPath = process.env.GAS_CODE_PATH;
const btPath = process.env.BT_JSON_PATH;
const scanPath = process.env.SCAN_JSON_PATH;
const code = fs.readFileSync(gasPath, "utf8");
new vm.Script(code, { filename: "Code.gs" });

function formatInNY(dateObj, pattern) {
  const parts = {};
  const fmt = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    weekday: "short",
    hour12: false,
  });
  for (const p of fmt.formatToParts(dateObj)) {
    parts[p.type] = p.value;
  }
  const hh = parts.hour === "24" ? "00" : parts.hour;
  const dayMap = { Mon: "1", Tue: "2", Wed: "3", Thu: "4", Fri: "5", Sat: "6", Sun: "7" };
  if (pattern === "u") return dayMap[parts.weekday];
  if (pattern === "yyyy-MM-dd") return `${parts.year}-${parts.month}-${parts.day}`;
  if (pattern === "HHmm") return `${hh}${parts.minute}`;
  if (pattern === "yyyy-MM-dd HH:mm:ss") return `${parts.year}-${parts.month}-${parts.day} ${hh}:${parts.minute}:${parts.second}`;
  return `${parts.year}-${parts.month}-${parts.day}`;
}

function makeSandbox(opts) {
  const store = Object.assign({}, opts.props || {});
  const tgCalls = [];
  const ghCalls = [];
  const errors = [];
  let fixedNow = opts.now || new Date("2026-11-19T14:35:00Z"); // 09:35 EST Thu

  const OrigDate = Date;
  function MockDate(...args) {
    if (args.length === 0) return new OrigDate(fixedNow.getTime());
    return new OrigDate(...args);
  }
  MockDate.now = () => fixedNow.getTime();
  MockDate.UTC = OrigDate.UTC;
  MockDate.parse = OrigDate.parse;
  MockDate.prototype = OrigDate.prototype;

  const sandbox = {
    Date: MockDate,
    console: {
      log: () => {},
      error: (...a) => errors.push(a.map(String).join(" ")),
    },
    PropertiesService: {
      getScriptProperties: () => ({
        getProperty: (k) => (k in store ? store[k] : null),
        setProperty: (k, v) => { store[k] = String(v); },
      }),
    },
    ScriptApp: {
      getProjectTriggers: () => (opts.triggers || ["triggerDayOptionMarketOpen"]).map((fn) => ({
        getHandlerFunction: () => fn,
      })),
      deleteTrigger: () => {},
      newTrigger: () => ({ timeBased: () => ({ everyMinutes: () => ({ create: () => {} }) }) }),
    },
    Utilities: {
      formatDate: (d, tz, pat) => formatInNY(d, pat),
      base64Decode: (s) => Buffer.from(s, "base64"),
      newBlob: (buf) => ({ getDataAsString: () => Buffer.from(buf).toString("utf8"), getBytes: () => buf }),
      base64Encode: (b) => Buffer.from(b).toString("base64"),
    },
    ContentService: {
      MimeType: { JSON: "JSON", TEXT: "TEXT" },
      createTextOutput: (txt) => ({
        _txt: txt,
        setMimeType: function() { return this; },
        getContent: function() { return this._txt; },
      }),
    },
    HtmlService: {
      createHtmlOutput: (txt) => ({ _txt: txt }),
    },
    UrlFetchApp: {
      fetch: (url, params) => {
        if (url.includes("api.telegram.org")) {
          tgCalls.push({ url, payload: JSON.parse(params.payload) });
          return { getResponseCode: () => 200, getContentText: () => '{"ok":true}' };
        }
        if (url.includes("/actions/workflows/")) {
          ghCalls.push({ url, payload: JSON.parse(params.payload) });
          if (opts.ghThrow) throw new Error("DNS timeout");
          const code = opts.ghCode != null ? opts.ghCode : 204;
          const body = opts.ghBody != null ? opts.ghBody : "";
          return { getResponseCode: () => code, getContentText: () => body };
        }
        if (url.includes("/contents/")) {
          const mockJson = opts.ghContents ? opts.ghContents(url) : {};
          const b64 = Buffer.from(JSON.stringify(mockJson), "utf8").toString("base64");
          return {
            getResponseCode: () => 200,
            getContentText: () => JSON.stringify({ sha: "abc", content: b64 }),
          };
        }
        return { getResponseCode: () => 200, getContentText: () => "{}" };
      },
    },
  };
  const ctx = vm.createContext(sandbox);
  vm.runInContext(code, ctx);
  return { ctx, store, tgCalls, ghCalls, errors, setNow: (d) => { fixedNow = d; } };
}

// 1. triggerGitHubWorkflowWithInputs structured return & missing GH_TOKEN
{
  const s1 = makeSandbox({ props: {} });
  const rMissing = s1.ctx.triggerGitHubWorkflowWithInputs("mjchatchai/Day_Option_Bot", "day_option_cloud.yml", { mode: "scan_once" });
  assert.strictEqual(rMissing.ok, false);
  assert.strictEqual(rMissing.code, 0);
  assert.ok(rMissing.body.includes("Missing GH_TOKEN"));

  const s2 = makeSandbox({ props: { GH_TOKEN: "ghp_secret_123" }, ghCode: 204 });
  const rOk = s2.ctx.triggerGitHubWorkflowWithInputs("mjchatchai/Day_Option_Bot", "day_option_cloud.yml", { mode: "scan_once" });
  assert.strictEqual(rOk.ok, true);
  assert.strictEqual(rOk.code, 204);

  const s3 = makeSandbox({ props: { GH_TOKEN: "ghp_secret_123" }, ghCode: 401, ghBody: "Bad credentials" });
  const r401 = s3.ctx.triggerGitHubWorkflowWithInputs("mjchatchai/Day_Option_Bot", "day_option_cloud.yml", { mode: "scan_once" });
  assert.strictEqual(r401.ok, false);
  assert.strictEqual(r401.code, 401);
  assert.strictEqual(r401.body, "Bad credentials");

  const s4 = makeSandbox({ props: { GH_TOKEN: "ghp_secret_123" }, ghThrow: true });
  const rExc = s4.ctx.triggerGitHubWorkflowWithInputs("mjchatchai/Day_Option_Bot", "day_option_cloud.yml", { mode: "scan_once" });
  assert.strictEqual(rExc.ok, false);
  assert.strictEqual(rExc.code, 0);
}

// 2. triggerDayOptionMarketOpen: EST & EDT 09:35 NY, dedup, error alerting without setting date
{
  // Success EST (14:35Z = 09:35 EST) & dedup
  const sOk = makeSandbox({ props: { GH_TOKEN: "ghp_secret_123" }, ghCode: 204, now: new Date("2026-11-19T14:35:00Z") });
  sOk.ctx.triggerDayOptionMarketOpen();
  assert.strictEqual(sOk.ghCalls.length, 1);
  assert.strictEqual(sOk.ghCalls[0].payload.inputs.mode, "live_session_2h");
  assert.strictEqual(sOk.store.LAST_DAY_OPTION_OPEN_DATE, "2026-11-19");
  sOk.ctx.triggerDayOptionMarketOpen();
  assert.strictEqual(sOk.ghCalls.length, 1); // deduped

  // Success EDT (13:35Z = 09:35 EDT in July)
  const sEdt = makeSandbox({ props: { GH_TOKEN: "ghp_secret_123" }, ghCode: 204, now: new Date("2026-07-15T13:35:00Z") });
  sEdt.ctx.triggerDayOptionMarketOpen();
  assert.strictEqual(sEdt.ghCalls.length, 1);
  assert.strictEqual(sEdt.store.LAST_DAY_OPTION_OPEN_DATE, "2026-07-15");

  // Failure on 403 -> logs error, sends Telegram alert, does NOT set LAST_DAY_OPTION_OPEN_DATE
  const sFail = makeSandbox({ props: { GH_TOKEN: "ghp_secret_123" }, ghCode: 403, ghBody: "<Forbidden>", now: new Date("2026-11-19T14:35:00Z") });
  sFail.ctx.triggerDayOptionMarketOpen();
  assert.strictEqual(sFail.store.LAST_DAY_OPTION_OPEN_DATE, undefined);
  assert.ok(sFail.errors.length > 0);
  assert.strictEqual(sFail.tgCalls.length, 1);
  assert.ok(sFail.tgCalls[0].payload.text.includes("403"));
  assert.ok(sFail.tgCalls[0].payload.text.includes("&lt;Forbidden&gt;"));
}

// 3. doGet ?action=day_option_health exposes NO secrets
{
  const sH = makeSandbox({ props: { GH_TOKEN: "ghp_top_secret_value_XYZ", LAST_DAY_OPTION_OPEN_DATE: "2026-11-19" } });
  const out = sH.ctx.doGet({ parameter: { action: "day_option_health" } }).getContent();
  const parsed = JSON.parse(out);
  assert.strictEqual(parsed.ok, true);
  assert.strictEqual(parsed.hasGhToken, true);
  assert.strictEqual(parsed.triggerCount, 1);
  assert.deepStrictEqual(Array.from(parsed.triggerFunctions), ["triggerDayOptionMarketOpen"]);
  assert.strictEqual(parsed.lastOpenDate, "2026-11-19");
  assert.strictEqual(parsed.nyDate, "2026-11-19");
  assert.strictEqual(parsed.isWeekday, true);
  assert.ok(!out.includes("ghp_top_secret_value_XYZ"));
  assert.ok(!out.includes("AAGCqTKB"));
}

// 4. /scan, /fibo, /winrate, /backtest + negative formatting & OPTIONS regression
{
  const realBt = JSON.parse(fs.readFileSync(btPath, "utf8"));
  const realScan = JSON.parse(fs.readFileSync(scanPath, "utf8"));

  // /scan failure reports HTTP error
  const sScanFail = makeSandbox({ props: { GH_TOKEN: "tok" }, ghCode: 422, ghBody: "Workflow error" });
  sScanFail.ctx.doPost({
    parameter: { market: "DAY_OPTION" },
    postData: { contents: JSON.stringify({ update_id: 1, message: { text: "/scan", chat: { id: "8964251574" } } }) },
  });
  assert.ok(sScanFail.tgCalls[0].payload.text.includes("422"));

  // /fibo against real latest_scan.json: NVDA BLACKOUT & AMD +delta
  const sCmd = makeSandbox({
    props: { GH_TOKEN: "tok" },
    ghContents: (url) => {
      if (url.includes("latest_scan.json")) return realScan;
      if (url.includes("backtest_90d.json")) return { overall_golden: { win_rate_pct: 55.0, wins: 5, losses: 5, net_pnl: -120.5, profit_factor: 0.9 } };
      if (url.includes("paper_trades.json")) return {
        open_positions: [{ ticker: "SPY" }],
        closed_positions: [
          { ticker: "AMD", fibo_level: "0.618", net_pnl_usd: -125.5 },
          { ticker: "SPY", fibo_level: "0.500", net_pnl_usd: 50.0 },
        ],
      };
      return {};
    },
  });

  // /fibo NVDA -> BLACKOUT tag
  sCmd.ctx.doPost({
    parameter: { market: "DAY_OPTION" },
    postData: { contents: JSON.stringify({ update_id: 2, message: { text: "/fibo NVDA", chat: { id: "8964251574" } } }) },
  });
  assert.ok(sCmd.tgCalls[0].payload.text.includes("BLACKOUT"));

  // /winrate -> -$75.50 (never $-75.50), Fibo 0.618 & 0.500, per-ticker
  sCmd.ctx.doPost({
    parameter: { market: "DAY_OPTION" },
    postData: { contents: JSON.stringify({ update_id: 3, message: { text: "/winrate", chat: { id: "8964251574" } } }) },
  });
  const wrMsg = sCmd.tgCalls[1].payload.text;
  assert.ok(wrMsg.includes("-$75.50"));
  assert.ok(!wrMsg.includes("$-75.50"));
  assert.ok(wrMsg.includes("Fibo 0.618") && wrMsg.includes("Fibo 0.500") && wrMsg.includes("AMD"));

  // /backtest -> -$120.50 (never +$-120.50) & real PF check
  sCmd.ctx.doPost({
    parameter: { market: "DAY_OPTION" },
    postData: { contents: JSON.stringify({ update_id: 4, message: { text: "/backtest", chat: { id: "8964251574" } } }) },
  });
  const btMsg = sCmd.tgCalls[2].payload.text;
  assert.ok(btMsg.includes("-$120.50"));
  assert.ok(!btMsg.includes("+$-"));

  // Non-DAY_OPTION regression: ?market=OPTIONS /scan still triggers workflow via triggerGitHubWorkflow
  const sOpt = makeSandbox({ props: { GH_TOKEN: "tok" }, ghCode: 204 });
  sOpt.ctx.doPost({
    parameter: { market: "OPTIONS" },
    postData: { contents: JSON.stringify({ update_id: 5, message: { text: "/scan", chat: { id: "8964251574" } } }) },
  });
  assert.strictEqual(sOpt.ghCalls.length, 1);
  assert.ok(sOpt.ghCalls[0].url.includes("Webull_Options_Bot"));

  // /backtest catch fallback uses 1.58 matching backtest_90d.json
  assert.strictEqual(realBt.overall_golden.profit_factor, 1.58);
  const sBtErr = makeSandbox({ props: { GH_TOKEN: "tok" } });
  sBtErr.ctx.UrlFetchApp.fetch = (url, params) => {
    if (url.includes("api.telegram.org")) {
      sBtErr.tgCalls.push({ payload: JSON.parse(params.payload) });
      return { getResponseCode: () => 200, getContentText: () => "{}" };
    }
    throw new Error("network");
  };
  sBtErr.ctx.doPost({
    parameter: { market: "DAY_OPTION" },
    postData: { contents: JSON.stringify({ update_id: 6, message: { text: "/backtest", chat: { id: "8964251574" } } }) },
  });
  assert.ok(sBtErr.tgCalls[0].payload.text.includes("1.58"));
}
console.log("GAS_VM_TESTS_OK");
"""
    env = os.environ.copy()
    env["GAS_CODE_PATH"] = GAS_CODE_PATH
    env["BT_JSON_PATH"] = backtest_json_path
    env["SCAN_JSON_PATH"] = latest_scan_path

    proc = subprocess.run(
        ["node", "-e", node_test_script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Node GAS VM test failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "GAS_VM_TESTS_OK" in proc.stdout
