"""
Interactive Telegram Command Handler for Day Option Bot.
Supports commands in the Day Option Telegram room:
  /scan      - Run live EMA 200 + Fibo 0.500/0.618 scan immediately
  /fibo <TK> - Show detailed Previous-Day Fibonacci map & Real-Time Option Quote for a ticker
  /winrate   - Show Paper Trade Win Rate & Per-Ticker Leaderboard
  /backtest  - Show 90-Day Backtest Summary (81,900 bars)
  /help      - List available commands
"""

import html
import json
import os
from typing import Dict, Optional
import requests

from .fibo_scanner import analyze_single_ticker, run_full_day_option_scan
from .paper_trader import PAPER_STATE_PATH, compute_paper_kpis, load_paper_state, update_paper_portfolio
from .telegram_notifier import (
    BASE_DIR,
    build_scan_telegram_report,
    format_signal_card,
    get_telegram_credentials,
    send_telegram_message,
)


OFFSET_FILE = os.path.join(BASE_DIR, "config", "telegram_offset.txt")
BACKTEST_FILE = os.path.join(BASE_DIR, "config", "backtest_90d.json")
_WEBHOOK_CACHE: Dict[str, bool] = {}


def has_active_telegram_webhook(token: Optional[str] = None, force_refresh: bool = False) -> bool:
    """Check Telegram getWebhookInfo safely (cached per process/token) to avoid getUpdates 409 conflicts."""
    env_tk, _ = get_telegram_credentials()
    tk = (token if token is not None else env_tk).strip()
    if not tk or tk.lower().startswith("your_"):
        return False

    if not force_refresh and tk in _WEBHOOK_CACHE:
        return _WEBHOOK_CACHE[tk]

    url = f"https://api.telegram.org/bot{tk}/getWebhookInfo"
    try:
        resp = requests.get(url, timeout=6)
        if resp.status_code != 200:
            print(f"[Telegram] getWebhookInfo failed (HTTP {resp.status_code}): {getattr(resp, 'text', '')}")
            _WEBHOOK_CACHE[tk] = False
            return False
        data = resp.json()
        result = data.get("result") if isinstance(data, dict) else None
        webhook_url = str(result.get("url") or "").strip() if isinstance(result, dict) else ""
        active = bool(webhook_url)
        _WEBHOOK_CACHE[tk] = active
        return active
    except Exception as e:
        print(f"[Telegram] getWebhookInfo error: {e}")
        return False


def _send_or_warn(text: str, token: str, chat_id: str, context: str) -> None:
    if not send_telegram_message(text, token, chat_id):
        print(f"[Telegram] WARNING: Failed to send response for {context}.")


def _load_offset() -> int:
    if os.path.exists(OFFSET_FILE):
        try:
            with open(OFFSET_FILE, "r", encoding="utf-8") as f:
                return int(f.read().strip())
        except Exception:
            pass
    return 0


def _save_offset(offset: int) -> None:
    os.makedirs(os.path.dirname(OFFSET_FILE), exist_ok=True)
    with open(OFFSET_FILE, "w", encoding="utf-8") as f:
        f.write(str(offset))


def format_winrate_message(config: Dict) -> str:
    state = load_paper_state(PAPER_STATE_PATH)
    start_bal = float(config.get("paper_trading", {}).get("starting_balance_usd", 5000.0))
    kpis = compute_paper_kpis(state.get("closed_positions", []), state.get("open_positions", []), start_bal)

    lines = [
        "🏆 <b>DAY OPTION — PAPER TRADE WIN RATE REPORT</b>",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"• <b>จำนวนไม้ที่ปิดแล้ว:</b> <code>{kpis['total_closed']}</code> ไม้ ({kpis['wins']} ชนะ / {kpis['losses']} แพ้)",
        f"• <b>อัตราชนะรวม (Win Rate):</b> <code>{kpis['win_rate_pct']:.1f}%</code>",
        f"• <b>กำไรสุทธิสะสม (Net PnL):</b> <code>${kpis['net_pnl_usd']:+,.2f}</code> (พอร์ต <code>${kpis['current_equity_usd']:,.2f}</code>)",
        f"• <b>Profit Factor:</b> <code>{kpis['profit_factor']:.2f}</code> | <b>Realized R:R:</b> <code>{kpis['realized_rr']:.2f}</code>",
        "",
        "📊 <b>แยกตามระดับเส้น Fibonacci:</b>",
        f"• <b>Fibo 0.618 (Golden Zone):</b> <code>{kpis['fibo_0618']['win_rate_pct']:.1f}%</code> ({kpis['fibo_0618']['count']} ไม้ | <code>${kpis['fibo_0618']['net_pnl_usd']:+,.2f}</code>)",
        f"• <b>Fibo 0.500 (Mid Zone):</b> <code>{kpis['fibo_0500']['win_rate_pct']:.1f}%</code> ({kpis['fibo_0500']['count']} ไม้ | <code>${kpis['fibo_0500']['net_pnl_usd']:+,.2f}</code>)",
    ]

    per_tk = kpis.get("per_ticker", {})
    if per_tk:
        lines.append("\n📈 <b>สถิติแยกตามรายหุ้น:</b>")
        for tk, st in sorted(per_tk.items(), key=lambda x: -x[1]["net_pnl_usd"]):
            esc_tk = html.escape(str(tk), quote=False)
            lines.append(
                f"• <b>{esc_tk}:</b> WR <code>{st['win_rate_pct']:.1f}%</code> "
                f"({st['wins']}W/{st['losses']}L) | Net <code>${st['net_pnl_usd']:+,.2f}</code>"
            )

    return "\n".join(lines)


def format_backtest_summary_message() -> str:
    if not os.path.exists(BACKTEST_FILE):
        return "⚠️ ยังไม่มีไฟล์ผลลัพธ์ Backtest 90 วัน กรุณารันโหมด <code>--backtest</code> ก่อนครับ"
    with open(BACKTEST_FILE, "r", encoding="utf-8") as f:
        bt = json.load(f)

    gold = bt.get("overall_golden", {})
    orig = bt.get("overall_original", {})
    lines = [
        "📊 <b>ผลทดสอบ BACKTEST ย้อนหลัง 90 วัน (60 วันทำการ)</b>",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"• <b>สูตรดั้งเดิม (Original):</b> WR <code>{orig.get('win_rate_pct', 0):.1f}%</code> | Net <code>${orig.get('net_pnl', 0):+,.2f}</code>",
        f"• <b>สูตรคัดเส้นทองคำ (Golden Smart Fibo):</b> WR <code>{gold.get('win_rate_pct', 0):.1f}%</code> | Net <code>${gold.get('net_pnl', 0):+,.2f}</code> (PF <code>{gold.get('profit_factor', 0):.2f}</code>)",
        "\n🏆 <b>ผลงานรายหุ้น (Golden Mode 90 วัน):</b>",
    ]
    for tk, info in bt.get("per_ticker", {}).items():
        g = info.get("golden", {})
        esc_tk = html.escape(str(tk), quote=False)
        lines.append(
            f"• <b>{esc_tk}:</b> WR <code>{g.get('win_rate_pct', 0):.1f}%</code> "
            f"({g.get('wins', 0)}W/{g.get('losses', 0)}L) | Net <code>${g.get('net_pnl', 0):+,.2f}</code>"
        )
    return "\n".join(lines)


def poll_telegram_commands(config: Dict) -> None:
    token, chat_id = get_telegram_credentials()
    if (
        not token
        or not chat_id
        or token.lower().startswith("your_")
        or chat_id.lower().startswith("your_")
    ):
        return

    offset = _load_offset()
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        resp = requests.get(url, params={"offset": offset + 1, "timeout": 2}, timeout=6)
        if resp.status_code != 200:
            print(f"[Telegram] getUpdates failed (HTTP {resp.status_code}): {getattr(resp, 'text', '')}")
            return
        updates = resp.json().get("result", [])
    except Exception as e:
        print(f"[Telegram] getUpdates error: {e}")
        return

    for upd in updates:
        upd_id = int(upd.get("update_id", 0))
        _save_offset(upd_id)

        msg_obj = upd.get("message") or {}
        text = (msg_obj.get("text") or "").strip()
        if not text.startswith("/"):
            continue

        parts = text.split()
        cmd = parts[0].lower().split("@")[0]

        if cmd == "/scan":
            scan_out = run_full_day_option_scan(config)
            paper_out = update_paper_portfolio(scan_out, config)
            _send_or_warn(build_scan_telegram_report(scan_out, paper_out), token, chat_id, "/scan")
        elif cmd == "/fibo":
            tk = parts[1].upper() if len(parts) > 1 else "AMD"
            esc_tk = html.escape(tk, quote=False)
            item = analyze_single_ticker(tk, config)
            st = item.get("status")
            if st == "NO_DATA":
                _send_or_warn(f"⚠️ ไม่พบข้อมูลของ {esc_tk}", token, chat_id, "/fibo")
            elif st == "NEUTRAL" or not item.get("trade_plan") or not item.get("option_quote"):
                _send_or_warn(format_signal_card(item), token, chat_id, "/fibo")
            else:
                _send_or_warn(format_signal_card(item), token, chat_id, "/fibo")
        elif cmd in ("/winrate", "/paper"):
            _send_or_warn(format_winrate_message(config), token, chat_id, cmd)
        elif cmd == "/backtest":
            _send_or_warn(format_backtest_summary_message(), token, chat_id, "/backtest")
        elif cmd == "/help":
            help_txt = (
                "🤖 <b>คำสั่งบอท DAY OPTION:</b>\n"
                "• <code>/scan</code> — สแกนหาจุดเข้า Fibo 0.500 / 0.618 สดทันที\n"
                "• <code>/fibo AMD</code> — ดูแผนที่เส้น Fibo และราคา Option สดของหุ้นรายตัว\n"
                "• <code>/winrate</code> — ดูสถิติอัตราชนะ (Paper Trade Win Rate)\n"
                "• <code>/backtest</code> — ดูสรุปผล Backtest ย้อนหลัง 90 วัน"
            )
            _send_or_warn(help_txt, token, chat_id, "/help")
