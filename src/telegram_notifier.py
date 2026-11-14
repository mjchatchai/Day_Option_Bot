"""
Telegram Notification Dispatcher for Day Option Bot (Room: Day Option).
Ultra-concise format focused strictly on Recommended Option Contracts, Entry Limit Price, and TP/SL.
"""

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv
import requests


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv(os.path.join(ROOT_DIR, ".env"))


def get_telegram_credentials() -> tuple[str, str]:
    token = (
        os.getenv("DAY_OPTION_TELEGRAM_BOT_TOKEN")
        or os.getenv("TELEGRAM_BOT_TOKEN")
        or ""
    ).strip()
    chat_id = (
        os.getenv("DAY_OPTION_TELEGRAM_CHAT_ID")
        or os.getenv("TELEGRAM_CHAT_ID")
        or ""
    ).strip()
    return token, chat_id


def send_telegram_message(text: str, token: Optional[str] = None, chat_id: Optional[str] = None) -> bool:
    tk, cid = token, chat_id
    if not tk or not cid:
        tk, cid = get_telegram_credentials()
    if not tk or not cid or tk.startswith("your_"):
        print("[Telegram] Skipping send: DAY_OPTION_TELEGRAM_BOT_TOKEN / CHAT_ID not configured.")
        return False

    url = f"https://api.telegram.org/bot{tk}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={
                "chat_id": cid,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=15,
        )
        return resp.status_code == 200
    except Exception as e:
        print(f"[Telegram] Error sending message: {e}")
        return False


def format_signal_card(item: Dict) -> str:
    """Concise 4-line Option Recommendation Card."""
    tk = item["ticker"]
    act = item["option_action"]
    emoji = "📉" if "PUT" in act else "📈"
    plan = item["trade_plan"]
    oq = item["option_quote"]
    status_tag = "🔥 เข้าทันที" if item.get("status") == "ACTIONABLE" else f"👀 รอแตะ ${plan['fibo_entry_price']:.2f}"

    lines = [
        f"🎫 <b>{tk} — {act} {emoji}</b> [{status_tag} | Fibo {plan['fibo_entry_level']}]",
        f"• <b>สัญญา:</b> <code>{tk} {oq['expiration']} ${oq['strike']:.2f} {oq['option_type']}</code> ({oq['dte']} DTE | Δ {oq['delta']:+.2f})",
        f"• <b>ราคาเข้า (Real-Time):</b> <code>~${oq['realtime_theo_price']:.2f}</code> (Limit ไม่เกิน <code>${oq['max_limit_buy']:.2f}</code>)",
        f"• ✅ <b>TP1:</b> <code>${oq['tp1_option_price']:.2f}</code> ({oq['tp1_gain_pct']:+.1f}%) | 🚀 <b>TP2:</b> <code>${oq['tp2_option_price']:.2f}</code> ({oq['tp2_gain_pct']:+.1f}%)",
        f"• 🛑 <b>SL:</b> <code>${oq['sl_option_price']:.2f}</code> ({oq['sl_loss_pct']:+.1f}%) <i>(คัทเมื่อหุ้นทะลุ ${plan['sl_underlying']:.2f})</i>",
    ]
    return "\n".join(lines)


def build_scan_telegram_report(scan_output: Dict, paper_summary: Dict) -> str:
    """Concise Telegram report showing only recommended Option contracts."""
    kpis = paper_summary.get("kpis", {})
    results: List[Dict] = scan_output.get("results", [])

    actionable = [r for r in results if r.get("status") == "ACTIONABLE"]
    near_setup = [r for r in results if r.get("status") == "NEAR_SETUP"]
    picks = actionable + near_setup

    # If none are actionable/near yet, show top 3 closest valid option setups
    if not picks:
        valid = [r for r in results if "option_quote" in r and r.get("status") == "WATCHING"]
        valid.sort(key=lambda x: x.get("distance_to_active_fibo_pct", 99.0))
        picks = valid[:3]

    msg = [
        f"⚡ <b>DAY OPTION — สัญญาแนะนำ ({scan_output.get('timestamp')})</b>",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    for item in picks:
        msg.append(format_signal_card(item))
        msg.append("")

    msg.append(
        f"🏆 <b>Paper Win Rate:</b> <code>{kpis.get('win_rate_pct', 0.0):.1f}%</code> "
        f"({kpis.get('wins', 0)}W/{kpis.get('losses', 0)}L | Net <code>${kpis.get('net_pnl_usd', 0.0):+,.2f}</code>)"
    )
    return "\n".join(msg)
