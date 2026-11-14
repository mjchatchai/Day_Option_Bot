"""
Telegram Notification Dispatcher for Day Option Bot (Room: Day Option).
Formats rich Thai Tree-Branch reports with:
- Real-Time Underlying Trigger Price + EMA 200 Trend + Volume RVOL
- Previous-Day Asymmetric Fibonacci Anchors & Entry Level (0.500 / 0.618)
- Real-Time Black-Scholes Option Pricing (Zero 15-min Delay) + Max Limit Buy
- TP1 (Next Fibo Line + Lock Profit), TP2 (Runner), and SL (Opposite Fibo Line)
- Live Paper Trade Win Rate Scoreboard
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
    tk = item["ticker"]
    act = item["option_action"]
    emoji = "📉" if "PUT" in act else "📈"
    spot = item["spot_price"]
    ema = item["ema200"]
    rvol = item["rvol"]
    fibo = item["fibo_levels"]
    plan = item["trade_plan"]
    oq = item["option_quote"]
    candle = item["candle_reversal"]

    lines = [
        f"🚨 <b>[{tk}] {act} {emoji} — Fibo {plan['fibo_entry_level']} ({item['status']})</b>",
        f"├ <b>ราคาหุ้นแม่ (Real-Time):</b> <code>${spot:.2f}</code> | <b>EMA200:</b> <code>${ema:.2f}</code>",
        f"├ <b>สถานะเทรนด์:</b> {item['trend_thai']}",
        f"├ <b>Volume (RVOL):</b> <code>{rvol:.2f}x</code> | <b>แท่งเทียน:</b> {candle['pattern_name']}",
        f"├ <b>กรอบ Fibo เมื่อวาน:</b> จุด 1 {fibo['anchor_point_1_label']} → จุด 2 {fibo['anchor_point_2_label']}",
        f"├ <b>จุดเข้า (Fibo {plan['fibo_entry_level']}):</b> <code>${plan['fibo_entry_price']:.2f}</code> (ห่าง <code>{item['distance_to_active_fibo_pct']:.2f}%</code>)",
        f"│",
        f"🎫 <b>สัญญา Option แนะนำ (คำนวณสด Black-Scholes ไร้ดีเลย์ 15 นาที):</b>",
        f"├ <b>Contract:</b> <code>{tk} {oq['expiration']} ${oq['strike']:.2f} {oq['option_type']}</code> ({oq['dte']} DTE | Δ <code>{oq['delta']:+.2f}</code>)",
        f"├ <b>ราคาประเมินสด (Real-Time):</b> <code>~${oq['realtime_theo_price']:.2f}</code> (Bid <code>${oq['est_bid']:.2f}</code> / Ask <code>${oq['est_ask']:.2f}</code>)",
        f"├ ⚠️ <b>ตั้ง Limit Buy ไม่เกิน:</b> <code>${oq['max_limit_buy']:.2f}</code> ({oq['recommended_contracts']} สัญญา)",
        f"│",
        f"🎯 <b>แผนทำกำไร &amp; ตัดขาดทุน (TP / SL อ้างอิงราคาหุ้นแม่ Real-Time):</b>",
        f"├ ✅ <b>TP 1 ({plan['tp1_label']}):</b> หุ้นแตะ <code>${plan['tp1_underlying']:.2f}</code> → Option <code>~${oq['tp1_option_price']:.2f}</code> (<b>{oq['tp1_gain_pct']:+.1f}%</b>) [ขยับ SL ล็อกกำไรที่ <code>${plan['lock_sl_after_tp1']:.2f}</code>]",
        f"├ 🚀 <b>TP 2 ({plan['tp2_label']}):</b> หุ้นแตะ <code>${plan['tp2_underlying']:.2f}</code> → Option <code>~${oq['tp2_option_price']:.2f}</code> (<b>{oq['tp2_gain_pct']:+.1f}%</b>)",
        f"└ 🛑 <b>SL ({plan['sl_label']}):</b> คัทถ้าหุ้นทะลุ <code>${plan['sl_underlying']:.2f}</code> → Option <code>~${oq['sl_option_price']:.2f}</code> (<b>{oq['sl_loss_pct']:+.1f}%</b>)",
    ]
    return "\n".join(lines)


def build_scan_telegram_report(scan_output: Dict, paper_summary: Dict) -> str:
    kpis = paper_summary.get("kpis", {})
    results: List[Dict] = scan_output.get("results", [])

    actionable = [r for r in results if r.get("status") == "ACTIONABLE"]
    near_setup = [r for r in results if r.get("status") == "NEAR_SETUP"]
    watching = [r for r in results if r.get("status") == "WATCHING"]
    blackouts = [r for r in results if r.get("status") == "EARNINGS_BLACKOUT"]

    msg = [
        "⚡ <b>DAY OPTION BOT — REAL-TIME FIBO &amp; EMA 200 REPORT</b>",
        f"🕒 อัปเดต: <code>{scan_output.get('timestamp')}</code>",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if actionable:
        msg.append(f"🔥 <b>สัญญาณเข้าเทรดทันที (ACTIONABLE: {len(actionable)} ตัว)</b>\n")
        for item in actionable:
            msg.append(format_signal_card(item))
            msg.append("")
    else:
        msg.append("⏳ <b>ยังไม่มีสัญญาณแท่งกลับตัวพร้อม Volume ณ วินาทีนี้</b>")

    if near_setup:
        msg.append(f"👀 <b>หุ้นเข้าใกล้โซน Fibo 0.500 / 0.618 เตรียมพร้อม ({len(near_setup)} ตัว):</b>")
        for item in near_setup:
            msg.append(format_signal_card(item))
            msg.append("")

    if watching:
        msg.append("📋 <b>สรุปแนวรับ-แนวต้าน Fibonacci รายวันของหุ้นใน Watchlist:</b>")
        for r in watching:
            if "fibo_levels" not in r:
                continue
            f = r["fibo_levels"]
            msg.append(
                f"• <b>{r['ticker']}</b> (<code>${r['spot_price']:.2f}</code> | {r['option_action']}) "
                f"→ Fibo 0.500: <code>${f['0.500']:.2f}</code> | <b>Fibo 0.618: <code>${f['0.618']:.2f}</code></b> "
                f"(ห่าง {r['distance_to_active_fibo_pct']:.2f}%)"
            )
        msg.append("")

    if blackouts:
        msg.append("🚫 <b>หุ้นที่งดเทรดเนื่องจากติดช่วงประกาศงบ (Earnings Filter):</b>")
        for b in blackouts:
            msg.append(f"• <b>{b['ticker']}</b>: {b['earnings']['status_label']}")
        msg.append("")

    # Paper Trade Win Rate Scoreboard
    f618 = kpis.get("fibo_0618", {})
    f500 = kpis.get("fibo_0500", {})
    msg.extend([
        "━━━━━━━━━━━━━━━━━━━━━━",
        "🏆 <b>สถิติ PAPER TRADE &amp; WIN RATE TRACKER</b>",
        f"├ <b>อัตราชนะรวม (Win Rate):</b> <code>{kpis.get('win_rate_pct', 0.0):.1f}%</code> "
        f"({kpis.get('wins', 0)}W / {kpis.get('losses', 0)}L | กำไรสุทธิ <code>${kpis.get('net_pnl_usd', 0.0):+,.2f}</code>)",
        f"├ <b>แยกตามเส้น Fibo 0.618:</b> Win Rate <code>{f618.get('win_rate_pct', 0.0):.1f}%</code> (<code>${f618.get('net_pnl_usd', 0.0):+,.2f}</code>)",
        f"├ <b>แยกตามเส้น Fibo 0.500:</b> Win Rate <code>{f500.get('win_rate_pct', 0.0):.1f}%</code> (<code>${f500.get('net_pnl_usd', 0.0):+,.2f}</code>)",
        f"└ <b>ออเดอร์ที่กำลังถืออยู่ (Open Paper Trades):</b> <code>{kpis.get('open_count', 0)}</code> สัญญา",
    ])

    return "\n".join(msg)
