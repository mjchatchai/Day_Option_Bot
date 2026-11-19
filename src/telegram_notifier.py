"""
Telegram Notification Dispatcher for Day Option Bot (Room: Day Option).
Ultra-concise format focused strictly on Recommended Option Contracts, Entry Limit Price, and TP/SL.
"""

import html
import os
import re
import time
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import requests


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

_HTML_TAG_RE = re.compile(r"</?([a-zA-Z][a-zA-Z0-9-]*)(?:\s+[^>]*)?>")


def _esc(val: object) -> str:
    return html.escape(str(val if val is not None else ""), quote=False)


def get_telegram_credentials(allow_fallback: bool = False) -> Tuple[str, str]:
    token = (os.getenv("DAY_OPTION_TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (os.getenv("DAY_OPTION_TELEGRAM_CHAT_ID") or "").strip()

    env_allow = (os.getenv("DAY_OPTION_ALLOW_TELEGRAM_FALLBACK") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    use_fallback = allow_fallback or env_allow

    if use_fallback:
        used_fallback = False
        if not token:
            fb_token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
            if fb_token:
                token = fb_token
                used_fallback = True
        if not chat_id:
            fb_chat = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
            if fb_chat:
                chat_id = fb_chat
                used_fallback = True
        if used_fallback:
            print("[Telegram] WARNING: Using fallback TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID because DAY_OPTION_* is unset.")

    return token, chat_id


def split_telegram_html(text: str, max_len: int = 4096) -> List[str]:
    """Split HTML message into chunks <= max_len without breaking <tags>, &entities;, or open tags."""
    if not text:
        return [""]
    if len(text) <= max_len:
        return [text]

    tokens: List[Tuple[str, str]] = []
    pos = 0
    for m in _HTML_TAG_RE.finditer(text):
        if m.start() > pos:
            tokens.append(("text", text[pos : m.start()]))
        tokens.append(("tag", m.group(0)))
        pos = m.end()
    if pos < len(text):
        tokens.append(("text", text[pos:]))

    chunks: List[str] = []
    stack: List[Tuple[str, str]] = []
    buf = ""

    def _closing_suffix(st: List[Tuple[str, str]]) -> str:
        return "".join(f"</{name}>" for name, _ in reversed(st))

    def _opening_prefix(st: List[Tuple[str, str]]) -> str:
        return "".join(open_tag for _, open_tag in st)

    def _flush() -> None:
        nonlocal buf
        if buf:
            chunks.append(buf + _closing_suffix(stack))
        buf = _opening_prefix(stack)

    def _safe_text_split_idx(seg: str, limit: int) -> int:
        if len(seg) <= limit:
            return len(seg)
        if limit <= 0:
            return 0
        window = seg[:limit]
        for sep in ("\n\n", "\n", " "):
            idx = window.rfind(sep)
            if idx > 0:
                return idx + (1 if sep == " " else len(sep))
        cut = limit
        amp = window.rfind("&")
        if amp != -1 and ";" not in window[amp:] and (limit - amp) <= 12:
            if amp > 0:
                cut = amp
        return max(1, cut)

    for kind, tok in tokens:
        if kind == "tag":
            is_close = tok.startswith("</")
            is_self_close = tok.endswith("/>")
            m_tag = _HTML_TAG_RE.match(tok)
            tag_name = m_tag.group(1).lower() if m_tag else ""

            if is_close:
                new_stack = list(stack)
                for i in range(len(new_stack) - 1, -1, -1):
                    if new_stack[i][0] == tag_name:
                        new_stack.pop(i)
                        break
                projected_len = len(buf) + len(tok) + len(_closing_suffix(new_stack))
                if projected_len > max_len and buf:
                    _flush()
                buf += tok
                stack = new_stack
            else:
                new_stack = list(stack)
                if not is_self_close and tag_name:
                    new_stack.append((tag_name, tok))
                projected_len = len(buf) + len(tok) + len(_closing_suffix(new_stack))
                if projected_len > max_len and buf:
                    _flush()
                buf += tok
                stack = new_stack
        else:
            remaining = tok
            while remaining:
                avail = max_len - len(buf) - len(_closing_suffix(stack))
                if avail <= 0:
                    if buf and buf != _opening_prefix(stack):
                        _flush()
                        avail = max_len - len(buf) - len(_closing_suffix(stack))
                    if avail <= 0:
                        # Extreme edge case where max_len is smaller than tag wrapper overhead
                        if buf:
                            _flush()
                        buf = ""
                        stack = []
                        avail = max_len
                if len(remaining) <= avail:
                    buf += remaining
                    remaining = ""
                else:
                    cut = _safe_text_split_idx(remaining, avail)
                    part = remaining[:cut]
                    buf += part
                    remaining = remaining[cut:]
                    _flush()

    if buf and (buf != _opening_prefix(stack) or not chunks):
        chunks.append(buf + _closing_suffix(stack))

    return [c for c in chunks if c] or [""]


def send_telegram_message(
    text: str,
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
    max_retries: int = 3,
    backoff_base: float = 0.5,
) -> bool:
    env_tk, env_cid = get_telegram_credentials()
    tk = (token if token is not None else env_tk).strip()
    cid = (chat_id if chat_id is not None else env_cid).strip()

    if (
        not tk
        or not cid
        or tk.lower().startswith("your_")
        or cid.lower().startswith("your_")
    ):
        print("[Telegram] Skipping send: DAY_OPTION_TELEGRAM_BOT_TOKEN / CHAT_ID not configured.")
        return False

    url = f"https://api.telegram.org/bot{tk}/sendMessage"
    chunks = split_telegram_html(str(text or ""), max_len=4096)
    attempts = max(1, int(max_retries))

    for chunk in chunks:
        chunk_sent = False
        for attempt in range(attempts):
            try:
                resp = requests.post(
                    url,
                    json={
                        "chat_id": cid,
                        "text": chunk,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                    timeout=15,
                )
                if resp.status_code == 200:
                    ok_flag = True
                    try:
                        data = resp.json()
                        if isinstance(data, dict) and data.get("ok") is False:
                            ok_flag = False
                    except Exception:
                        pass
                    if ok_flag:
                        chunk_sent = True
                        break
                    print(f"[Telegram] API returned ok=False (HTTP 200): {getattr(resp, 'text', '')}")
                    return False

                resp_text = getattr(resp, "text", "")
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    print(
                        f"[Telegram] Send failed (HTTP {resp.status_code}, "
                        f"attempt {attempt + 1}/{attempts}): {resp_text}"
                    )
                    if attempt < attempts - 1:
                        delay = backoff_base * (2 ** attempt)
                        if resp.status_code == 429:
                            try:
                                ra = resp.json().get("parameters", {}).get("retry_after")
                                if ra is not None:
                                    delay = min(float(ra), 10.0)
                            except Exception:
                                pass
                        time.sleep(delay)
                        continue
                    return False

                print(f"[Telegram] Send failed (HTTP {resp.status_code}): {resp_text}")
                return False
            except Exception as e:
                print(f"[Telegram] Error sending message (attempt {attempt + 1}/{attempts}): {e}")
                if attempt < attempts - 1:
                    time.sleep(backoff_base * (2 ** attempt))
                    continue
                return False

        if not chunk_sent:
            return False

    return True


def format_signal_card(item: Dict) -> str:
    """Concise 5-line Option Recommendation Card."""
    tk = _esc(item.get("ticker", "UNKNOWN"))
    status = str(item.get("status", "UNKNOWN"))
    plan = item.get("trade_plan")
    oq = item.get("option_quote")

    if status in ("NO_DATA", "NEUTRAL") or not plan or not oq:
        if status == "NO_DATA":
            reason = _esc(item.get("reason") or "ไม่มีข้อมูลเพียงพอ")
            return f"⚠️ <b>{tk} — NO_DATA</b>: {reason}"
        if status == "NEUTRAL":
            spot = item.get("spot_price")
            ema = item.get("ema200")
            if isinstance(spot, (int, float)) and isinstance(ema, (int, float)):
                return (
                    f"⚖️ <b>{tk} — NEUTRAL</b> (Spot <code>${spot:.2f}</code> "
                    f"≈ EMA200 <code>${ema:.2f}</code> — รอเลือกทิศทาง)"
                )
            note = _esc(item.get("status_thai") or item.get("reason") or "ตลาดทรงตัว ยังไม่มีทิศทางชัดเจน")
            return f"⚖️ <b>{tk} — NEUTRAL</b>: {note}"
        st_esc = _esc(status)
        note = _esc(item.get("status_thai") or item.get("reason") or "ยังไม่มีสัญญาณ Option ที่พร้อมเทรด")
        return f"ℹ️ <b>{tk} [{st_esc}]</b> — {note}"

    raw_act = str(item.get("option_action", ""))
    act = _esc(raw_act)
    emoji = "📉" if "PUT" in raw_act else "📈"
    if status == "ACTIONABLE":
        status_tag = "🔥 เข้าทันที"
    elif status == "EARNINGS_BLACKOUT":
        status_tag = "🚫 งดเทรดใกล้งบ"
    elif status == "GAP_SKIP":
        status_tag = "⏭️ GAP_SKIP"
    else:
        status_tag = f"👀 รอแตะ ${float(plan['fibo_entry_price']):.2f}"

    fibo_lvl = _esc(plan["fibo_entry_level"])
    exp = _esc(oq["expiration"])
    opt_type = _esc(oq["option_type"])

    lines = [
        f"🎫 <b>{tk} — {act} {emoji}</b> [{status_tag} | Fibo {fibo_lvl}]",
        f"• <b>สัญญา:</b> <code>{tk} {exp} ${float(oq['strike']):.2f} {opt_type}</code> ({int(oq['dte'])} DTE | Δ {float(oq['delta']):+.2f})",
        f"• <b>ราคาเข้า (Real-Time):</b> <code>~${float(oq['realtime_theo_price']):.2f}</code> (Limit ไม่เกิน <code>${float(oq['max_limit_buy']):.2f}</code>)",
        f"• ✅ <b>TP1:</b> <code>${float(oq['tp1_option_price']):.2f}</code> ({float(oq['tp1_gain_pct']):+.1f}%) | 🚀 <b>TP2:</b> <code>${float(oq['tp2_option_price']):.2f}</code> ({float(oq['tp2_gain_pct']):+.1f}%)",
        f"• 🛑 <b>SL:</b> <code>${float(oq['sl_option_price']):.2f}</code> ({float(oq['sl_loss_pct']):+.1f}%) <i>(คัทเมื่อหุ้นทะลุ ${float(plan['sl_underlying']):.2f})</i>",
    ]
    return "\n".join(lines)


def build_scan_telegram_report(scan_output: Dict, paper_summary: Dict) -> str:
    """Concise Telegram report showing only recommended Option contracts."""
    kpis = paper_summary.get("kpis", {})
    results: List[Dict] = scan_output.get("results", [])

    def _has_valid_quote(r: Dict) -> bool:
        return bool(r.get("option_quote")) and bool(r.get("trade_plan"))

    actionable = [r for r in results if r.get("status") == "ACTIONABLE" and _has_valid_quote(r)]
    near_setup = [r for r in results if r.get("status") == "NEAR_SETUP" and _has_valid_quote(r)]
    picks = actionable + near_setup

    # If none are actionable/near yet, show top 3 closest valid option setups
    if not picks:
        valid = [r for r in results if _has_valid_quote(r) and r.get("status") == "WATCHING"]
        valid.sort(key=lambda x: x.get("distance_to_active_fibo_pct", 99.0))
        picks = valid[:3]

    ts_esc = _esc(scan_output.get("timestamp", ""))
    msg = [
        f"⚡ <b>DAY OPTION — สัญญาแนะนำ ({ts_esc})</b>",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if picks:
        for item in picks:
            msg.append(format_signal_card(item))
            msg.append("")
    else:
        msg.append("ℹ️ <b>ไม่มีสัญญาณพร้อมเทรดในรอบนี้</b>")
        for r in results:
            r_tk = _esc(r.get("ticker", "UNKNOWN"))
            r_st = _esc(r.get("status", "UNKNOWN"))
            spot = r.get("spot_price")
            spot_str = f" (${float(spot):.2f})" if isinstance(spot, (int, float)) else ""
            reason = r.get("reason")
            reason_str = f" — {_esc(reason)}" if reason else ""
            msg.append(f"• <b>{r_tk}</b>: <code>{r_st}</code>{spot_str}{reason_str}")
        msg.append("")

    msg.append(
        f"🏆 <b>Paper Win Rate:</b> <code>{float(kpis.get('win_rate_pct', 0.0)):.1f}%</code> "
        f"({int(kpis.get('wins', 0))}W/{int(kpis.get('losses', 0))}L | Net <code>${float(kpis.get('net_pnl_usd', 0.0)):+,.2f}</code>)"
    )
    return "\n".join(msg)
