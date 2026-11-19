"""
Main Entrypoint for Day Option Bot (Bot Option Daytrade).
Modes:
  python -m day_option_bot.src.main --once      (Single scan + Paper Trade update + HTML Dashboard + Telegram report)
  python -m day_option_bot.src.main --loop      (Continuous 30-second real-time monitor + Telegram commands)
  python -m day_option_bot.src.main --backtest  (Run 90-Day Intraday Backtest and update dashboard)
"""

import argparse
import html
import json
import os
import time
import traceback
from typing import Dict, Optional, Tuple

from .backtester import BACKTEST_OUT_PATH, run_90d_backtest
from .command_handler import has_active_telegram_webhook, poll_telegram_commands
from .dashboard_builder import build_html_dashboard
from .fibo_scanner import run_full_day_option_scan
from .paper_trader import flush_open_paper_positions_at_session_end, update_paper_portfolio
from .telegram_notifier import build_scan_telegram_report, format_signal_card, send_telegram_message
from .time_utils import format_ny_timestamp, now_ny


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.json")
LATEST_SCAN_PATH = os.path.join(BASE_DIR, "config", "latest_scan.json")


def _notify_or_warn(msg: str, context: str) -> bool:
    ok = send_telegram_message(msg)
    if not ok:
        print(f"[Day Option] WARNING: Telegram notification failed ({context}).")
    return ok


def load_config(path: str = CONFIG_PATH) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def execute_single_cycle(config: Dict, notify_telegram: bool = True, ny_session_date: str = "") -> Dict:
    print("[Day Option] Running real-time EMA 200 (TF 2m) + Previous-Day Fibonacci (0.500 / 0.618) scan...")
    scan_out = run_full_day_option_scan(config)
    if ny_session_date:
        scan_out["ny_session_date"] = ny_session_date

    os.makedirs(os.path.dirname(LATEST_SCAN_PATH), exist_ok=True)
    with open(LATEST_SCAN_PATH, "w", encoding="utf-8") as f:
        json.dump(scan_out, f, ensure_ascii=False, indent=2)

    paper_out = update_paper_portfolio(scan_out, config)
    dash_path = build_html_dashboard(scan_out, paper_out)

    print(
        f"[Day Option] Scan complete: {scan_out['total_scanned']} tickers | "
        f"Actionable: {scan_out['actionable_count']} | Near Setup: {scan_out['near_setup_count']} | "
        f"Paper Win Rate: {paper_out['kpis']['win_rate_pct']:.1f}%"
    )
    print(f"[Day Option] Dashboard generated at: {dash_path}")

    if notify_telegram:
        report_msg = build_scan_telegram_report(scan_out, paper_out)
        _notify_or_warn(report_msg, "execute_single_cycle")

    return {"scan": scan_out, "paper": paper_out, "dashboard": dash_path}


def run_continuous_30s_loop(
    config: Dict,
    interval_sec: int = 30,
    max_minutes: int = 0,
    send_opening_report: bool = False,
    ny_session_date: str = "",
    no_command_poll: bool = False,
    end_ny_hm: Optional[Tuple[int, int]] = (11, 30),
) -> None:
    print(
        f"[Day Option] Starting 30-second real-time loop "
        f"(interval={interval_sec}s, max_minutes={max_minutes or 'unlimited'}, "
        f"end_ny_hm={end_ny_hm})..."
    )
    if no_command_poll:
        poll_enabled = False
        print("[Day Option] Telegram getUpdates polling disabled (--no-command-poll).")
    elif has_active_telegram_webhook():
        poll_enabled = False
        print("[Day Option] Active Telegram webhook detected via getWebhookInfo; skipping getUpdates polling.")
    else:
        poll_enabled = True

    alerted_keys = set()
    start_ts = time.time()
    start_ny = now_ny()
    cutoff_dt = (
        start_ny.replace(hour=int(end_ny_hm[0]), minute=int(end_ny_hm[1]), second=0, microsecond=0)
        if end_ny_hm is not None
        else None
    )
    first_cycle = True
    success_cycles = 0
    failed_cycles = 0
    last_scan_out: Optional[Dict] = None

    def _stop_reason() -> Optional[str]:
        if max_minutes > 0 and (time.time() - start_ts) >= (max_minutes * 60):
            return f"2-hour window limit ({max_minutes} mins)"
        if cutoff_dt is not None and now_ny() >= cutoff_dt:
            return f"11:30 America/New_York cutoff ({cutoff_dt.strftime('%Y-%m-%d %H:%M:%S %Z')})"
        return None

    while True:
        reason = _stop_reason()
        if reason is not None:
            print(f"[Day Option] Reached {reason}. Running final MTM scan & closing remaining open paper trades...")
            final_scan: Optional[Dict] = None
            try:
                final_scan = run_full_day_option_scan(config)
                if ny_session_date:
                    final_scan["ny_session_date"] = ny_session_date
                os.makedirs(os.path.dirname(LATEST_SCAN_PATH), exist_ok=True)
                with open(LATEST_SCAN_PATH, "w", encoding="utf-8") as f:
                    json.dump(final_scan, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[Day Option] Final MTM scan failed, falling back to last scan: {e}")
                traceback.print_exc()
                final_scan = last_scan_out

            flush_res = flush_open_paper_positions_at_session_end(config, latest_scan=final_scan)
            dash_scan = final_scan if final_scan is not None else {
                "timestamp": format_ny_timestamp(),
                "ny_session_date": ny_session_date,
                "actionable_count": 0,
                "near_setup_count": 0,
                "results": [],
            }
            build_html_dashboard(dash_scan, flush_res)

            for closed in flush_res.get("flushed", []):
                outcome_emoji = "✅ WIN" if closed.get("is_win") else "🛑 LOSS"
                tk_esc = html.escape(str(closed.get("ticker", "")), quote=False)
                act_esc = html.escape(str(closed.get("option_action", "")), quote=False)
                msg = (
                    f"{outcome_emoji} <b>[ปิดจบ 2 ชม. แรก] {tk_esc} ({act_esc})</b>\n"
                    f"• <b>ราคา Option:</b> เข้า <code>${closed['entry_option_price']:.2f}</code> → ออก <code>${closed['exit_option_price']:.2f}</code>\n"
                    f"• <b>กำไร/ขาดทุนสุทธิ:</b> <code>${closed['net_pnl_usd']:+,.2f}</code>\n"
                    f"• <b>Paper Win Rate:</b> <code>{flush_res['kpis']['win_rate_pct']:.1f}%</code>"
                )
                _notify_or_warn(msg, f"session_end_flush:{closed.get('ticker', '')}")
            break

        try:
            if poll_enabled:
                poll_telegram_commands(config)
            scan_out = run_full_day_option_scan(config)
            if ny_session_date:
                scan_out["ny_session_date"] = ny_session_date
            os.makedirs(os.path.dirname(LATEST_SCAN_PATH), exist_ok=True)
            with open(LATEST_SCAN_PATH, "w", encoding="utf-8") as f:
                json.dump(scan_out, f, ensure_ascii=False, indent=2)
            last_scan_out = scan_out

            paper_out = update_paper_portfolio(scan_out, config)
            build_html_dashboard(scan_out, paper_out)
            success_cycles += 1

            if first_cycle and send_opening_report:
                _notify_or_warn(build_scan_telegram_report(scan_out, paper_out), "opening_report")
                first_cycle = False

            # Alert immediately on new ACTIONABLE setups
            for item in scan_out.get("results", []):
                if item.get("status") == "ACTIONABLE":
                    key = f"{item['ticker']}-{item.get('current_trading_date')}-{item.get('active_fibo_level')}"
                    if key not in alerted_keys:
                        alerted_keys.add(key)
                        _notify_or_warn(format_signal_card(item), f"actionable:{item.get('ticker', '')}")

            # Alert on newly closed paper trades (TP1 / TP2 / SL)
            for closed in paper_out.get("newly_closed", []):
                outcome_emoji = "✅ WIN" if closed.get("is_win") else "🛑 LOSS"
                tk_esc = html.escape(str(closed.get("ticker", "")), quote=False)
                act_esc = html.escape(str(closed.get("option_action", "")), quote=False)
                reason_esc = html.escape(str(closed.get("exit_reason", "")), quote=False)
                msg = (
                    f"{outcome_emoji} <b>[PAPER TRADE CLOSED] {tk_esc} ({act_esc})</b>\n"
                    f"• <b>สาเหตุการออก:</b> <code>{reason_esc}</code>\n"
                    f"• <b>ราคาหุ้นแม่:</b> เข้า <code>${closed['entry_underlying']:.2f}</code> → ออก <code>${closed['exit_underlying']:.2f}</code>\n"
                    f"• <b>ราคา Option:</b> เข้า <code>${closed['entry_option_price']:.2f}</code> → ออก <code>${closed['exit_option_price']:.2f}</code>\n"
                    f"• <b>กำไร/ขาดทุนสุทธิ:</b> <code>${closed['net_pnl_usd']:+,.2f}</code>\n"
                    f"• <b>อัตราชนะสะสม (Win Rate):</b> <code>{paper_out['kpis']['win_rate_pct']:.1f}%</code>"
                )
                _notify_or_warn(msg, f"paper_closed:{closed.get('ticker', '')}")

        except KeyboardInterrupt:
            print("[Day Option] Stopped by user.")
            break
        except Exception as e:
            failed_cycles += 1
            print(f"[Day Option] Loop error: {e}")
            traceback.print_exc()

        # Poll Telegram commands every 5 seconds while waiting for the next 30-second scan
        wait_total = max(5, interval_sec)
        waited = 0
        while waited < wait_total:
            if _stop_reason() is not None:
                break
            sleep_chunk = min(5, wait_total - waited)
            time.sleep(sleep_chunk)
            waited += sleep_chunk
            if _stop_reason() is not None:
                break
            if poll_enabled:
                try:
                    poll_telegram_commands(config)
                except Exception as e:
                    print(f"[Day Option] Command poll error: {e}")

    if failed_cycles > 0 and success_cycles == 0:
        raise RuntimeError(f"All {failed_cycles} scan cycle(s) failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day Option Bot (EMA 200 + Fibo 0.500/0.618 + Real-Time BS)")
    parser.add_argument("--once", action="store_true", help="Run a single scan cycle and update dashboard")
    parser.add_argument("--loop", action="store_true", help="Run continuous 30-second real-time monitoring loop")
    parser.add_argument("--interval", type=int, default=30, help="Loop interval in seconds (default: 30)")
    parser.add_argument("--max-minutes", type=int, default=0, help="Max session duration in minutes for Cloud/GitHub runner")
    parser.add_argument("--send-opening-report", action="store_true", help="Send full scan report on first loop cycle")
    parser.add_argument("--mark-session-date", type=str, default="", help="NY trading session date stamp (YYYY-MM-DD)")
    parser.add_argument("--backtest", action="store_true", help="Run 90-day intraday backtest across watchlist")
    parser.add_argument("--no-telegram", action="store_true", help="Skip sending Telegram message on --once")
    parser.add_argument("--no-command-poll", action="store_true", help="Disable Telegram getUpdates command polling in --loop mode")
    args = parser.parse_args()

    config = load_config()

    if args.backtest:
        print("[Day Option] Running 90-Day Intraday Backtest...")
        bt_report = run_90d_backtest(config)
        print(
            f"[Day Option] 90-Day Backtest saved to {BACKTEST_OUT_PATH} | "
            f"Golden Win Rate: {bt_report['overall_golden']['win_rate_pct']:.1f}% | "
            f"Net PnL: ${bt_report['overall_golden']['net_pnl']:+,.2f}"
        )

    if args.loop:
        run_continuous_30s_loop(
            config,
            interval_sec=args.interval,
            max_minutes=args.max_minutes,
            send_opening_report=args.send_opening_report,
            ny_session_date=args.mark_session_date,
            no_command_poll=args.no_command_poll,
        )
    else:
        execute_single_cycle(
            config,
            notify_telegram=not args.no_telegram,
            ny_session_date=args.mark_session_date,
        )


if __name__ == "__main__":
    main()
