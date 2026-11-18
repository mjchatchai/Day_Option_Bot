"""
Main Entrypoint for Day Option Bot (Bot Option Daytrade).
Modes:
  python -m day_option_bot.src.main --once      (Single scan + Paper Trade update + HTML Dashboard + Telegram report)
  python -m day_option_bot.src.main --loop      (Continuous 30-second real-time monitor + Telegram commands)
  python -m day_option_bot.src.main --backtest  (Run 90-Day Intraday Backtest and update dashboard)
"""

import argparse
import json
import os
import time
from typing import Dict

from .backtester import BACKTEST_OUT_PATH, run_90d_backtest
from .command_handler import poll_telegram_commands
from .dashboard_builder import build_html_dashboard
from .fibo_scanner import run_full_day_option_scan
from .paper_trader import flush_open_paper_positions_at_session_end, update_paper_portfolio
from .telegram_notifier import build_scan_telegram_report, format_signal_card, send_telegram_message


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.json")
LATEST_SCAN_PATH = os.path.join(BASE_DIR, "config", "latest_scan.json")


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
        send_telegram_message(report_msg)

    return {"scan": scan_out, "paper": paper_out, "dashboard": dash_path}


def run_continuous_30s_loop(
    config: Dict,
    interval_sec: int = 30,
    max_minutes: int = 0,
    send_opening_report: bool = False,
    ny_session_date: str = "",
) -> None:
    print(
        f"[Day Option] Starting 30-second real-time loop "
        f"(interval={interval_sec}s, max_minutes={max_minutes or 'unlimited'})..."
    )
    alerted_keys = set()
    start_ts = time.time()
    first_cycle = True

    while True:
        if max_minutes > 0 and (time.time() - start_ts) >= (max_minutes * 60):
            print(f"[Day Option] Reached 2-hour window limit ({max_minutes} mins). Closing any remaining open paper trades...")
            flush_res = flush_open_paper_positions_at_session_end(config)
            for closed in flush_res.get("flushed", []):
                outcome_emoji = "✅ WIN" if closed.get("is_win") else "🛑 LOSS"
                msg = (
                    f"{outcome_emoji} <b>[ปิดจบ 2 ชม. แรก] {closed['ticker']} ({closed['option_action']})</b>\n"
                    f"• <b>ราคา Option:</b> เข้า <code>${closed['entry_option_price']:.2f}</code> → ออก <code>${closed['exit_option_price']:.2f}</code>\n"
                    f"• <b>กำไร/ขาดทุนสุทธิ:</b> <code>${closed['net_pnl_usd']:+,.2f}</code>\n"
                    f"• <b>Paper Win Rate:</b> <code>{flush_res['kpis']['win_rate_pct']:.1f}%</code>"
                )
                send_telegram_message(msg)
            break

        try:
            poll_telegram_commands(config)
            scan_out = run_full_day_option_scan(config)
            if ny_session_date:
                scan_out["ny_session_date"] = ny_session_date
            os.makedirs(os.path.dirname(LATEST_SCAN_PATH), exist_ok=True)
            with open(LATEST_SCAN_PATH, "w", encoding="utf-8") as f:
                json.dump(scan_out, f, ensure_ascii=False, indent=2)

            paper_out = update_paper_portfolio(scan_out, config)
            build_html_dashboard(scan_out, paper_out)

            if first_cycle and send_opening_report:
                send_telegram_message(build_scan_telegram_report(scan_out, paper_out))
                first_cycle = False

            # Alert immediately on new ACTIONABLE setups
            for item in scan_out.get("results", []):
                if item.get("status") == "ACTIONABLE":
                    key = f"{item['ticker']}-{item.get('current_trading_date')}-{item.get('active_fibo_level')}"
                    if key not in alerted_keys:
                        alerted_keys.add(key)
                        send_telegram_message(format_signal_card(item))

            # Alert on newly closed paper trades (TP1 / TP2 / SL)
            for closed in paper_out.get("newly_closed", []):
                outcome_emoji = "✅ WIN" if closed.get("is_win") else "🛑 LOSS"
                msg = (
                    f"{outcome_emoji} <b>[PAPER TRADE CLOSED] {closed['ticker']} ({closed['option_action']})</b>\n"
                    f"• <b>สาเหตุการออก:</b> <code>{closed['exit_reason']}</code>\n"
                    f"• <b>ราคาหุ้นแม่:</b> เข้า <code>${closed['entry_underlying']:.2f}</code> → ออก <code>${closed['exit_underlying']:.2f}</code>\n"
                    f"• <b>ราคา Option:</b> เข้า <code>${closed['entry_option_price']:.2f}</code> → ออก <code>${closed['exit_option_price']:.2f}</code>\n"
                    f"• <b>กำไร/ขาดทุนสุทธิ:</b> <code>${closed['net_pnl_usd']:+,.2f}</code>\n"
                    f"• <b>อัตราชนะสะสม (Win Rate):</b> <code>{paper_out['kpis']['win_rate_pct']:.1f}%</code>"
                )
                send_telegram_message(msg)

        except KeyboardInterrupt:
            print("[Day Option] Stopped by user.")
            break
        except Exception as e:
            print(f"[Day Option] Loop error: {e}")

        # Poll Telegram commands every 5 seconds while waiting for the next 30-second scan
        wait_total = max(5, interval_sec)
        waited = 0
        while waited < wait_total:
            sleep_chunk = min(5, wait_total - waited)
            time.sleep(sleep_chunk)
            waited += sleep_chunk
            try:
                poll_telegram_commands(config)
            except Exception:
                pass


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
        )
    else:
        execute_single_cycle(
            config,
            notify_telegram=not args.no_telegram,
            ny_session_date=args.mark_session_date,
        )


if __name__ == "__main__":
    main()
