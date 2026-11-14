"""
Realistic Paper Trading & Win Rate Tracker for Day Option Bot.
Tracks:
- Auto-Entry on ACTIONABLE Fibonacci 0.500 / 0.618 signals (with Ask-side spread crossing)
- Real-Time Underlying & Black-Scholes Option Mark-to-Market
- Two-Stage Exit Engine:
  * Stage 1 (TP1 Hit at Next Fibo Line): Locks trailing stop in profit (Breakeven + 45% of TP1 move)
  * Stage 2 (TP2 Runner at 2nd Fibo Line): Full target exit
  * Stop Loss (SL at Opposite Fibo Line): Cuts loss immediately
  * 15:45 EST / Next-Day Hard Flush: Zero overnight option exposure
- Full Win Rate Analytics (Overall Win Rate %, Fibo 0.618 vs 0.500 Win Rate, Per-Ticker Leaderboard)
"""

from datetime import datetime
import json
import os
import tempfile
from typing import Dict, List, Optional

from .options_pricing import black_scholes_price


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPER_STATE_PATH = os.path.join(BASE_DIR, "config", "paper_trades.json")


def load_paper_state(path: str = PAPER_STATE_PATH) -> Dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_updated": "",
        "starting_balance_usd": 5000.0,
        "open_positions": [],
        "closed_positions": [],
    }


def save_paper_state(state: Dict, path: str = PAPER_STATE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def compute_paper_kpis(closed_positions: List[Dict], open_positions: List[Dict], starting_balance: float = 5000.0) -> Dict:
    total = len(closed_positions)
    wins = [t for t in closed_positions if float(t.get("net_pnl_usd", 0.0)) > 0]
    losses = [t for t in closed_positions if float(t.get("net_pnl_usd", 0.0)) <= 0]

    win_count = len(wins)
    loss_count = len(losses)
    win_rate_pct = round((win_count / total) * 100.0, 1) if total > 0 else 0.0

    gross_profit = sum(float(t.get("net_pnl_usd", 0.0)) for t in wins)
    gross_loss = abs(sum(float(t.get("net_pnl_usd", 0.0)) for t in losses))
    net_pnl = round(gross_profit - gross_loss, 2)

    avg_win = round(gross_profit / win_count, 2) if win_count > 0 else 0.0
    avg_loss = round(gross_loss / loss_count, 2) if loss_count > 0 else 0.0
    realized_rr = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0.0
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    # Breakdown by Fibo Level (0.500 vs 0.618)
    f500 = [t for t in closed_positions if str(t.get("fibo_level")) == "0.500"]
    f618 = [t for t in closed_positions if str(t.get("fibo_level")) == "0.618"]

    f500_wins = sum(1 for t in f500 if float(t.get("net_pnl_usd", 0.0)) > 0)
    f618_wins = sum(1 for t in f618 if float(t.get("net_pnl_usd", 0.0)) > 0)

    # Breakdown by Ticker
    per_ticker = {}
    for t in closed_positions:
        tk = t.get("ticker", "UNKNOWN")
        if tk not in per_ticker:
            per_ticker[tk] = {"count": 0, "wins": 0, "losses": 0, "net_pnl_usd": 0.0}
        per_ticker[tk]["count"] += 1
        pnl = float(t.get("net_pnl_usd", 0.0))
        per_ticker[tk]["net_pnl_usd"] = round(per_ticker[tk]["net_pnl_usd"] + pnl, 2)
        if pnl > 0:
            per_ticker[tk]["wins"] += 1
        else:
            per_ticker[tk]["losses"] += 1

    for tk, st in per_ticker.items():
        st["win_rate_pct"] = round(st["wins"] / st["count"] * 100.0, 1) if st["count"] > 0 else 0.0

    return {
        "total_closed": total,
        "open_count": len(open_positions),
        "wins": win_count,
        "losses": loss_count,
        "win_rate_pct": win_rate_pct,
        "net_pnl_usd": net_pnl,
        "current_equity_usd": round(starting_balance + net_pnl, 2),
        "avg_win_usd": avg_win,
        "avg_loss_usd": avg_loss,
        "realized_rr": realized_rr,
        "profit_factor": profit_factor,
        "fibo_0500": {
            "count": len(f500),
            "wins": f500_wins,
            "win_rate_pct": round(f500_wins / len(f500) * 100.0, 1) if f500 else 0.0,
            "net_pnl_usd": round(sum(float(t.get("net_pnl_usd", 0.0)) for t in f500), 2),
        },
        "fibo_0618": {
            "count": len(f618),
            "wins": f618_wins,
            "win_rate_pct": round(f618_wins / len(f618) * 100.0, 1) if f618 else 0.0,
            "net_pnl_usd": round(sum(float(t.get("net_pnl_usd", 0.0)) for t in f618), 2),
        },
        "per_ticker": per_ticker,
    }


def update_paper_portfolio(scan_output: Dict, config: Dict, path: str = PAPER_STATE_PATH) -> Dict:
    """
    1. Marks open paper trades to market using live spot prices from scan_output.
    2. Evaluates TP1 (Lock Profit), TP2 (Full Runner), SL (Opposite Fibo Line), and EOD Flush.
    3. Opens new paper positions for ACTIONABLE signals.
    4. Saves state and returns updated KPIs + newly opened/closed events.
    """
    state = load_paper_state(path)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    paper_cfg = config.get("paper_trading", {})
    starting_bal = float(paper_cfg.get("starting_balance_usd", 5000.0))
    etf_friction = float(paper_cfg.get("etf_friction_per_contract_usd", 3.0))
    stock_friction = float(paper_cfg.get("stock_friction_per_contract_usd", 5.0))

    scan_map = {r["ticker"]: r for r in scan_output.get("results", []) if "spot_price" in r}

    still_open = []
    newly_closed = []
    newly_opened = []

    # 1. Update existing open positions
    for pos in state.get("open_positions", []):
        tk = pos["ticker"]
        scan_item = scan_map.get(tk)
        if not scan_item:
            still_open.append(pos)
            continue

        spot = float(scan_item["spot_price"])
        curr_date = str(scan_item.get("current_trading_date", ""))
        direction = pos["option_action"]  # "BUY PUT" or "BUY CALL"
        opt_type = "PUT" if "PUT" in direction else "CALL"

        entry_u = float(pos["entry_underlying"])
        tp1_u = float(pos["tp1_underlying"])
        tp2_u = float(pos["tp2_underlying"])
        sl_u = float(pos["sl_underlying"])
        lock_sl_u = float(pos["lock_sl_after_tp1"])
        contracts = int(pos.get("contracts", 1))
        friction = etf_friction if pos.get("is_etf") else stock_friction

        # Live Black-Scholes option mark at current spot
        curr_opt_mark = black_scholes_price(
            spot=spot,
            strike=float(pos["strike"]),
            dte_days=max(1, int(pos.get("dte", 3))),
            iv=float(pos.get("iv", 0.35)),
            option_type=opt_type,
        )
        pos["current_underlying"] = spot
        pos["current_option_price"] = round(curr_opt_mark, 2)

        exit_reason = None
        exit_u = None

        if opt_type == "PUT":
            if spot <= tp2_u:
                exit_u = tp2_u
                exit_reason = "TP2_FULL_WIN"
            elif not pos.get("tp1_hit") and spot <= tp1_u:
                pos["tp1_hit"] = True
                pos["sl_underlying"] = lock_sl_u
                sl_u = lock_sl_u
            elif spot >= sl_u:
                exit_u = sl_u
                exit_reason = "TP1_TRAIL_WIN" if pos.get("tp1_hit") else "SL_LOSS"
        else:  # CALL
            if spot >= tp2_u:
                exit_u = tp2_u
                exit_reason = "TP2_FULL_WIN"
            elif not pos.get("tp1_hit") and spot >= tp1_u:
                pos["tp1_hit"] = True
                pos["sl_underlying"] = lock_sl_u
                sl_u = lock_sl_u
            elif spot <= sl_u:
                exit_u = sl_u
                exit_reason = "TP1_TRAIL_WIN" if pos.get("tp1_hit") else "SL_LOSS"

        # Force EOD flush if trading date changed (Zero Overnight Risk rule)
        if exit_u is None and curr_date and pos.get("trading_date") and curr_date != pos.get("trading_date"):
            exit_u = spot
            exit_reason = "EOD_FORCE_FLUSH"

        if exit_u is not None:
            exit_opt = black_scholes_price(
                spot=exit_u,
                strike=float(pos["strike"]),
                dte_days=max(1, int(pos.get("dte", 3))),
                iv=float(pos.get("iv", 0.35)),
                option_type=opt_type,
            )
            entry_opt = float(pos["entry_option_price"])
            gross_pnl = (exit_opt - entry_opt) * 100.0 * contracts
            net_pnl = round(gross_pnl - (friction * contracts), 2)
            pos["exit_underlying"] = round(exit_u, 2)
            pos["exit_option_price"] = round(exit_opt, 2)
            pos["exit_reason"] = exit_reason
            pos["closed_at"] = now_str
            pos["gross_pnl_usd"] = round(gross_pnl, 2)
            pos["net_pnl_usd"] = net_pnl
            pos["is_win"] = net_pnl > 0
            state["closed_positions"].append(pos)
            newly_closed.append(pos)
        else:
            still_open.append(pos)

    state["open_positions"] = still_open

    # 2. Auto-enter new ACTIONABLE signals
    if paper_cfg.get("auto_enter_actionable", True):
        for item in scan_output.get("results", []):
            if item.get("status") != "ACTIONABLE":
                continue
            tk = item["ticker"]
            curr_date = str(item.get("current_trading_date", ""))
            fibo_lvl = str(item.get("active_fibo_level", "0.618"))

            # Prevent duplicate trade on same ticker + Fibo level on the same day
            already_traded = any(
                p.get("ticker") == tk
                and p.get("trading_date") == curr_date
                and str(p.get("fibo_level")) == fibo_lvl
                for p in (state["open_positions"] + state["closed_positions"])
            )
            if already_traded:
                continue

            plan = item["trade_plan"]
            oq = item["option_quote"]
            new_pos = {
                "trade_id": f"{tk}-{curr_date}-{fibo_lvl}-{len(state['closed_positions']) + len(state['open_positions']) + 1}",
                "ticker": tk,
                "is_etf": bool(item.get("is_etf", False)),
                "trading_date": curr_date,
                "opened_at": now_str,
                "option_action": item["option_action"],
                "fibo_level": fibo_lvl,
                "entry_underlying": float(plan["entry_underlying"]),
                "current_underlying": float(plan["entry_underlying"]),
                "tp1_underlying": float(plan["tp1_underlying"]),
                "lock_sl_after_tp1": float(plan["lock_sl_after_tp1"]),
                "tp2_underlying": float(plan["tp2_underlying"]),
                "sl_underlying": float(plan["sl_underlying"]),
                "tp1_hit": False,
                "expiration": oq["expiration"],
                "dte": int(oq["dte"]),
                "strike": float(oq["strike"]),
                "iv": float(oq["iv"]),
                "delta": float(oq["delta"]),
                "entry_option_price": float(oq["est_ask"]),  # Realistic Ask-side fill
                "current_option_price": float(oq["realtime_theo_price"]),
                "contracts": int(oq["recommended_contracts"]),
            }
            state["open_positions"].append(new_pos)
            newly_opened.append(new_pos)

    state["last_updated"] = now_str
    save_paper_state(state, path)

    kpis = compute_paper_kpis(state["closed_positions"], state["open_positions"], starting_bal)
    return {
        "state": state,
        "kpis": kpis,
        "newly_opened": newly_opened,
        "newly_closed": newly_closed,
    }
