"""
Realistic Paper Trading & Win Rate Tracker for Day Option Bot.
Tracks:
- Auto-Entry on ACTIONABLE Fibonacci 0.500 / 0.618 signals strictly within 09:35-11:30 NY
  (with Ask-side spread crossing)
- Real-Time Underlying & Black-Scholes Option Mark-to-Market
- Two-Stage Exit Engine:
  * Hard EOD Date-Change Flush (evaluated before TP1/TP2/SL to prevent overnight stale checks)
  * Stage 1 (TP1 Hit at Next Fibo Line): Locks trailing stop in profit anchored to entry_underlying
  * Stage 2 (TP2 Runner at 2nd Fibo Line): Full target exit
  * Stop Loss (SL at Opposite Fibo Line): Cuts loss immediately
  * 2H Session-End Flush (11:30 NY): Zero post-window exposure
- Full Win Rate Analytics (Overall Win Rate %, Fibo 0.618 vs 0.500 Win Rate, Per-Ticker Leaderboard)
"""

from contextlib import contextmanager
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from typing import Dict, Iterator, List, Optional

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from .options_pricing import black_scholes_price
from .time_utils import format_ny_timestamp, is_within_2h_window_ny, now_ny, today_ny


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPER_STATE_PATH = os.path.join(BASE_DIR, "config", "paper_trades.json")

_THREAD_LOCK = threading.RLock()
_LOCK_DEPTH = threading.local()


def _get_lock_path(state_path: str) -> str:
    norm = os.path.abspath(state_path)
    digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]
    return os.path.join(tempfile.gettempdir(), f"day_option_paper_{digest}.lock")


@contextmanager
def _paper_state_lock(
    state_path: str = PAPER_STATE_PATH,
    timeout_sec: float = 10.0,
) -> Iterator[None]:
    norm = os.path.abspath(state_path)
    depths = getattr(_LOCK_DEPTH, "counts", None)
    if depths is None:
        depths = {}
        _LOCK_DEPTH.counts = depths

    current_depth = depths.get(norm, 0)
    if current_depth > 0:
        depths[norm] = current_depth + 1
        try:
            yield
        finally:
            depths[norm] -= 1
        return

    lock_path = _get_lock_path(norm)
    with _THREAD_LOCK:
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        fh = open(lock_path, "a+b")
        acquired = False
        try:
            fh.seek(0, os.SEEK_END)
            if fh.tell() == 0:
                fh.write(b"\0")
                fh.flush()

            deadline = time.monotonic() + timeout_sec
            while True:
                try:
                    if os.name == "nt":
                        fh.seek(0)
                        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Timed out waiting for paper state lock: {lock_path}"
                        )
                    time.sleep(0.02)

            depths[norm] = 1
            yield
        finally:
            depths.pop(norm, None)
            if acquired:
                try:
                    if os.name == "nt":
                        fh.seek(0)
                        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            fh.close()


def _default_paper_state(starting_balance: float = 5000.0) -> Dict:
    return {
        "created_at": format_ny_timestamp(),
        "last_updated": "",
        "starting_balance_usd": float(starting_balance),
        "open_positions": [],
        "closed_positions": [],
    }


def load_paper_state(path: Optional[str] = None) -> Dict:
    eff_path = path if path is not None else PAPER_STATE_PATH
    with _paper_state_lock(eff_path):
        if os.path.exists(eff_path):
            try:
                with open(eff_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError(
                        f"Expected JSON object, got {type(data).__name__}"
                    )
                data.setdefault("created_at", format_ny_timestamp())
                data.setdefault("last_updated", "")
                data.setdefault("starting_balance_usd", 5000.0)
                if not isinstance(data.get("open_positions"), list):
                    data["open_positions"] = []
                if not isinstance(data.get("closed_positions"), list):
                    data["closed_positions"] = []
                return data
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                ts = now_ny().strftime("%Y%m%dT%H%M%S_%f")
                bak_path = f"{eff_path}.corrupt.{ts}.bak"
                try:
                    shutil.copy2(eff_path, bak_path)
                    print(
                        f"[PaperTrader] Corrupt JSON in {eff_path} ({exc}); backed up to {bak_path}"
                    )
                except Exception as copy_exc:
                    print(
                        f"[PaperTrader] Corrupt JSON in {eff_path} ({exc}); backup failed: {copy_exc}"
                    )
        return _default_paper_state()


def save_paper_state(state: Dict, path: Optional[str] = None) -> None:
    eff_path = path if path is not None else PAPER_STATE_PATH
    parent = os.path.dirname(os.path.abspath(eff_path))
    os.makedirs(parent, exist_ok=True)
    with _paper_state_lock(eff_path):
        fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, eff_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise


def compute_paper_kpis(
    closed_positions: List[Dict],
    open_positions: List[Dict],
    starting_balance: float = 5000.0,
) -> Dict:
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
    profit_factor = (
        round(gross_profit / gross_loss, 2)
        if gross_loss > 0
        else (99.0 if gross_profit > 0 else 0.0)
    )

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

    for st in per_ticker.values():
        st["win_rate_pct"] = (
            round(st["wins"] / st["count"] * 100.0, 1) if st["count"] > 0 else 0.0
        )

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
            "net_pnl_usd": round(
                sum(float(t.get("net_pnl_usd", 0.0)) for t in f500), 2
            ),
        },
        "fibo_0618": {
            "count": len(f618),
            "wins": f618_wins,
            "win_rate_pct": round(f618_wins / len(f618) * 100.0, 1) if f618 else 0.0,
            "net_pnl_usd": round(
                sum(float(t.get("net_pnl_usd", 0.0)) for t in f618), 2
            ),
        },
        "per_ticker": per_ticker,
    }


def _compute_safe_lock_sl(
    opt_type: str,
    entry_u: float,
    tp1_u: float,
    candidate_lock: Optional[float] = None,
    lock_ratio: float = 0.45,
) -> float:
    """
    Guarantee trailing SL after TP1 sits on the profitable side of entry_underlying
    toward tp1_underlying (preventing TP1_TRAIL_WIN from locking in a losing underlying level).
    """
    if opt_type == "PUT":
        anchored = (
            round(entry_u - (entry_u - tp1_u) * lock_ratio, 2)
            if entry_u > tp1_u
            else round(entry_u, 2)
        )
        if candidate_lock is not None and tp1_u <= float(candidate_lock) < entry_u:
            return round(float(candidate_lock), 2)
        return min(anchored, round(entry_u, 2))
    else:
        anchored = (
            round(entry_u + (tp1_u - entry_u) * lock_ratio, 2)
            if entry_u < tp1_u
            else round(entry_u, 2)
        )
        if candidate_lock is not None and entry_u < float(candidate_lock) <= tp1_u:
            return round(float(candidate_lock), 2)
        return max(anchored, round(entry_u, 2))


def update_paper_portfolio(
    scan_output: Dict,
    config: Dict,
    enforce_window: bool = True,
    path: Optional[str] = None,
) -> Dict:
    """
    1. Marks open paper trades to market using live spot prices from scan_output.
    2. Evaluates EOD Date-Change Flush (FIRST), then TP2, TP1 (Lock Profit), and SL.
    3. Opens new paper positions for ACTIONABLE signals only when within 09:35-11:30 NY
       (when enforce_window=True and paper_trading.enforce_2h_window is True).
    4. Saves state under cross-process lock and returns updated KPIs + events.
    """
    eff_path = path if path is not None else PAPER_STATE_PATH
    with _paper_state_lock(eff_path):
        state = load_paper_state(eff_path)
        now_str = format_ny_timestamp()
        paper_cfg = config.get("paper_trading", {})
        starting_bal = float(
            paper_cfg.get(
                "starting_balance_usd", state.get("starting_balance_usd", 5000.0)
            )
        )
        state["starting_balance_usd"] = starting_bal
        lock_ratio = float(
            config.get("strategy_rules", {})
            .get("tp_sl_rules", {})
            .get("lock_profit_ratio_at_tp1", 0.45)
        )
        etf_friction = float(paper_cfg.get("etf_friction_per_contract_usd", 3.0))
        stock_friction = float(paper_cfg.get("stock_friction_per_contract_usd", 5.0))

        should_enforce_window = bool(enforce_window) and bool(
            paper_cfg.get("enforce_2h_window", True)
        )
        within_entry_window = (
            True if not should_enforce_window else is_within_2h_window_ny()
        )

        scan_map = {
            r["ticker"]: r for r in scan_output.get("results", []) if "spot_price" in r
        }

        still_open = []
        newly_closed = []
        newly_opened = []

        # 1. Update existing open positions (always evaluated regardless of 2h entry window)
        for pos in state.get("open_positions", []):
            tk = pos["ticker"]
            scan_item = scan_map.get(tk)
            if not scan_item:
                still_open.append(pos)
                continue

            spot = float(scan_item["spot_price"])
            curr_date = str(scan_item.get("current_trading_date") or "")
            pos_date = str(pos.get("trading_date") or "")
            direction = pos["option_action"]
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

            # Force EOD flush FIRST if trading date changed (Zero Overnight Risk rule)
            if curr_date and pos_date and curr_date != pos_date:
                exit_u = spot
                exit_reason = "EOD_FORCE_FLUSH"
            elif opt_type == "PUT":
                if spot <= tp2_u:
                    exit_u = tp2_u
                    exit_reason = "TP2_FULL_WIN"
                elif not pos.get("tp1_hit") and spot <= tp1_u:
                    safe_lock = _compute_safe_lock_sl(
                        "PUT", entry_u, tp1_u, lock_sl_u, lock_ratio
                    )
                    pos["tp1_hit"] = True
                    pos["lock_sl_after_tp1"] = safe_lock
                    pos["sl_underlying"] = safe_lock
                    sl_u = safe_lock
                elif spot >= sl_u:
                    exit_u = sl_u
                    exit_reason = (
                        "TP1_TRAIL_WIN" if pos.get("tp1_hit") else "SL_LOSS"
                    )
            else:  # CALL
                if spot >= tp2_u:
                    exit_u = tp2_u
                    exit_reason = "TP2_FULL_WIN"
                elif not pos.get("tp1_hit") and spot >= tp1_u:
                    safe_lock = _compute_safe_lock_sl(
                        "CALL", entry_u, tp1_u, lock_sl_u, lock_ratio
                    )
                    pos["tp1_hit"] = True
                    pos["lock_sl_after_tp1"] = safe_lock
                    pos["sl_underlying"] = safe_lock
                    sl_u = safe_lock
                elif spot <= sl_u:
                    exit_u = sl_u
                    exit_reason = (
                        "TP1_TRAIL_WIN" if pos.get("tp1_hit") else "SL_LOSS"
                    )

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

        # 2. Auto-enter new ACTIONABLE signals strictly inside entry window
        if (
            paper_cfg.get("enabled", True)
            and paper_cfg.get("auto_enter_actionable", True)
            and within_entry_window
        ):
            for item in scan_output.get("results", []):
                if item.get("status") != "ACTIONABLE":
                    continue
                if should_enforce_window and item.get("is_current_session") is False:
                    continue

                tk = item["ticker"]
                curr_date = str(
                    item.get("current_trading_date") or today_ny().isoformat()
                )
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

                plan = item.get("trade_plan")
                oq = item.get("option_quote")
                if not plan or not oq:
                    continue

                opt_type = "PUT" if "PUT" in str(item.get("option_action", "")) else "CALL"
                entry_u = float(plan["entry_underlying"])
                tp1_u = float(plan["tp1_underlying"])
                safe_lock = _compute_safe_lock_sl(
                    opt_type, entry_u, tp1_u, plan.get("lock_sl_after_tp1"), lock_ratio
                )

                new_pos = {
                    "trade_id": (
                        f"{tk}-{curr_date}-{fibo_lvl}-"
                        f"{len(state['closed_positions']) + len(state['open_positions']) + 1}"
                    ),
                    "ticker": tk,
                    "is_etf": bool(item.get("is_etf", False)),
                    "trading_date": curr_date,
                    "opened_at": now_str,
                    "option_action": item["option_action"],
                    "fibo_level": fibo_lvl,
                    "entry_underlying": entry_u,
                    "current_underlying": entry_u,
                    "tp1_underlying": tp1_u,
                    "lock_sl_after_tp1": safe_lock,
                    "tp2_underlying": float(plan["tp2_underlying"]),
                    "sl_underlying": float(plan["sl_underlying"]),
                    "tp1_hit": False,
                    "expiration": oq["expiration"],
                    "dte": int(oq["dte"]),
                    "strike": float(oq["strike"]),
                    "iv": float(oq["iv"]),
                    "delta": float(oq["delta"]),
                    "entry_option_price": float(oq["est_ask"]),
                    "current_option_price": float(oq["realtime_theo_price"]),
                    "contracts": int(oq["recommended_contracts"]),
                }
                state["open_positions"].append(new_pos)
                newly_opened.append(new_pos)

        state["last_updated"] = now_str
        save_paper_state(state, eff_path)

        kpis = compute_paper_kpis(
            state["closed_positions"], state["open_positions"], starting_bal
        )
        return {
            "state": state,
            "kpis": kpis,
            "newly_opened": newly_opened,
            "newly_closed": newly_closed,
            "entry_window_open": within_entry_window,
        }


def flush_open_paper_positions_at_session_end(
    config: Dict,
    latest_scan: Optional[Dict] = None,
    path: Optional[str] = None,
) -> Dict:
    """
    Force-closes any remaining open paper positions at the end of the 2-hour trading window (11:30 NY)
    using fresh BSM mark prices from latest_scan (if provided) or last stored mark prices.
    """
    eff_path = path if path is not None else PAPER_STATE_PATH
    with _paper_state_lock(eff_path):
        state = load_paper_state(eff_path)
        paper_cfg = config.get("paper_trading", {})
        starting_bal = float(
            paper_cfg.get(
                "starting_balance_usd", state.get("starting_balance_usd", 5000.0)
            )
        )
        open_list = state.get("open_positions", [])
        if not open_list:
            return {
                "flushed": [],
                "kpis": compute_paper_kpis(
                    state.get("closed_positions", []), [], starting_bal
                ),
            }

        now_str = format_ny_timestamp()
        etf_friction = float(paper_cfg.get("etf_friction_per_contract_usd", 3.0))
        stock_friction = float(paper_cfg.get("stock_friction_per_contract_usd", 5.0))
        scan_map = {
            r["ticker"]: r
            for r in (latest_scan or {}).get("results", [])
            if "spot_price" in r
        }

        flushed = []
        for pos in open_list:
            contracts = int(pos.get("contracts", 1))
            friction = etf_friction if pos.get("is_etf") else stock_friction
            entry_opt = float(pos["entry_option_price"])

            scan_item = scan_map.get(pos.get("ticker"))
            if scan_item is not None:
                exit_u = float(scan_item["spot_price"])
                opt_type = (
                    "PUT" if "PUT" in str(pos.get("option_action", "")) else "CALL"
                )
                exit_opt = black_scholes_price(
                    spot=exit_u,
                    strike=float(pos["strike"]),
                    dte_days=max(1, int(pos.get("dte", 3))),
                    iv=float(pos.get("iv", 0.35)),
                    option_type=opt_type,
                )
                pos["current_underlying"] = exit_u
                pos["current_option_price"] = round(exit_opt, 2)
            else:
                exit_opt = float(pos.get("current_option_price", entry_opt))
                exit_u = float(pos.get("current_underlying", pos["entry_underlying"]))

            gross_pnl = (exit_opt - entry_opt) * 100.0 * contracts
            net_pnl = round(gross_pnl - (friction * contracts), 2)

            pos["exit_underlying"] = round(exit_u, 2)
            pos["exit_option_price"] = round(exit_opt, 2)
            pos["exit_reason"] = "2H_WINDOW_CLOSE"
            pos["closed_at"] = now_str
            pos["gross_pnl_usd"] = round(gross_pnl, 2)
            pos["net_pnl_usd"] = net_pnl
            pos["is_win"] = net_pnl > 0
            state["closed_positions"].append(pos)
            flushed.append(pos)

        state["open_positions"] = []
        state["last_updated"] = now_str
        save_paper_state(state, eff_path)
        kpis = compute_paper_kpis(state["closed_positions"], [], starting_bal)
        return {"flushed": flushed, "kpis": kpis}
