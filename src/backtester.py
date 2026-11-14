"""
90-Day Intraday Backtester for Day Option Bot.
Simulates Original Scope vs Upgraded (RVOL + Trailing TP) vs Golden Smart Fibo (0.500 ETF / 0.618 Stock)
across the watchlist and saves verified results to config/backtest_90d.json.
"""

from datetime import datetime
import json
import os
from typing import Dict, List
import numpy as np
import pandas as pd
import yfinance as yf


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKTEST_OUT_PATH = os.path.join(BASE_DIR, "config", "backtest_90d.json")

FRICTION_USD = {
    "SPY": 3.0,
    "QQQ": 3.0,
    "AMD": 5.0,
    "PLTR": 4.0,
    "META": 7.0,
    "NVDA": 5.0,
    "AAPL": 4.0,
    "AMZN": 5.0,
}

OPTION_DELTA = 0.58


def fetch_90d_intraday(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    df_5m = t.history(period="60d", interval="5m")
    df_2m = t.history(period="60d", interval="2m")
    if df_5m.empty and df_2m.empty:
        return pd.DataFrame()
    if not df_2m.empty and not df_5m.empty:
        first_2m_date = df_2m.index[0].date()
        older_5m = df_5m[df_5m.index.date < first_2m_date]
        combined = pd.concat([older_5m, df_2m]).sort_index()
    elif not df_2m.empty:
        combined = df_2m
    else:
        combined = df_5m
    return combined[~combined.index.duplicated(keep="first")]


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["VolMA20"] = df["Volume"].rolling(20, min_periods=5).mean()
    df["RVOL"] = np.where(df["VolMA20"] > 0, df["Volume"] / df["VolMA20"], 1.0)
    df["BodyHigh"] = np.maximum(df["Open"], df["Close"])
    df["BodyLow"] = np.minimum(df["Open"], df["Close"])
    df["Date"] = df.index.date
    return df


def simulate_ticker_90d(ticker: str, df: pd.DataFrame, mode: str = "golden") -> List[Dict]:
    dates = sorted(df["Date"].unique())
    trades = []
    friction = FRICTION_USD.get(ticker, 5.0)

    for d_idx in range(1, len(dates)):
        prev_df = df[df["Date"] == dates[d_idx - 1]]
        curr_df = df[df["Date"] == dates[d_idx]]
        if len(prev_df) < 10 or len(curr_df) < 10:
            continue

        idx_min = prev_df["Low"].idxmin()
        idx_max = prev_df["High"].idxmax()
        wick_low = float(prev_df.loc[idx_min, "Low"])
        body_low = float(prev_df.loc[idx_min, "BodyLow"])
        wick_high = float(prev_df.loc[idx_max, "High"])
        body_high = float(prev_df.loc[idx_max, "BodyHigh"])

        entered_levels = set()
        in_trade = None
        rows = list(curr_df.itertuples())

        for b_idx, row in enumerate(rows):
            price = float(row.Close)
            open_p = float(row.Open)
            high_p = float(row.High)
            low_p = float(row.Low)
            ema200 = float(row.EMA200)
            rvol = float(row.RVOL)
            contracts = 2 if price < 80 else 1

            if in_trade is not None:
                direction = in_trade["direction"]
                entry_u = in_trade["entry_underlying"]
                tp1 = in_trade["tp1"]
                tp2 = in_trade["tp2"]
                sl = in_trade["sl"]
                exit_u = None
                exit_reason = None

                if mode == "original":
                    if direction == "BUY_PUT":
                        if high_p >= sl:
                            exit_u, exit_reason = sl, "SL"
                        elif low_p <= tp1:
                            exit_u, exit_reason = tp1, "TP"
                    else:
                        if low_p <= sl:
                            exit_u, exit_reason = sl, "SL"
                        elif high_p >= tp1:
                            exit_u, exit_reason = tp1, "TP"
                else:
                    if direction == "BUY_PUT":
                        if low_p <= tp2:
                            exit_u, exit_reason = tp2, "TP2_FULL"
                        elif not in_trade["tp1_hit"] and low_p <= tp1:
                            in_trade["tp1_hit"] = True
                            in_trade["sl"] = entry_u - (entry_u - tp1) * 0.45
                        elif high_p >= sl:
                            exit_u = sl
                            exit_reason = "TP1_TRAIL" if in_trade["tp1_hit"] else "SL"
                    else:
                        if high_p >= tp2:
                            exit_u, exit_reason = tp2, "TP2_FULL"
                        elif not in_trade["tp1_hit"] and high_p >= tp1:
                            in_trade["tp1_hit"] = True
                            in_trade["sl"] = entry_u + (tp1 - entry_u) * 0.45
                        elif low_p <= sl:
                            exit_u = sl
                            exit_reason = "TP1_TRAIL" if in_trade["tp1_hit"] else "SL"

                if exit_u is None and b_idx == len(rows) - 1:
                    exit_u, exit_reason = price, "EOD"

                if exit_u is not None:
                    move = (entry_u - exit_u) if direction == "BUY_PUT" else (exit_u - entry_u)
                    gross = move * OPTION_DELTA * 100.0 * contracts
                    net = gross - (friction * contracts)
                    in_trade["exit_underlying"] = round(exit_u, 2)
                    in_trade["exit_reason"] = exit_reason
                    in_trade["gross_pnl"] = round(gross, 2)
                    in_trade["net_pnl"] = round(net, 2)
                    in_trade["is_win"] = net > 0
                    trades.append(in_trade)
                    in_trade = None
                continue

            if b_idx >= len(rows) - 5 or b_idx < 2:
                continue

            is_downtrend = price < ema200
            is_uptrend = price > ema200

            if mode in ("upgraded", "golden"):
                prev_rvol = float(rows[b_idx - 1].RVOL)
                ema_prev = float(rows[max(0, b_idx - 5)].EMA200)
                if is_downtrend and ema200 > ema_prev * 1.0005:
                    continue
                if is_uptrend and ema200 < ema_prev * 0.9995:
                    continue
                if max(rvol, prev_rvol) < 1.15:
                    continue

            if is_downtrend:
                rng = body_high - wick_low
                if rng <= price * 0.008:
                    continue
                lvl_236 = wick_low + 0.236 * rng
                lvl_382 = wick_low + 0.382 * rng
                lvl_500 = wick_low + 0.500 * rng
                lvl_618 = wick_low + 0.618 * rng
                lvl_786 = wick_low + 0.786 * rng

                cands = [
                    ("0.500", lvl_500, lvl_382, lvl_236, lvl_618),
                    ("0.618", lvl_618, lvl_500, lvl_382, lvl_786),
                ]
                if mode == "golden":
                    cands = [cands[0]] if ticker in ("SPY", "QQQ") else [cands[1]]

                for l_name, l_val, tp1, tp2, sl_v in cands:
                    if l_name in entered_levels:
                        continue
                    tol = rng * 0.015
                    if (high_p >= l_val - tol) and (low_p <= l_val + tol) and (price < open_p) and (price <= l_val):
                        in_trade = {
                            "ticker": ticker,
                            "date": str(dates[d_idx]),
                            "direction": "BUY_PUT",
                            "fibo_level": l_name,
                            "entry_underlying": round(price, 2),
                            "tp1": round(tp1, 2),
                            "tp2": round(tp2, 2),
                            "sl": round(sl_v, 2),
                            "tp1_hit": False,
                        }
                        entered_levels.add(l_name)
                        break

            elif is_uptrend:
                rng = wick_high - body_low
                if rng <= price * 0.008:
                    continue
                lvl_236 = wick_high - 0.236 * rng
                lvl_382 = wick_high - 0.382 * rng
                lvl_500 = wick_high - 0.500 * rng
                lvl_618 = wick_high - 0.618 * rng
                lvl_786 = wick_high - 0.786 * rng

                cands = [
                    ("0.500", lvl_500, lvl_382, lvl_236, lvl_618),
                    ("0.618", lvl_618, lvl_500, lvl_382, lvl_786),
                ]
                if mode == "golden":
                    cands = [cands[0]] if ticker in ("SPY", "QQQ") else [cands[1]]

                for l_name, l_val, tp1, tp2, sl_v in cands:
                    if l_name in entered_levels:
                        continue
                    tol = rng * 0.015
                    if (low_p <= l_val + tol) and (high_p >= l_val - tol) and (price > open_p) and (price >= l_val):
                        in_trade = {
                            "ticker": ticker,
                            "date": str(dates[d_idx]),
                            "direction": "BUY_CALL",
                            "fibo_level": l_name,
                            "entry_underlying": round(price, 2),
                            "tp1": round(tp1, 2),
                            "tp2": round(tp2, 2),
                            "sl": round(sl_v, 2),
                            "tp1_hit": False,
                        }
                        entered_levels.add(l_name)
                        break

    return trades


def summarize_trades(trades: List[Dict]) -> Dict:
    if not trades:
        return {"count": 0, "wins": 0, "losses": 0, "win_rate_pct": 0.0, "net_pnl": 0.0, "profit_factor": 0.0}
    total = len(trades)
    wins = sum(1 for t in trades if t["is_win"])
    win_pnls = [t["net_pnl"] for t in trades if t["net_pnl"] > 0]
    loss_pnls = [abs(t["net_pnl"]) for t in trades if t["net_pnl"] <= 0]
    pf = round(sum(win_pnls) / sum(loss_pnls), 2) if sum(loss_pnls) > 0 else 99.0
    return {
        "count": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate_pct": round(wins / total * 100.0, 1),
        "gross_pnl": round(sum(t["gross_pnl"] for t in trades), 2),
        "net_pnl": round(sum(t["net_pnl"] for t in trades), 2),
        "profit_factor": pf,
    }


def run_90d_backtest(config: Dict, out_path: str = BACKTEST_OUT_PATH) -> Dict:
    wl = config.get("watchlist", {})
    tickers = list(wl.get("etf_index", [])) + list(wl.get("golden_stocks", []))

    all_orig, all_upgr, all_gold = [], [], []
    per_ticker = {}

    for tk in tickers:
        df = fetch_90d_intraday(tk)
        if df.empty:
            continue
        df = prepare_indicators(df)
        t_orig = simulate_ticker_90d(tk, df, "original")
        t_upgr = simulate_ticker_90d(tk, df, "upgraded")
        t_gold = simulate_ticker_90d(tk, df, "golden")
        all_orig.extend(t_orig)
        all_upgr.extend(t_upgr)
        all_gold.extend(t_gold)
        per_ticker[tk] = {
            "days_tested": len(set(df.index.date)),
            "bars_tested": len(df),
            "original": summarize_trades(t_orig),
            "upgraded": summarize_trades(t_upgr),
            "golden": summarize_trades(t_gold),
        }

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall_original": summarize_trades(all_orig),
        "overall_upgraded": summarize_trades(all_upgr),
        "overall_golden": summarize_trades(all_gold),
        "per_ticker": per_ticker,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report
