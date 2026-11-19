"""
Shared E2E fixtures, isolation guards, synthetic data/chain builders,
Telegram/yfinance/clock mocks, Cloud YAML + Git Bash runner, and Node.js GAS VM harness
for the Day Option Bot 4-tier opaque-box E2E test suite.

IMPORTANT (Concurrent M1/M2 Safety):
- Do NOT import `day_option_bot.src.time_utils` or `has_active_telegram_webhook` at
  module top-level; import them lazily inside helper functions/tests so
  `pytest --collect-only` always succeeds with 0 collection errors.
"""

from __future__ import annotations

import copy
import dataclasses
from datetime import date, datetime, timedelta
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence
import urllib.request
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
import requests
import yaml
import yfinance as yf


WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
BOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
REPO_PARENT = WORKSPACE_ROOT

if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from day_option_bot.src import (  # noqa: E402
    backtester,
    command_handler,
    dashboard_builder,
    earnings_filter,
    fibo_scanner,
    main as bot_main,
    options_pricing,
    paper_trader,
    telegram_notifier,
)

NY_TZ = ZoneInfo("America/New_York")
CONFIG_JSON_PATH = os.path.join(BOT_DIR, "config", "config.json")
WORKFLOW_YAML_PATH = os.path.join(BOT_DIR, ".github", "workflows", "day_option_cloud.yml")
GAS_CODE_PATH = os.path.join(WORKSPACE_ROOT, "gas_webhook", "Code.gs")
GIT_BASH_EXE = r"C:\Program Files\Git\bin\bash.exe"


# ---------------------------------------------------------------------------
# Lazy M1 / M2 symbol loaders (collection-safe)
# ---------------------------------------------------------------------------

def import_time_utils():
    """Lazily import M1 `day_option_bot.src.time_utils`; fail cleanly inside test if missing."""
    try:
        return importlib.import_module("day_option_bot.src.time_utils")
    except Exception as exc:
        pytest.fail(f"M1 module day_option_bot.src.time_utils unavailable: {exc}")


def get_has_active_telegram_webhook() -> Callable[..., bool]:
    """Lazily resolve M2 `has_active_telegram_webhook` from command_handler or telegram_notifier."""
    for mod in (command_handler, telegram_notifier):
        fn = getattr(mod, "has_active_telegram_webhook", None)
        if callable(fn):
            return fn
    pytest.fail("M2 symbol `has_active_telegram_webhook` not yet implemented in command_handler or telegram_notifier")


# ---------------------------------------------------------------------------
# Autouse network & repo-state isolation guards
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def block_live_network(monkeypatch):
    """Fail immediately if any test attempts unmocked HTTP or yfinance network access."""

    def _forbidden(*args, **kwargs):
        raise RuntimeError(f"Live network access forbidden in E2E tests: args={args[:2]}")

    monkeypatch.setattr(requests, "get", _forbidden)
    monkeypatch.setattr(requests, "post", _forbidden)
    monkeypatch.setattr(requests.Session, "request", _forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)
    monkeypatch.setattr(yf, "Ticker", _forbidden)
    monkeypatch.setattr(yf, "download", _forbidden)
    for mod in (fibo_scanner, earnings_filter, options_pricing, backtester):
        if hasattr(mod, "yf"):
            monkeypatch.setattr(mod.yf, "Ticker", _forbidden)
            if hasattr(mod.yf, "download"):
                monkeypatch.setattr(mod.yf, "download", _forbidden)
    if hasattr(telegram_notifier, "requests"):
        monkeypatch.setattr(telegram_notifier.requests, "post", _forbidden)
    if hasattr(command_handler, "requests"):
        monkeypatch.setattr(command_handler.requests, "get", _forbidden)


@pytest.fixture(autouse=True)
def isolated_bot_paths(tmp_path, monkeypatch):
    """Redirect all mutable JSON/HTML/offset file paths to tmp_path and clear Telegram env vars."""
    cfg_dir = tmp_path / "config"
    docs_dir = tmp_path / "docs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    paper_path = str(cfg_dir / "paper_trades.json")
    scan_path = str(cfg_dir / "latest_scan.json")
    dash_path = str(docs_dir / "index.html")
    bt_path = str(cfg_dir / "backtest_90d.json")
    offset_path = str(cfg_dir / "telegram_offset.txt")

    monkeypatch.setattr(paper_trader, "PAPER_STATE_PATH", paper_path)
    monkeypatch.setattr(command_handler, "PAPER_STATE_PATH", paper_path)
    monkeypatch.setattr(bot_main, "LATEST_SCAN_PATH", scan_path)
    monkeypatch.setattr(dashboard_builder, "DASHBOARD_PATH", dash_path)
    monkeypatch.setattr(dashboard_builder, "BACKTEST_PATH", bt_path)
    monkeypatch.setattr(command_handler, "OFFSET_FILE", offset_path)
    monkeypatch.setattr(command_handler, "BACKTEST_FILE", bt_path)
    monkeypatch.setattr(backtester, "BACKTEST_OUT_PATH", bt_path)
    monkeypatch.setattr(bot_main, "BACKTEST_OUT_PATH", bt_path)

    # Rebind default-argument paths evaluated at import time
    if dashboard_builder.build_html_dashboard.__defaults__:
        monkeypatch.setattr(dashboard_builder.build_html_dashboard, "__defaults__", (dash_path,))
    if backtester.run_90d_backtest.__defaults__:
        monkeypatch.setattr(backtester.run_90d_backtest, "__defaults__", (bt_path,))
    lock_fn = getattr(paper_trader, "_paper_state_lock", None)
    if lock_fn is not None:
        lock_target = getattr(lock_fn, "__wrapped__", lock_fn)
        if getattr(lock_target, "__defaults__", None):
            monkeypatch.setattr(lock_target, "__defaults__", (paper_path, 10.0))

    for var in (
        "DAY_OPTION_TELEGRAM_BOT_TOKEN",
        "DAY_OPTION_TELEGRAM_CHAT_ID",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
    ):
        monkeypatch.delenv(var, raising=False)

    return {
        "tmp_path": tmp_path,
        "config_dir": cfg_dir,
        "docs_dir": docs_dir,
        "paper_state_path": paper_path,
        "latest_scan_path": scan_path,
        "dashboard_path": dash_path,
        "backtest_path": bt_path,
        "offset_path": offset_path,
    }


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Fresh deep copy of day_option_bot/config/config.json."""
    with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
        return copy.deepcopy(json.load(f))


# ---------------------------------------------------------------------------
# inspect.signature Adapters (handle M1/M2 positional/dataclass variations)
# ---------------------------------------------------------------------------

def call_check_earnings(
    ticker: str,
    reference_date: Optional[date] = None,
    days_before: int = 2,
    days_after: int = 1,
    enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    sig = inspect.signature(earnings_filter.check_earnings_blackout)
    kwargs: Dict[str, Any] = {}
    if "reference_date" in sig.parameters and reference_date is not None:
        kwargs["reference_date"] = reference_date
    if "days_before" in sig.parameters:
        kwargs["days_before"] = days_before
    if "days_after" in sig.parameters:
        kwargs["days_after"] = days_after
    if enabled is not None:
        if "enabled" in sig.parameters:
            kwargs["enabled"] = enabled
        elif enabled is False:
            pytest.fail("check_earnings_blackout missing `enabled` parameter per PROJECT.md contract")
    return earnings_filter.check_earnings_blackout(ticker, **kwargs)


def normalize_option_quote(quote: Any) -> Optional[Dict[str, Any]]:
    if quote is None:
        return None
    if isinstance(quote, dict):
        return dict(quote)
    if dataclasses.is_dataclass(quote):
        return dataclasses.asdict(quote)
    return dict(vars(quote))


def call_select_option(
    ticker: str,
    option_type: str,
    spot_realtime: float,
    fibo_entry_underlying: Optional[float],
    tp1_underlying: float,
    tp2_underlying: float,
    sl_underlying: float,
    min_dte: int = 1,
    max_dte: int = 7,
    target_abs_delta: Optional[float] = None,
    max_spread_pct: Optional[float] = None,
    reference_date: Optional[date] = None,
    fallback_hv: float = 0.35,
) -> Optional[Dict[str, Any]]:
    fn = options_pricing.select_and_price_realtime_option
    sig = inspect.signature(fn)
    params = sig.parameters
    kwargs: Dict[str, Any] = {"ticker": ticker}

    if "option_type" in params:
        kwargs["option_type"] = option_type
    elif "direction" in params:
        kwargs["direction"] = f"BUY {option_type.upper()}" if option_type.upper() in ("CALL", "PUT") else option_type

    kwargs["spot_realtime"] = spot_realtime
    kwargs["tp1_underlying"] = tp1_underlying
    kwargs["tp2_underlying"] = tp2_underlying
    kwargs["sl_underlying"] = sl_underlying
    if "fibo_entry_underlying" in params:
        kwargs["fibo_entry_underlying"] = fibo_entry_underlying
    if "fallback_hv" in params:
        kwargs["fallback_hv"] = fallback_hv
    if "min_dte" in params:
        kwargs["min_dte"] = min_dte
    if "max_dte" in params:
        kwargs["max_dte"] = max_dte
    if target_abs_delta is not None and "target_abs_delta" in params:
        kwargs["target_abs_delta"] = target_abs_delta
    if max_spread_pct is not None and "max_spread_pct" in params:
        kwargs["max_spread_pct"] = max_spread_pct
    if reference_date is not None and "reference_date" in params:
        kwargs["reference_date"] = reference_date

    raw = fn(**kwargs)
    return normalize_option_quote(raw)


def call_update_paper(
    scan_output: Dict[str, Any],
    config: Dict[str, Any],
    path: Optional[str] = None,
    enforce_window: Optional[bool] = None,
) -> Dict[str, Any]:
    fn = paper_trader.update_paper_portfolio
    sig = inspect.signature(fn)
    kwargs: Dict[str, Any] = {}
    if path is not None and "path" in sig.parameters:
        kwargs["path"] = path
    if enforce_window is not None and "enforce_window" in sig.parameters:
        kwargs["enforce_window"] = enforce_window
    return fn(scan_output, config, **kwargs)


def call_flush_paper(
    config: Dict[str, Any],
    path: Optional[str] = None,
    latest_scan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    fn = paper_trader.flush_open_paper_positions_at_session_end
    sig = inspect.signature(fn)
    kwargs: Dict[str, Any] = {}
    if path is not None and "path" in sig.parameters:
        kwargs["path"] = path
    if latest_scan is not None and "latest_scan" in sig.parameters:
        kwargs["latest_scan"] = latest_scan
    return fn(config, **kwargs)


# ---------------------------------------------------------------------------
# Synthetic 2m OHLCV DataFrame Generator
# ---------------------------------------------------------------------------

def _ny_day_grid(date_str: str, count: Optional[int] = None) -> pd.DatetimeIndex:
    grid = pd.date_range(
        f"{date_str} 09:30",
        f"{date_str} 15:58",
        freq="2min",
        tz="America/New_York",
    )
    if count is not None:
        return grid[:count]
    return grid


def _exact_rvol_volume(prior_vols: Sequence[float], target_rvol: float, vol_ma_period: int = 20) -> float:
    """Compute exact last-bar Volume so V / mean(last_{N-1}, V) == target_rvol."""
    n = int(vol_ma_period)
    window_prev = list(prior_vols[-(n - 1):])
    sum_prev = float(sum(window_prev))
    if target_rvol >= n:
        target_rvol = n - 0.1
    return max(0.0, (sum_prev * target_rvol) / (n - target_rvol))


def make_intraday_2m_df(
    prev_date: str = "2026-11-17",
    curr_date: str = "2026-11-18",
    trend: str = "DOWNTREND",
    active_level: float = 106.18,
    candle_mode: str = "CONFIRMED_REVERSAL",
    rvol_target: float = 1.30,
    vol_ma_period: int = 20,
    curr_bars: int = 25,
    prev_bars: int = 195,
    morning_gap_pct: Optional[float] = None,
    gap_direction: str = "UP",
    spot_override: Optional[float] = None,
    tz_mode: str = "NY",
    multiindex: bool = False,
    ticker: str = "AMD",
    include_extended_hours: bool = False,
    inject_duplicates_and_nans: bool = False,
) -> pd.DataFrame:
    """
    Build deterministic regular-hours 2m OHLCV bars (>=200 bars default) with:
    - Exact previous-day asymmetric anchors:
      wick_low=100.00, body_low=101.00, body_high=110.00, wick_high=112.00
    - Exact EMA200 direction: DOWNTREND (spot < ema200), UPTREND (spot > ema200), NEUTRAL (spot == ema200)
    - Exact RVOL on final bar (or max of last 2 bars)
    - Exact reversal candle pattern relative to `active_level`
    """
    prev_idx = _ny_day_grid(prev_date, count=prev_bars)
    curr_idx = _ny_day_grid(curr_date, count=curr_bars)

    rows: List[Dict[str, float]] = []

    if trend == "NEUTRAL":
        for i in range(len(prev_idx)):
            if i == 10:
                rows.append({"Open": 105.0, "High": 107.0, "Low": 104.0, "Close": 105.0, "Volume": 1000.0})
            elif i == 11:
                rows.append({"Open": 105.0, "High": 106.0, "Low": 103.0, "Close": 105.0, "Volume": 1000.0})
            else:
                rows.append({"Open": 105.0, "High": 105.0, "Low": 105.0, "Close": 105.0, "Volume": 1000.0})
        for _ in range(len(curr_idx)):
            rows.append({"Open": 105.0, "High": 105.0, "Low": 105.0, "Close": 105.0, "Volume": 1000.0})
        df = pd.DataFrame(rows, index=prev_idx.append(curr_idx))
        return _apply_df_variants(
            df,
            prev_date=prev_date,
            tz_mode=tz_mode,
            multiindex=multiindex,
            ticker=ticker,
            include_extended_hours=include_extended_hours,
            inject_duplicates_and_nans=inject_duplicates_and_nans,
        )

    base_close = 110.0 if trend == "DOWNTREND" else 101.0
    for i in range(len(prev_idx)):
        if i == len(prev_idx) - 3 and len(prev_idx) >= 3:
            # Anchor low bar: wick_low=100.00, body_low=101.00
            rows.append({"Open": 102.0, "High": 103.0, "Low": 100.0, "Close": 101.0, "Volume": 1000.0})
        elif i == len(prev_idx) - 2 and len(prev_idx) >= 2:
            # Anchor high bar: wick_high=112.00, body_high=110.00
            rows.append({"Open": 109.0, "High": 112.0, "Low": 108.0, "Close": 110.0, "Volume": 1000.0})
        elif i == len(prev_idx) - 1:
            # Last prev bar closes near 105.00 so default gap is 0.0%
            rows.append({"Open": 105.0, "High": 105.5, "Low": 104.5, "Close": 105.0, "Volume": 1000.0})
        else:
            rows.append({
                "Open": base_close,
                "High": base_close + 0.5,
                "Low": base_close - 0.5,
                "Close": base_close,
                "Volume": 1000.0,
            })

    prev_close = rows[-1]["Close"]
    if morning_gap_pct is not None:
        sign = 1.0 if gap_direction.upper() == "UP" else -1.0
        first_open = round(prev_close * (1.0 + sign * (morning_gap_pct / 100.0)), 4)
    else:
        first_open = prev_close

    # Baseline drift level for early curr bars (stays on intended EMA side, away from active_level until end)
    drift_px = spot_override if spot_override is not None else (
        active_level - 1.20 if trend == "DOWNTREND" else active_level + 1.20
    )
    if candle_mode == "NO_TOUCH":
        drift_px = spot_override if spot_override is not None else (
            active_level - 3.00 if trend == "DOWNTREND" else active_level + 3.00
        )

    for j in range(len(curr_idx)):
        o = first_open if j == 0 else drift_px
        rows.append({
            "Open": o,
            "High": max(o, drift_px) + 0.10,
            "Low": min(o, drift_px) - 0.10,
            "Close": drift_px,
            "Volume": 1000.0,
        })

    # Shape final 2 bars for requested candle_mode
    if len(curr_idx) >= 2:
        prev_c_idx = len(prev_idx) + len(curr_idx) - 2
        last_c_idx = len(prev_idx) + len(curr_idx) - 1
        lvl = float(active_level)

        if candle_mode == "CONFIRMED_REVERSAL":
            if trend == "DOWNTREND":
                rows[prev_c_idx].update({"Open": lvl - 0.30, "High": lvl + 0.02, "Low": lvl - 0.35, "Close": lvl - 0.05})
                rows[last_c_idx].update({"Open": lvl - 0.03, "High": lvl + 0.35, "Low": lvl - 0.13, "Close": lvl - 0.08})
            else:
                rows[prev_c_idx].update({"Open": lvl + 0.30, "High": lvl + 0.35, "Low": lvl - 0.02, "Close": lvl + 0.05})
                rows[last_c_idx].update({"Open": lvl + 0.03, "High": lvl + 0.13, "Low": lvl - 0.35, "Close": lvl + 0.08})
        elif candle_mode == "ENGULFING":
            if trend == "DOWNTREND":
                rows[prev_c_idx].update({"Open": lvl - 0.15, "High": lvl + 0.03, "Low": lvl - 0.20, "Close": lvl + 0.01})
                rows[last_c_idx].update({"Open": lvl + 0.02, "High": lvl + 0.04, "Low": lvl - 0.30, "Close": lvl - 0.25})
            else:
                rows[prev_c_idx].update({"Open": lvl + 0.15, "High": lvl + 0.20, "Low": lvl - 0.03, "Close": lvl - 0.01})
                rows[last_c_idx].update({"Open": lvl - 0.02, "High": lvl + 0.30, "Low": lvl - 0.04, "Close": lvl + 0.25})
        elif candle_mode == "TOUCHED_ONLY":
            # Touches lvl cleanly, small neutral body, low wick ratio (<0.25), does not engulf
            if trend == "DOWNTREND":
                rows[prev_c_idx].update({"Open": lvl - 0.20, "High": lvl - 0.05, "Low": lvl - 0.25, "Close": lvl - 0.10})
                rows[last_c_idx].update({"Open": lvl - 0.08, "High": lvl + 0.02, "Low": lvl - 0.12, "Close": lvl - 0.06})
            else:
                rows[prev_c_idx].update({"Open": lvl + 0.20, "High": lvl + 0.25, "Low": lvl + 0.05, "Close": lvl + 0.10})
                rows[last_c_idx].update({"Open": lvl + 0.08, "High": lvl + 0.12, "Low": lvl - 0.02, "Close": lvl + 0.06})
        elif candle_mode == "NO_TOUCH":
            px = spot_override if spot_override is not None else (lvl - 3.00 if trend == "DOWNTREND" else lvl + 3.00)
            if trend == "DOWNTREND":
                rows[prev_c_idx].update({"Open": px + 0.10, "High": px + 0.15, "Low": px - 0.10, "Close": px + 0.05})
                rows[last_c_idx].update({"Open": px + 0.08, "High": px + 0.10, "Low": px - 0.15, "Close": px - 0.12})
            else:
                rows[prev_c_idx].update({"Open": px - 0.10, "High": px + 0.10, "Low": px - 0.15, "Close": px - 0.05})
                rows[last_c_idx].update({"Open": px - 0.08, "High": px + 0.15, "Low": px - 0.10, "Close": px + 0.12})
        elif candle_mode == "NAN_CLOSE":
            rows[last_c_idx]["Close"] = float("nan")

        if spot_override is not None and candle_mode != "NAN_CLOSE":
            rows[last_c_idx]["Close"] = float(spot_override)
            rows[last_c_idx]["High"] = max(rows[last_c_idx]["High"], float(spot_override))
            rows[last_c_idx]["Low"] = min(rows[last_c_idx]["Low"], float(spot_override))

        # Exact RVOL on final bar
        if rvol_target < 1.0:
            rows[prev_c_idx]["Volume"] = 1000.0 * rvol_target
        prior_vols = [r["Volume"] for r in rows[:-1]]
        rows[last_c_idx]["Volume"] = _exact_rvol_volume(prior_vols, rvol_target, vol_ma_period=vol_ma_period)

    full_idx = prev_idx.append(curr_idx)
    df = pd.DataFrame(rows, index=full_idx)
    return _apply_df_variants(
        df,
        prev_date=prev_date,
        tz_mode=tz_mode,
        multiindex=multiindex,
        ticker=ticker,
        include_extended_hours=include_extended_hours,
        inject_duplicates_and_nans=inject_duplicates_and_nans,
    )


def _apply_df_variants(
    df: pd.DataFrame,
    prev_date: str,
    tz_mode: str,
    multiindex: bool,
    ticker: str,
    include_extended_hours: bool,
    inject_duplicates_and_nans: bool,
) -> pd.DataFrame:
    if include_extended_hours:
        ext_pre = pd.date_range(f"{prev_date} 08:00", f"{prev_date} 09:28", freq="2min", tz="America/New_York")
        ext_post = pd.date_range(f"{prev_date} 16:00", f"{prev_date} 16:30", freq="2min", tz="America/New_York")
        pre_df = pd.DataFrame(
            {"Open": 60.0, "High": 200.0, "Low": 50.0, "Close": 60.0, "Volume": 5000.0},
            index=ext_pre,
        )
        post_df = pd.DataFrame(
            {"Open": 180.0, "High": 220.0, "Low": 40.0, "Close": 180.0, "Volume": 5000.0},
            index=ext_post,
        )
        df = pd.concat([pre_df, df, post_df]).sort_index()

    if inject_duplicates_and_nans and len(df) >= 5:
        dup_row = df.iloc[[2]].copy()
        nan_ts = df.index[3] + pd.Timedelta(seconds=30)
        nan_row = pd.DataFrame(
            {"Open": np.nan, "High": np.nan, "Low": np.nan, "Close": np.nan, "Volume": np.nan},
            index=pd.DatetimeIndex([nan_ts], tz=df.index.tz),
        )
        df = pd.concat([df, dup_row, nan_row])
        # Deliberately shuffle slightly to test sorting
        df = pd.concat([df.iloc[1:], df.iloc[:1]])

    tz_upper = tz_mode.upper()
    if tz_upper == "UTC":
        df = df.copy()
        df.index = df.index.tz_convert("UTC")
    elif tz_upper == "NAIVE":
        df = df.copy()
        df.index = df.index.tz_localize(None)

    if multiindex:
        df = df.copy()
        df.columns = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], [ticker]])

    return df


# ---------------------------------------------------------------------------
# Option Chain & yfinance Mock Factory
# ---------------------------------------------------------------------------

def make_option_chain_df(
    spot: float,
    option_type: str = "CALL",
    strikes: Optional[Sequence[float]] = None,
    dte_days: float = 3.0,
    iv: float = 0.38,
    spread_pct: float = 2.0,
    volume: int = 200,
    oi: int = 500,
    ticker: str = "AMD",
) -> pd.DataFrame:
    if strikes is None:
        base = round(spot)
        strikes = [base - 5.0, base - 2.0, float(base), base + 2.0, base + 5.0]
    rows = []
    for k in strikes:
        theo = options_pricing.black_scholes_price(spot, float(k), dte_days, iv, option_type)
        bid = round(max(0.01, theo * (1.0 - spread_pct / 200.0)), 2)
        ask = round(max(bid + 0.01, theo * (1.0 + spread_pct / 200.0)), 2)
        rows.append({
            "contractSymbol": f"{ticker}261120{'C' if option_type.upper() == 'CALL' else 'P'}{int(k * 1000):08d}",
            "strike": float(k),
            "lastPrice": round(theo, 2),
            "bid": bid,
            "ask": ask,
            "volume": volume,
            "openInterest": oi,
            "impliedVolatility": iv,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def mock_yfinance_factory(monkeypatch):
    """Patch yf.Ticker across all bot modules with configurable per-ticker specs and call log."""

    def _install(ticker_specs: Dict[str, Dict[str, Any]]):
        call_log: List[Dict[str, Any]] = []

        class FakeTicker:
            def __init__(self, tk: str):
                self.tk = tk.upper().strip()
                self.spec = ticker_specs.get(self.tk, ticker_specs.get(tk, {}))

            def history(self, period: str = "5d", interval: str = "2m", **kwargs):
                call_log.append({"ticker": self.tk, "method": "history", "period": period, "interval": interval})
                if interval == "5m" and "history_5m" in self.spec:
                    val = self.spec["history_5m"]
                else:
                    val = self.spec.get("history_2m", pd.DataFrame())
                if callable(val):
                    res = val(period=period, interval=interval)
                    if isinstance(res, Exception):
                        raise res
                    return res
                if isinstance(val, Exception):
                    raise val
                return val.copy() if isinstance(val, pd.DataFrame) else val

            @property
            def calendar(self):
                call_log.append({"ticker": self.tk, "method": "calendar"})
                cal = self.spec.get("calendar")
                if callable(cal):
                    res = cal()
                    if isinstance(res, Exception):
                        raise res
                    return res
                if isinstance(cal, Exception):
                    raise cal
                return cal

            @property
            def options(self):
                call_log.append({"ticker": self.tk, "method": "options"})
                opts = self.spec.get("options", ())
                if isinstance(opts, Exception):
                    raise opts
                return opts

            def option_chain(self, exp: str):
                call_log.append({"ticker": self.tk, "method": "option_chain", "exp": exp})
                chains = self.spec.get("option_chains", {})
                val = chains.get(exp)
                if isinstance(val, Exception):
                    raise val
                if val is None:
                    return SimpleNamespace(calls=pd.DataFrame(), puts=pd.DataFrame())
                return val

        monkeypatch.setattr(yf, "Ticker", FakeTicker)
        for mod in (fibo_scanner, earnings_filter, options_pricing, backtester):
            if hasattr(mod, "yf"):
                monkeypatch.setattr(mod.yf, "Ticker", FakeTicker)
        return call_log

    return _install


# ---------------------------------------------------------------------------
# Telegram HTTP Mock + Call Recorder
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status_code: int = 200, payload: Optional[Dict[str, Any]] = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"ok": status_code == 200}
        self.text = text or json.dumps(self._payload)

    def json(self) -> Dict[str, Any]:
        return self._payload


@pytest.fixture
def telegram_mock(monkeypatch):
    """Intercept Telegram Bot API GET/POST calls and record all payloads."""

    def _install(
        webhook_url: str = "",
        webhook_status: int = 200,
        webhook_error: Optional[Exception] = None,
        updates_batches: Optional[List[Any]] = None,
        post_statuses: Optional[List[Any]] = None,
    ):
        state = {
            "calls": [],
            "webhook_url": webhook_url,
            "updates_batches": list(updates_batches or [[]]),
            "post_statuses": list(post_statuses or []),
        }

        def _fake_get(url: str, params=None, timeout=None, **kwargs):
            state["calls"].append({"method": "GET", "url": url, "params": params, "timeout": timeout})
            if url.endswith("/getWebhookInfo"):
                if webhook_error is not None:
                    raise webhook_error
                return FakeResponse(
                    status_code=webhook_status,
                    payload={"ok": webhook_status == 200, "result": {"url": state["webhook_url"]}},
                )
            if url.endswith("/getUpdates"):
                nxt = state["updates_batches"].pop(0) if state["updates_batches"] else []
                if isinstance(nxt, Exception):
                    raise nxt
                if isinstance(nxt, tuple):
                    code, body = nxt
                    return FakeResponse(status_code=code, payload=body)
                return FakeResponse(status_code=200, payload={"ok": True, "result": nxt})
            raise RuntimeError(f"Unexpected GET URL in telegram_mock: {url}")

        def _fake_post(url: str, json=None, timeout=None, **kwargs):
            state["calls"].append({"method": "POST", "url": url, "json": json, "timeout": timeout})
            if state["post_statuses"]:
                nxt = state["post_statuses"].pop(0)
                if isinstance(nxt, Exception):
                    raise nxt
                return FakeResponse(status_code=int(nxt), payload={"ok": int(nxt) == 200})
            return FakeResponse(status_code=200, payload={"ok": True, "result": {"message_id": len(state["calls"])}})

        monkeypatch.setattr(requests, "get", _fake_get)
        monkeypatch.setattr(requests, "post", _fake_post)
        if hasattr(telegram_notifier, "requests"):
            monkeypatch.setattr(telegram_notifier.requests, "post", _fake_post)
            monkeypatch.setattr(telegram_notifier.requests, "get", _fake_get)
        if hasattr(command_handler, "requests"):
            monkeypatch.setattr(command_handler.requests, "get", _fake_get)
            monkeypatch.setattr(command_handler.requests, "post", _fake_post)
        return state

    return _install


# ---------------------------------------------------------------------------
# NY Clock Freeze Helper
# ---------------------------------------------------------------------------

@pytest.fixture
def freeze_ny_time(monkeypatch):
    """Deterministically freeze America/New_York time across time_utils & consumer modules."""

    def _freeze(dt_ny: datetime):
        aware_ny = dt_ny.astimezone(NY_TZ) if dt_ny.tzinfo else dt_ny.replace(tzinfo=NY_TZ)
        clock = {"now": aware_ny}

        def _now_ny() -> datetime:
            return clock["now"]

        def _today_ny() -> date:
            return clock["now"].date()

        tu = importlib.import_module("day_option_bot.src.time_utils") if _module_exists("day_option_bot.src.time_utils") else None
        if tu is not None:
            monkeypatch.setattr(tu, "now_ny", _now_ny)
            monkeypatch.setattr(tu, "today_ny", _today_ny)

        for mod in (fibo_scanner, earnings_filter, options_pricing, paper_trader, bot_main):
            if hasattr(mod, "now_ny"):
                monkeypatch.setattr(mod, "now_ny", _now_ny)
            if hasattr(mod, "today_ny"):
                monkeypatch.setattr(mod, "today_ny", _today_ny)

        return clock

    return _freeze


def _module_exists(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Node.js GAS Sandbox Runner (`gas_webhook/Code.gs`)
# ---------------------------------------------------------------------------

_NODE_GAS_DRIVER = r"""
const fs = require("fs");
const vm = require("vm");

const req = JSON.parse(fs.readFileSync(0, "utf8"));
const code = fs.readFileSync(req.codePath, "utf8");

const propsStore = Object.assign({}, req.properties || {});
const triggers = (req.triggers || []).map((t, i) => ({
  handlerFunction: t.handlerFunction,
  everyMinutes: t.everyMinutes || 5,
  uniqueId: t.uniqueId || `trig_${i + 1}`,
  eventType: t.eventType || "CLOCK",
  getHandlerFunction() { return this.handlerFunction; },
  getUniqueId() { return this.uniqueId; },
  getEventType() { return this.eventType; },
}));
const httpRequests = [];
const logs = [];
const mockResponses = req.mockResponses || [];
const frozenMs = req.frozenIso ? new Date(req.frozenIso).getTime() : Date.now();

const RealDate = Date;
class MockDate extends RealDate {
  constructor(...args) {
    if (args.length === 0) {
      super(frozenMs);
    } else {
      super(...args);
    }
  }
  static now() { return frozenMs; }
}

function getTzParts(dateObj, tz) {
  const d = dateObj instanceof RealDate ? dateObj : new RealDate(dateObj);
  const fmt = new Intl.DateTimeFormat("en-US", {
    timeZone: tz,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hourCycle: "h23",
    weekday: "short",
    timeZoneName: "short",
  });
  const map = {};
  for (const p of fmt.formatToParts(d)) {
    map[p.type] = p.value;
  }
  const wk = { Mon: "1", Tue: "2", Wed: "3", Thu: "4", Fri: "5", Sat: "6", Sun: "7" }[map.weekday] || "1";
  return {
    yyyy: map.year,
    MM: map.month,
    dd: map.day,
    HH: map.hour,
    mm: map.minute,
    ss: map.second,
    u: wk,
    EEE: map.weekday,
    z: map.timeZoneName || tz,
  };
}

function formatGasDate(dateObj, tz, fmt) {
  const p = getTzParts(dateObj, tz);
  let out = "";
  let i = 0;
  const s = String(fmt);
  const tokens = [
    ["yyyy", p.yyyy],
    ["EEE", p.EEE],
    ["MM", p.MM],
    ["dd", p.dd],
    ["HH", p.HH],
    ["mm", p.mm],
    ["ss", p.ss],
    ["u", p.u],
    ["z", p.z],
  ];
  while (i < s.length) {
    if (s[i] === "'") {
      const end = s.indexOf("'", i + 1);
      if (end === -1) throw new Error("Unsupported Utilities.formatDate pattern (unclosed quote): " + fmt);
      out += s.slice(i + 1, end);
      i = end + 1;
      continue;
    }
    let matched = false;
    for (const [tok, val] of tokens) {
      if (s.startsWith(tok, i)) {
        out += val;
        i += tok.length;
        matched = true;
        break;
      }
    }
    if (matched) continue;
    const ch = s[i];
    if ((ch >= "A" && ch <= "Z") || (ch >= "a" && ch <= "z")) {
      throw new Error("Unsupported Utilities.formatDate pattern: " + fmt);
    }
    out += ch;
    i += 1;
  }
  return out;
}

const sandbox = {
  Date: MockDate,
  Math,
  JSON,
  Number,
  String,
  Boolean,
  Array,
  Object,
  RegExp,
  Error,
  parseInt,
  parseFloat,
  isNaN,
  encodeURIComponent,
  decodeURIComponent,
  console: {
    log: (...a) => logs.push({ level: "log", msg: a.map(String).join(" ") }),
    warn: (...a) => logs.push({ level: "warn", msg: a.map(String).join(" ") }),
    error: (...a) => logs.push({ level: "error", msg: a.map(String).join(" ") }),
  },
  Logger: {
    log: (...a) => logs.push({ level: "logger", msg: a.map(String).join(" ") }),
    getLog: () => logs.filter((l) => l.level === "logger").map((l) => l.msg).join("\n"),
  },
  LockService: {
    getScriptLock() {
      return {
        _locked: false,
        tryLock() { this._locked = true; return true; },
        waitLock() { this._locked = true; },
        releaseLock() { this._locked = false; },
        hasLock() { return this._locked; },
      };
    },
  },
  PropertiesService: {
    getScriptProperties() {
      return {
        getProperty(k) { return Object.prototype.hasOwnProperty.call(propsStore, k) ? propsStore[k] : null; },
        setProperty(k, v) { propsStore[k] = String(v); },
        deleteProperty(k) { delete propsStore[k]; },
        getProperties() { return Object.assign({}, propsStore); },
      };
    },
  },
  ScriptApp: {
    EventType: { CLOCK: "CLOCK" },
    getProjectTriggers() { return triggers.slice(); },
    deleteTrigger(t) {
      const idx = triggers.indexOf(t);
      if (idx !== -1) triggers.splice(idx, 1);
    },
    newTrigger(fnName) {
      return {
        timeBased() {
          return {
            everyMinutes(n) {
              return {
                create() {
                  const trig = {
                    handlerFunction: fnName,
                    everyMinutes: n,
                    uniqueId: `trig_${triggers.length + 1}`,
                    eventType: "CLOCK",
                    getHandlerFunction() { return fnName; },
                    getUniqueId() { return this.uniqueId; },
                    getEventType() { return this.eventType; },
                  };
                  triggers.push(trig);
                  return trig;
                },
              };
            },
          };
        },
      };
    },
  },
  Utilities: {
    formatDate(dateObj, tz, fmt) {
      return formatGasDate(dateObj, tz, fmt);
    },
    formatString(fmt, ...args) {
      let idx = 0;
      return String(fmt).replace(/%(%|[-+ 0#]*\d*(?:\.\d+)?[sdfi])/g, (match, spec) => {
        if (spec === "%") return "%";
        const val = args[idx++];
        const type = spec[spec.length - 1];
        const flagsAndWidth = spec.slice(0, -1);
        if (type === "s") return String(val);
        const num = Number(val);
        if (type === "d" || type === "i") {
          const intStr = String(Math.trunc(num));
          const widthMatch = flagsAndWidth.match(/0?(\d+)$/);
          if (widthMatch) {
            const w = parseInt(widthMatch[1], 10);
            const pad = flagsAndWidth.includes("0") ? "0" : " ";
            return intStr.padStart(w, pad);
          }
          return intStr;
        }
        if (type === "f") {
          const precMatch = flagsAndWidth.match(/\.(\d+)/);
          const prec = precMatch ? parseInt(precMatch[1], 10) : 6;
          return num.toFixed(prec);
        }
        return String(val);
      });
    },
    base64Decode(str) { return Buffer.from(str, "base64"); },
    base64Encode(bytes) { return Buffer.from(bytes).toString("base64"); },
    newBlob(data, contentType) {
      const buf = Buffer.isBuffer(data) ? data : Buffer.from(String(data), "utf8");
      return {
        getBytes() { return buf; },
        getDataAsString() { return buf.toString("utf8"); },
      };
    },
  },
  ContentService: {
    MimeType: { JSON: "JSON", TEXT: "TEXT" },
    createTextOutput(txt) {
      return {
        _text: txt,
        _mime: "TEXT",
        setMimeType(m) { this._mime = m; return this; },
        getContent() { return this._text; },
        getMimeType() { return this._mime; },
      };
    },
  },
  HtmlService: {
    createHtmlOutput(html) {
      return {
        _html: html,
        _title: "",
        setTitle(t) { this._title = String(t); return this; },
        getTitle() { return this._title; },
        getContent() { return this._html; },
      };
    },
  },
  UrlFetchApp: {
    fetch(url, opts) {
      const rec = {
        url,
        method: (opts && opts.method) ? String(opts.method).toLowerCase() : "get",
        headers: (opts && opts.headers) || {},
        payload: opts && opts.payload ? opts.payload : null,
      };
      httpRequests.push(rec);
      let matched = null;
      for (const m of mockResponses) {
        if (url.includes(m.match)) {
          matched = m;
          break;
        }
      }
      const status = matched ? matched.status : 200;
      const bodyText = matched
        ? (typeof matched.body === "string" ? matched.body : JSON.stringify(matched.body))
        : JSON.stringify({ ok: true });
      return {
        getResponseCode() { return status; },
        getContentText() { return bodyText; },
      };
    },
  },
};

const ctx = vm.createContext(sandbox);
vm.runInContext(code, ctx, { filename: "Code.gs" });

const result = vm.runInContext(req.callExpr, ctx);
const telegramSends = httpRequests
  .filter((r) => r.url.includes("api.telegram.org") && r.url.includes("/sendMessage"))
  .map((r) => ({ url: r.url, body: r.payload ? JSON.parse(r.payload) : {} }));
const githubDispatches = httpRequests
  .filter((r) => r.url.includes("api.github.com") && r.url.includes("/dispatches"))
  .map((r) => ({ url: r.url, headers: r.headers, body: r.payload ? JSON.parse(r.payload) : {} }));

let outputContent = null;
let outputMime = null;
if (result && typeof result.getContent === "function") {
  outputContent = result.getContent();
  if (typeof result.getMimeType === "function") outputMime = result.getMimeType();
}

process.stdout.write(JSON.stringify({
  result: outputContent !== null ? { content: outputContent, mime: outputMime } : result,
  httpRequests,
  telegramSends,
  githubDispatches,
  properties: propsStore,
  triggers: triggers.map((t) => ({ handlerFunction: t.handlerFunction, everyMinutes: t.everyMinutes })),
  logs,
}));
"""


def run_gas_function(
    call_expr: str,
    frozen_iso: str = "2026-11-18T09:35:00-05:00",
    properties: Optional[Dict[str, str]] = None,
    mock_responses: Optional[List[Dict[str, Any]]] = None,
    triggers: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Execute gas_webhook/Code.gs inside a deterministic Node.js vm sandbox."""
    payload = {
        "codePath": GAS_CODE_PATH,
        "callExpr": call_expr,
        "frozenIso": frozen_iso,
        "properties": properties if properties is not None else {"GH_TOKEN": "ghp_test_token_123"},
        "mockResponses": mock_responses or [],
        "triggers": triggers or [],
    }
    proc = subprocess.run(
        ["node", "-e", _NODE_GAS_DRIVER],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        encoding="utf-8",
        timeout=20,
    )
    if proc.returncode != 0:
        raise AssertionError(f"Node GAS VM failed (rc={proc.returncode}):\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}")
    return json.loads(proc.stdout)


# ---------------------------------------------------------------------------
# Cloud Workflow YAML + Git Bash / Shell Helpers
# ---------------------------------------------------------------------------

def load_cloud_workflow() -> Dict[str, Any]:
    with open(WORKFLOW_YAML_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_execute_step_run_script(workflow_dict: Dict[str, Any]) -> str:
    steps = workflow_dict["jobs"]["run-day-option-bot"]["steps"]
    for step in steps:
        if "Execute Day Option Bot" in str(step.get("name", "")):
            return str(step["run"])
    raise KeyError("Execute Day Option Bot step not found in day_option_cloud.yml")


def cli_has_flag(cmd: str, flag: str, value: Optional[str] = None) -> bool:
    """Return True if CLI command string contains `--flag value` or `--flag=value`."""
    tokens = shlex.split(cmd)
    if value is None:
        return flag in tokens or any(t.startswith(f"{flag}=") for t in tokens)
    expected_eq = f"{flag}={value}"
    for i, tok in enumerate(tokens):
        if tok == expected_eq:
            return True
        if tok == flag and i + 1 < len(tokens) and tokens[i + 1] == value:
            return True
    return False


def run_cloud_shell_simulation(
    tmp_path: Path,
    event_name: str,
    inputs_mode: str = "scan_once",
    ny_date: str = "2026-11-18",
    ny_h: int = 9,
    ny_m: int = 35,
    last_run_date: str = "",
) -> Dict[str, Any]:
    """
    Execute the actual `Execute Day Option Bot` bash block from day_option_cloud.yml
    under Git Bash with deterministic `date`, `python`, and `sleep` shims.
    """
    if not os.path.exists(GIT_BASH_EXE):
        pytest.skip("Git Bash not found on host; skipping live bash execution")

    wf = load_cloud_workflow()
    raw_script = get_execute_step_run_script(wf)
    script = (
        raw_script.replace("${{ github.event_name }}", event_name)
        .replace("${{ github.event.inputs.mode || inputs.mode }}", inputs_mode)
        .replace("${{ github.event.inputs.mode }}", inputs_mode)
        .replace("${{ inputs.mode }}", inputs_mode)
    )

    work_dir = tmp_path / "cloud_sim"
    bin_dir = work_dir / "bin"
    cfg_dir = work_dir / "config"
    bin_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    trace_file = work_dir / "trace.log"

    if last_run_date:
        (cfg_dir / "latest_scan.json").write_text(
            json.dumps({"ny_session_date": last_run_date}),
            encoding="utf-8",
        )

    ny_dt = datetime.strptime(ny_date, "%Y-%m-%d").replace(hour=int(ny_h), minute=int(ny_m), tzinfo=NY_TZ)
    ny_u = str(ny_dt.isoweekday())
    ny_epoch = str(int(ny_dt.timestamp()))
    ny_iso = ny_dt.isoformat()

    date_shim = bin_dir / "date"
    date_shim.write_text(
        "#!/bin/sh\n"
        'fmt=""\n'
        'while [ "$#" -gt 0 ]; do\n'
        '  case "$1" in\n'
        '    -u|-R|-I|--utc|--universal) shift ;;\n'
        '    -d|--date) shift 2 ;;\n'
        '    --date=*) shift ;;\n'
        '    +*) fmt="$1"; shift ;;\n'
        '    *) shift ;;\n'
        '  esac\n'
        'done\n'
        'case "$fmt" in\n'
        '  +%Y-%m-%d) echo "$FAKE_NY_DATE" ;;\n'
        '  +%-H) echo "$FAKE_NY_H" ;;\n'
        '  +%H) printf "%02d\\n" "$FAKE_NY_H" ;;\n'
        '  +%-M) echo "$FAKE_NY_M" ;;\n'
        '  +%M) printf "%02d\\n" "$FAKE_NY_M" ;;\n'
        '  +%-S) echo "0" ;;\n'
        '  +%S) echo "00" ;;\n'
        '  +%H:%M) printf "%02d:%02d\\n" "$FAKE_NY_H" "$FAKE_NY_M" ;;\n'
        '  +%H:%M:%S) printf "%02d:%02d:00\\n" "$FAKE_NY_H" "$FAKE_NY_M" ;;\n'
        '  +%u) echo "$FAKE_NY_U" ;;\n'
        '  +%s) echo "$FAKE_NY_EPOCH" ;;\n'
        '  *) echo "Unsupported date format in date_shim: $fmt" >&2; exit 2 ;;\n'
        "esac\n",
        encoding="utf-8",
    )

    py_runner = work_dir / "cloud_py_runner.py"
    py_runner.write_text(
        "import datetime as _dt\n"
        "import os, sys\n"
        "_fake_now = _dt.datetime.fromisoformat(os.environ['FAKE_NY_ISO'])\n"
        "class _FakeDT(_dt.datetime):\n"
        "    @classmethod\n"
        "    def now(cls, tz=None):\n"
        "        return _fake_now.astimezone(tz) if tz is not None else _fake_now.replace(tzinfo=None)\n"
        "_dt.datetime = _FakeDT\n"
        "exec(sys.argv[1], {'__name__': '__main__'})\n",
        encoding="utf-8",
    )

    python_shim = bin_dir / "python"
    python_shim.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-c" ]; then\n'
        '  shift\n'
        '  code="$1"\n'
        '  case "$code" in\n'
        '    *latest_scan.json*)\n'
        '      if [ -n "$FAKE_LAST_RUN_DATE" ]; then\n'
        '        echo "$FAKE_LAST_RUN_DATE"\n'
        "      fi\n"
        "      exit 0\n"
        "      ;;\n"
        "    *)\n"
        '      exec "$REAL_PYTHON" "$CLOUD_PY_RUNNER" "$code"\n'
        "      ;;\n"
        "  esac\n"
        "fi\n"
        'echo "PYTHON:$*" >> "$TRACE_FILE"\n'
        "exit 0\n",
        encoding="utf-8",
    )

    sleep_shim = bin_dir / "sleep"
    sleep_shim.write_text(
        "#!/bin/sh\n"
        'echo "SLEEP:$1" >> "$TRACE_FILE"\n'
        "exit 0\n",
        encoding="utf-8",
    )

    for p in (date_shim, python_shim, sleep_shim):
        os.chmod(p, 0o755)

    def _to_posix(p: Path) -> str:
        s = str(p).replace("\\", "/")
        if len(s) >= 2 and s[1] == ":":
            return f"/{s[0].lower()}{s[2:]}"
        return s

    posix_bin = _to_posix(bin_dir)
    script_with_path = f'export PATH="{posix_bin}:$PATH"\n{script}'

    env = os.environ.copy()
    env["FAKE_NY_DATE"] = ny_date
    env["FAKE_NY_H"] = str(ny_h)
    env["FAKE_NY_M"] = str(ny_m)
    env["FAKE_NY_U"] = ny_u
    env["FAKE_NY_EPOCH"] = ny_epoch
    env["FAKE_NY_ISO"] = ny_iso
    env["FAKE_LAST_RUN_DATE"] = last_run_date
    env["REAL_PYTHON"] = _to_posix(Path(sys.executable))
    env["CLOUD_PY_RUNNER"] = _to_posix(py_runner)
    env["TRACE_FILE"] = _to_posix(trace_file)

    proc = subprocess.run(
        [GIT_BASH_EXE, "-c", script_with_path],
        cwd=str(work_dir),
        env=env,
        text=True,
        capture_output=True,
        encoding="utf-8",
        timeout=15,
    )
    traces = trace_file.read_text(encoding="utf-8").splitlines() if trace_file.exists() else []
    python_cmds = [line[len("PYTHON:"):] for line in traces if line.startswith("PYTHON:")]
    sleeps = [int(float(line[len("SLEEP:"):])) for line in traces if line.startswith("SLEEP:")]
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "python_cmds": python_cmds,
        "sleeps": sleeps,
    }
