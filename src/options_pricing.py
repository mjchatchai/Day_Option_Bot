"""
Real-Time Black-Scholes-Merton Option Pricing & Greeks Engine for Day Option Bot.
Eliminates the 15-minute OPRA free-feed delay by combining the option chain's Implied Volatility (IV)
structure with the 0-second Real-Time Underlying Stock Price (Spot) and projecting exact Option Prices
at Entry, TP1 (Next Fibo Line), TP2 (Runner Fibo Line), and SL (Opposite Fibo Line).
"""

from dataclasses import dataclass
from datetime import datetime, date, timedelta
import math
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import yfinance as yf

from .time_utils import NY_TZ, today_ny


RISK_FREE_RATE = 0.043  # ~4.3% US Treasury yield baseline
ETF_TICKERS = {"SPY", "QQQ", "IWM", "DIA", "SOXL", "TQQQ"}


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert numeric/pandas/NaN values to finite float."""
    if val is None or isinstance(val, bool):
        return default
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        pass
    try:
        num = float(val)
        return num if math.isfinite(num) else default
    except (TypeError, ValueError):
        return default


def _coerce_ref_date(
    val: Optional[Union[date, datetime, pd.Timestamp, str]],
) -> Optional[date]:
    """Convert reference_date input into a NY calendar date when provided."""
    if val is None:
        return None
    if isinstance(val, pd.Timestamp):
        if pd.isna(val):
            return None
        return val.tz_convert(NY_TZ).date() if val.tzinfo is not None else val.date()
    if isinstance(val, datetime):
        return val.astimezone(NY_TZ).date() if val.tzinfo is not None else val.date()
    if isinstance(val, date):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.tz_convert(NY_TZ).date() if ts.tzinfo is not None else ts.date()
    return None


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


@dataclass
class RealTimeOptionQuote:
    ticker: str
    option_type: str          # "CALL" or "PUT"
    expiration: str           # YYYY-MM-DD
    dte: int
    strike: float
    spot_realtime: float
    iv: float
    realtime_theo_price: float
    est_bid: float
    est_ask: float
    max_limit_buy: float
    delta: float
    abs_delta: float
    gamma: float
    theta_daily: float
    tp1_option_price: float
    tp1_gain_pct: float
    tp2_option_price: float
    tp2_gain_pct: float
    sl_option_price: float
    sl_loss_pct: float
    recommended_contracts: int
    spread_pct: float
    fibo_entry_theo_price: Optional[float] = None


def black_scholes_price(
    spot: float,
    strike: float,
    dte_days: float,
    iv: float,
    option_type: str = "PUT",
    r: float = RISK_FREE_RATE,
) -> float:
    """Compute theoretical Black-Scholes option price per share using real-time spot price."""
    s_val = _safe_float(spot, -1.0)
    k_val = _safe_float(strike, -1.0)
    if s_val <= 0.0 or k_val <= 0.0:
        return 0.01

    dte_val = _safe_float(dte_days, 1.0)
    iv_val = _safe_float(iv, 0.35)
    t = max(dte_val, 0.5) / 365.0
    sigma = max(iv_val, 0.08)
    sqrt_t = max(0.001, math.sqrt(t))

    try:
        d1 = (math.log(s_val / k_val) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
    except (ZeroDivisionError, ValueError):
        return 0.01

    opt = str(option_type).upper()
    if opt == "CALL":
        val = s_val * norm_cdf(d1) - k_val * math.exp(-r * t) * norm_cdf(d2)
    else:
        val = k_val * math.exp(-r * t) * norm_cdf(-d2) - s_val * norm_cdf(-d1)
    return max(0.01, float(val))


def calculate_greeks(
    spot: float,
    strike: float,
    dte_days: float,
    iv: float,
    option_type: str = "PUT",
    r: float = RISK_FREE_RATE,
) -> Dict[str, float]:
    """Calculate Delta, Gamma, and daily Theta at the real-time spot price."""
    dte_val = _safe_float(dte_days, 1.0)
    iv_val = _safe_float(iv, 0.35)
    t = max(dte_val, 0.5) / 365.0
    sigma = max(iv_val, 0.08)
    sqrt_t = max(0.001, math.sqrt(t))
    safe_spot = max(0.01, _safe_float(spot, 1.0))
    safe_strike = max(0.01, _safe_float(strike, 1.0))

    d1 = (math.log(safe_spot / safe_strike) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    opt = str(option_type).upper()
    gamma = norm_pdf(d1) / (safe_spot * sigma * sqrt_t)
    if opt == "CALL":
        delta = norm_cdf(d1)
        theta_annual = (
            -(safe_spot * norm_pdf(d1) * sigma) / (2.0 * sqrt_t)
            - r * safe_strike * math.exp(-r * t) * norm_cdf(d2)
        )
    else:
        delta = norm_cdf(d1) - 1.0
        theta_annual = (
            -(safe_spot * norm_pdf(d1) * sigma) / (2.0 * sqrt_t)
            + r * safe_strike * math.exp(-r * t) * norm_cdf(-d2)
        )

    return {
        "delta": round(delta, 3),
        "abs_delta": round(abs(delta), 3),
        "gamma": round(gamma, 4),
        "theta_daily": round(theta_annual / 365.0, 3),
    }


def select_and_price_realtime_option(
    ticker: str,
    direction: Optional[str] = None,
    spot_realtime: float = 0.0,
    tp1_underlying: float = 0.0,
    tp2_underlying: float = 0.0,
    sl_underlying: float = 0.0,
    fibo_entry_underlying: Optional[float] = None,
    fallback_hv: float = 0.35,
    min_dte: int = 1,
    max_dte: int = 7,
    target_abs_delta: float = 0.58,
    max_spread_pct: float = 4.5,
    reference_date: Optional[Union[date, datetime, pd.Timestamp, str]] = None,
    default_contracts_under_80: int = 2,
    default_contracts_over_80: int = 1,
    default_contracts_stock: Optional[int] = None,
    default_contracts_etf: Optional[int] = None,
    is_etf: Optional[bool] = None,
    option_rules: Optional[Dict[str, Any]] = None,
    option_type: Optional[str] = None,
) -> RealTimeOptionQuote:
    """
    Selects the best liquid weekly Slight-ITM / ATM option contract (target |Delta| ~0.58)
    and computes synthetic Real-Time option prices at spot_realtime, TP1, TP2, and SL using the
    live underlying stock price (eliminating the 15-minute option chain delay).
    """
    rules = option_rules or {}
    eff_min_dte = int(rules.get("min_dte", min_dte)) if option_rules is not None else int(min_dte)
    eff_max_dte = max(
        eff_min_dte,
        int(rules.get("max_dte", max_dte)) if option_rules is not None else int(max_dte),
    )
    eff_target_delta = _safe_float(
        rules.get("target_abs_delta", target_abs_delta) if option_rules is not None else target_abs_delta,
        0.58,
    )
    eff_max_spread = max(
        0.8,
        _safe_float(
            rules.get("max_spread_pct", max_spread_pct) if option_rules is not None else max_spread_pct,
            4.5,
        ),
    )
    eff_c_under_80 = int(
        rules.get("default_contracts_under_80", default_contracts_under_80)
        if option_rules is not None
        else default_contracts_under_80
    )
    eff_c_over_80 = int(
        rules.get("default_contracts_over_80", default_contracts_over_80)
        if option_rules is not None
        else default_contracts_over_80
    )

    eff_direction = direction if direction is not None else (option_type or "CALL")
    opt_type = "PUT" if "PUT" in str(eff_direction).upper() else "CALL"
    tk_upper = str(ticker or "").upper().strip()
    is_etf_tk = bool(is_etf) if is_etf is not None else (tk_upper in ETF_TICKERS)
    ref_date = _coerce_ref_date(reference_date) or today_ny()

    spot_rt = max(0.01, _safe_float(spot_realtime, 1.0))
    tp1_u = max(0.01, _safe_float(tp1_underlying, spot_rt))
    tp2_u = max(0.01, _safe_float(tp2_underlying, spot_rt))
    sl_u = max(0.01, _safe_float(sl_underlying, spot_rt))

    # Synthetic fallback expiration & strike centered on spot_realtime targeting eff_target_delta
    selected_dte = max(eff_min_dte, min(3, eff_max_dte))
    selected_exp = (ref_date + timedelta(days=selected_dte)).strftime("%Y-%m-%d")
    selected_iv = max(0.20, min(_safe_float(fallback_hv, 0.35) * 1.05, 1.20))
    spread_pct = round(min(eff_max_spread, 1.4 if is_etf_tk else 2.2), 2)

    step = 1.0 if spot_rt < 100.0 else (2.5 if spot_rt < 250.0 else 5.0)
    base_k = round(spot_rt / step) * step
    cand_strikes = [
        round(base_k + m * step, 2)
        for m in (-2, -1, 0, 1, 2)
        if base_k + m * step > 0
    ]
    if not cand_strikes:
        cand_strikes = [round(spot_rt, 2)]

    def _fallback_key(k: float):
        g = calculate_greeks(spot_rt, k, selected_dte, selected_iv, opt_type)
        itm_pref = (
            0
            if ((opt_type == "CALL" and k <= spot_rt) or (opt_type == "PUT" and k >= spot_rt))
            else 1
        )
        return (abs(g["abs_delta"] - eff_target_delta), itm_pref, abs(k - spot_rt))

    selected_strike = min(cand_strikes, key=_fallback_key)

    try:
        yf_tk = yf.Ticker(tk_upper)
        expirations = list(yf_tk.options or [])
        in_window_exps = []
        longer_exps = []
        for exp_s in expirations:
            try:
                exp_str = str(exp_s).strip()[:10]
                exp_dt = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_dt - ref_date).days
                if eff_min_dte <= dte <= eff_max_dte:
                    in_window_exps.append((dte, exp_str))
                elif dte > eff_max_dte:
                    longer_exps.append((dte, exp_str))
            except Exception:
                continue

        valid_exps = in_window_exps if in_window_exps else longer_exps
        if valid_exps:
            valid_exps.sort(key=lambda x: x[0])
            selected_dte, selected_exp = valid_exps[0]
            chain = yf_tk.option_chain(selected_exp)
            df_side = chain.puts if opt_type == "PUT" else chain.calls
            if df_side is not None and not df_side.empty and "strike" in df_side.columns:
                df_side = df_side.copy()
                df_side["strike"] = pd.to_numeric(df_side["strike"], errors="coerce")
                df_side = df_side[df_side["strike"].notna() & (df_side["strike"] > 0)].copy()

                if not df_side.empty:
                    df_side["dist"] = (df_side["strike"] - spot_rt).abs()
                    nearest = df_side.sort_values("dist").head(16)
                    has_liq_cols = ("volume" in df_side.columns) or ("openInterest" in df_side.columns)

                    candidates: List[Dict[str, Any]] = []
                    for _, row in nearest.iterrows():
                        k = _safe_float(row.get("strike"), 0.0)
                        if k <= 0.0:
                            continue
                        bid = _safe_float(row.get("bid"), 0.0)
                        ask = _safe_float(row.get("ask"), 0.0)
                        last_p = _safe_float(row.get("lastPrice"), 0.0)
                        vol = _safe_float(row.get("volume"), 0.0)
                        oi = _safe_float(row.get("openInterest"), 0.0)
                        raw_iv = _safe_float(row.get("impliedVolatility"), 0.0)
                        cand_iv = raw_iv if 0.10 <= raw_iv <= 2.50 else selected_iv

                        cand_abs_delta = calculate_greeks(
                            spot_rt, k, selected_dte, cand_iv, opt_type
                        )["abs_delta"]
                        delta_diff = abs(cand_abs_delta - eff_target_delta)
                        itm_pref = (
                            0
                            if ((opt_type == "CALL" and k <= spot_rt) or (opt_type == "PUT" and k >= spot_rt))
                            else 1
                        )
                        is_quoted = (bid > 0.0) and (ask >= bid)
                        is_liquid = (vol > 0.0 or oi > 0.0) if has_liq_cols else True
                        row_spread = (
                            ((ask - bid) / (0.5 * (bid + ask)) * 100.0)
                            if (bid > 0.0 and ask > bid)
                            else None
                        )
                        spread_ok = (row_spread is not None) and (row_spread <= eff_max_spread)

                        candidates.append(
                            {
                                "strike": k,
                                "cand_iv": cand_iv,
                                "delta_diff": delta_diff,
                                "itm_pref": itm_pref,
                                "is_quoted": is_quoted,
                                "is_liquid": is_liquid,
                                "last_p": last_p,
                                "row_spread": row_spread,
                                "spread_ok": spread_ok,
                            }
                        )

                    if candidates:
                        tier1 = [
                            c for c in candidates if c["is_quoted"] and c["is_liquid"] and c["spread_ok"]
                        ]
                        tier2 = [c for c in candidates if c["is_quoted"] and c["is_liquid"]]
                        tier3 = [c for c in candidates if c["is_liquid"] or c["last_p"] > 0.0]
                        pool = tier1 or tier2 or tier3 or candidates

                        chosen = min(
                            pool,
                            key=lambda c: (c["delta_diff"], c["itm_pref"], abs(c["strike"] - spot_rt)),
                        )
                        selected_strike = chosen["strike"]
                        selected_iv = chosen["cand_iv"]
                        if chosen["row_spread"] is not None:
                            spread_pct = round(
                                min(eff_max_spread, max(0.8, chosen["row_spread"])), 2
                            )
    except Exception:
        pass

    # Real-time Black-Scholes price & Greeks at spot_realtime (NOT fibo_entry_underlying)
    rt_price = black_scholes_price(spot_rt, selected_strike, selected_dte, selected_iv, opt_type)
    greeks = calculate_greeks(spot_rt, selected_strike, selected_dte, selected_iv, opt_type)

    half_spread = max(0.01, rt_price * (spread_pct / 100.0) * 0.5)
    est_bid = round(max(0.01, rt_price - half_spread), 2)
    est_ask = round(rt_price + half_spread, 2)
    max_limit_buy = round(est_ask + 0.02, 2)

    fibo_entry_theo: Optional[float] = None
    fibo_u = _safe_float(fibo_entry_underlying, 0.0)
    if fibo_u > 0.0:
        fibo_entry_theo = round(
            black_scholes_price(fibo_u, selected_strike, selected_dte, selected_iv, opt_type),
            2,
        )

    # Project Option Prices at TP1, TP2, and SL relative to realtime_theo_price
    tp1_opt = black_scholes_price(tp1_u, selected_strike, selected_dte, selected_iv, opt_type)
    tp2_opt = black_scholes_price(tp2_u, selected_strike, selected_dte, selected_iv, opt_type)
    sl_opt = black_scholes_price(sl_u, selected_strike, selected_dte, selected_iv, opt_type)

    entry_ref = max(0.05, rt_price)
    tp1_gain_pct = round((tp1_opt - entry_ref) / entry_ref * 100.0, 1)
    tp2_gain_pct = round((tp2_opt - entry_ref) / entry_ref * 100.0, 1)
    sl_loss_pct = round((sl_opt - entry_ref) / entry_ref * 100.0, 1)

    if is_etf_tk and default_contracts_etf is not None:
        contracts = int(default_contracts_etf)
    elif (not is_etf_tk) and default_contracts_stock is not None:
        contracts = int(default_contracts_stock)
    else:
        contracts = eff_c_under_80 if spot_rt < 80.0 else eff_c_over_80

    return RealTimeOptionQuote(
        ticker=tk_upper or ticker,
        option_type=opt_type,
        expiration=selected_exp,
        dte=selected_dte,
        strike=round(selected_strike, 2),
        spot_realtime=round(spot_rt, 2),
        iv=round(selected_iv, 4),
        realtime_theo_price=round(rt_price, 2),
        est_bid=est_bid,
        est_ask=est_ask,
        max_limit_buy=max_limit_buy,
        delta=greeks["delta"],
        abs_delta=greeks["abs_delta"],
        gamma=greeks["gamma"],
        theta_daily=greeks["theta_daily"],
        tp1_option_price=round(tp1_opt, 2),
        tp1_gain_pct=tp1_gain_pct,
        tp2_option_price=round(tp2_opt, 2),
        tp2_gain_pct=tp2_gain_pct,
        sl_option_price=round(sl_opt, 2),
        sl_loss_pct=sl_loss_pct,
        recommended_contracts=contracts,
        spread_pct=round(spread_pct, 2),
        fibo_entry_theo_price=fibo_entry_theo,
    )
