"""
Real-Time Black-Scholes-Merton Option Pricing & Greeks Engine for Day Option Bot.
Eliminates the 15-minute OPRA free-feed delay by combining the option chain's Implied Volatility (IV)
structure with the 0-second Real-Time Underlying Stock Price (Spot) and projecting exact Option Prices
at Entry, TP1 (Next Fibo Line), TP2 (Runner Fibo Line), and SL (Opposite Fibo Line).
"""

from dataclasses import dataclass
from datetime import datetime, date
import math
from typing import Dict, Optional
import numpy as np
import yfinance as yf


RISK_FREE_RATE = 0.043  # ~4.3% US Treasury yield baseline


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


def black_scholes_price(
    spot: float,
    strike: float,
    dte_days: float,
    iv: float,
    option_type: str = "PUT",
    r: float = RISK_FREE_RATE,
) -> float:
    """Compute theoretical Black-Scholes option price per share using real-time spot price."""
    if spot <= 0 or strike <= 0:
        return 0.01
    t = max(float(dte_days), 0.5) / 365.0
    sigma = max(float(iv), 0.08)
    sqrt_t = max(0.001, math.sqrt(t))

    try:
        d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
    except (ZeroDivisionError, ValueError):
        return 0.01

    opt = option_type.upper()
    if opt == "CALL":
        val = spot * norm_cdf(d1) - strike * math.exp(-r * t) * norm_cdf(d2)
    else:
        val = strike * math.exp(-r * t) * norm_cdf(-d2) - spot * norm_cdf(-d1)
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
    t = max(float(dte_days), 0.5) / 365.0
    sigma = max(float(iv), 0.08)
    sqrt_t = max(0.001, math.sqrt(t))
    safe_spot = max(0.01, float(spot))
    safe_strike = max(0.01, float(strike))

    d1 = (math.log(safe_spot / safe_strike) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    opt = option_type.upper()
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
    direction: str,
    spot_realtime: float,
    tp1_underlying: float,
    tp2_underlying: float,
    sl_underlying: float,
    fibo_entry_underlying: Optional[float] = None,
    fallback_hv: float = 0.35,
    min_dte: int = 1,
    max_dte: int = 7,
) -> RealTimeOptionQuote:
    """
    Selects the best liquid weekly Slight-ITM / ATM option contract (target |Delta| ~0.55 - 0.65)
    and computes synthetic Real-Time option prices at Entry, TP1, TP2, and SL using the
    live underlying stock price (eliminating the 15-minute option chain delay).
    """
    opt_type = "PUT" if "PUT" in direction.upper() else "CALL"
    today = date.today()
    entry_spot_ref = float(fibo_entry_underlying) if fibo_entry_underlying else float(spot_realtime)

    # Default synthetic parameters if chain fetch fails or market is closed
    selected_exp = today.strftime("%Y-%m-%d")
    selected_dte = 3
    step = 1.0 if entry_spot_ref < 100 else (2.5 if entry_spot_ref < 250 else 5.0)
    # Slight ITM strike at the Fibo entry zone for higher Delta (~0.58) and lower Theta decay
    if opt_type == "CALL":
        selected_strike = round(math.floor(entry_spot_ref / step) * step, 2)
    else:
        selected_strike = round(math.ceil(entry_spot_ref / step) * step, 2)
    selected_iv = max(0.20, min(fallback_hv * 1.05, 1.20))
    spread_pct = 1.4 if ticker in ("SPY", "QQQ") else 2.2

    try:
        yf_tk = yf.Ticker(ticker)
        expirations = yf_tk.options or ()
        valid_exps = []
        for exp_s in expirations:
            try:
                exp_dt = datetime.strptime(exp_s, "%Y-%m-%d").date()
                dte = (exp_dt - today).days
                if min_dte <= dte <= max_dte:
                    valid_exps.append((dte, exp_s))
                elif dte > max_dte and not valid_exps:
                    valid_exps.append((dte, exp_s))
            except Exception:
                continue

        if valid_exps:
            valid_exps.sort(key=lambda x: x[0])
            selected_dte, selected_exp = valid_exps[0]
            chain = yf_tk.option_chain(selected_exp)
            df_side = chain.puts if opt_type == "PUT" else chain.calls
            if df_side is not None and not df_side.empty:
                df_side = df_side.copy()
                df_side["dist"] = (df_side["strike"] - entry_spot_ref).abs()
                nearest = df_side.sort_values("dist").head(5)
                # Prefer slight ITM strike at the Fibo entry level for Delta ~0.58
                if opt_type == "CALL":
                    itm = nearest[nearest["strike"] <= entry_spot_ref]
                else:
                    itm = nearest[nearest["strike"] >= entry_spot_ref]
                chosen_row = itm.iloc[0] if not itm.empty else nearest.iloc[0]
                selected_strike = float(chosen_row["strike"])
                raw_iv = float(chosen_row.get("impliedVolatility", 0.0) or 0.0)
                if 0.10 <= raw_iv <= 2.50:
                    selected_iv = raw_iv
                bid = float(chosen_row.get("bid", 0.0) or 0.0)
                ask = float(chosen_row.get("ask", 0.0) or 0.0)
                if bid > 0 and ask > bid:
                    mid = 0.5 * (bid + ask)
                    spread_pct = round(min(6.0, max(0.8, (ask - bid) / mid * 100.0)), 2)
    except Exception:
        pass

    # Real-time Black-Scholes price at the Fibo Entry Trigger level
    rt_price = black_scholes_price(entry_spot_ref, selected_strike, selected_dte, selected_iv, opt_type)
    greeks = calculate_greeks(entry_spot_ref, selected_strike, selected_dte, selected_iv, opt_type)

    half_spread = max(0.01, rt_price * (spread_pct / 100.0) * 0.5)
    est_bid = round(max(0.01, rt_price - half_spread), 2)
    est_ask = round(rt_price + half_spread, 2)
    # Pegged Marketable Limit Buy recommendation (Mid + $0.02 or Ask + $0.01)
    max_limit_buy = round(est_ask + 0.02, 2)

    # Project Option Prices at TP1, TP2, and SL using Real-Time Black-Scholes
    tp1_opt = black_scholes_price(tp1_underlying, selected_strike, selected_dte, selected_iv, opt_type)
    tp2_opt = black_scholes_price(tp2_underlying, selected_strike, selected_dte, selected_iv, opt_type)
    sl_opt = black_scholes_price(sl_underlying, selected_strike, selected_dte, selected_iv, opt_type)

    entry_ref = max(0.05, rt_price)
    tp1_gain_pct = round((tp1_opt - entry_ref) / entry_ref * 100.0, 1)
    tp2_gain_pct = round((tp2_opt - entry_ref) / entry_ref * 100.0, 1)
    sl_loss_pct = round((sl_opt - entry_ref) / entry_ref * 100.0, 1)

    contracts = 2 if spot_realtime < 80.0 else 1

    return RealTimeOptionQuote(
        ticker=ticker,
        option_type=opt_type,
        expiration=selected_exp,
        dte=selected_dte,
        strike=round(selected_strike, 2),
        spot_realtime=round(spot_realtime, 2),
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
    )
