"""
Core Fibonacci + EMA 200 + Volume RVOL + Reversal Candle Scanner for Day Option Bot.
Implements:
1. TF 2m EMA 200 Trend Filter:
   - Price < EMA 200 -> DOWNTREND (BUY PUT ONLY)
   - Price > EMA 200 -> UPTREND (BUY CALL ONLY)
   - Price == EMA 200 / NaN -> NEUTRAL (No trade)
2. Previous-Day Asymmetric Fibonacci Auto-Drawing (TF 2m NY Regular Hours 09:30-15:59):
   - Downtrend: Point 1 (Low) = Wick Low of lowest bar; Point 2 (High) = Body High of highest bar
   - Uptrend:   Point 1 (High) = Wick High of highest bar; Point 2 (Low) = Body Low of lowest bar
3. Smart Fibo Level Routing (0.500 for SPY/QQQ; 0.618 Golden Zone primary for Individual Stocks;
   secondary routing only with RVOL >= secondary_min_rvol)
4. Reversal Candle (gated on touched == True) + Volume Spike + Morning Gap & Earnings Blackout Filters
5. TP1 (Next Fibo Line + Lock Profit anchored to entry_underlying), TP2 (2nd Fibo Line Runner), SL (Opposite Fibo Line)
"""

import time
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import yfinance as yf

from .earnings_filter import check_earnings_blackout
from .options_pricing import select_and_price_realtime_option
from .time_utils import NY_TZ, ensure_ny_regular_hours, format_ny_timestamp, today_ny


def fetch_intraday_2m(
    ticker: str,
    period: str = "10d",
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch 2-minute regular-hours intraday OHLCV data for EMA 200 and Previous-Day Fibonacci calculation.
    Does NOT silently substitute 5m data if 2m data is empty.
    """
    attempts = max(1, int(max_retries))
    for attempt in range(attempts):
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=period, interval="2m", prepost=False)
            if df is None or df.empty:
                return pd.DataFrame()
            return ensure_ny_regular_hours(df)
        except Exception:
            if attempt < attempts - 1:
                time.sleep(0.5 * (2 ** attempt))
    return pd.DataFrame()


def compute_intraday_indicators(
    df: pd.DataFrame,
    ema_period: int = 200,
    vol_ma_period: int = 20,
) -> pd.DataFrame:
    """Compute EMA 200 (with min_periods warmup), Volume MA, NaN-safe RVOL, BodyHigh/BodyLow, and Session VWAP."""
    df = df.copy()
    for col in ("Open", "High", "Low", "Close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    vol_clean = (
        pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0).clip(lower=0.0)
        if "Volume" in df.columns
        else pd.Series(0.0, index=df.index)
    )
    df["Volume"] = vol_clean

    df["EMA200"] = (
        df["Close"].ewm(span=ema_period, min_periods=ema_period, adjust=False).mean()
    )
    df["VolMA"] = vol_clean.rolling(
        vol_ma_period, min_periods=min(5, vol_ma_period)
    ).mean()
    raw_rvol = np.where(df["VolMA"] > 0, vol_clean / df["VolMA"], 1.0)
    df["RVOL"] = np.where(np.isfinite(raw_rvol), raw_rvol, 1.0)
    df["BodyHigh"] = np.maximum(df["Open"], df["Close"])
    df["BodyLow"] = np.minimum(df["Open"], df["Close"])

    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is not None:
            df["Date"] = df.index.tz_convert(NY_TZ).date
        else:
            df["Date"] = df.index.date

    typ_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    df["TPV"] = typ_price * vol_clean
    if "Date" in df.columns:
        df["CumTPV"] = df.groupby("Date")["TPV"].cumsum()
        df["CumVol"] = df.groupby("Date")["Volume"].cumsum()
    else:
        df["CumTPV"] = df["TPV"].cumsum()
        df["CumVol"] = vol_clean.cumsum()
    df["VWAP"] = np.where(df["CumVol"] > 0, df["CumTPV"] / df["CumVol"], df["Close"])
    return df


def extract_prev_day_fibo_anchors(prev_day_df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract previous trading day's asymmetric Fibonacci anchors per user specification:
    - Downtrend Fibo: Point 1 = Wick Low of lowest bar, Point 2 = Body High of highest bar
    - Uptrend Fibo:   Point 1 = Wick High of highest bar, Point 2 = Body Low of lowest bar
    """
    df_in = prev_day_df
    if "BodyLow" not in df_in.columns or "BodyHigh" not in df_in.columns:
        df_in = df_in.copy()
        df_in["BodyHigh"] = np.maximum(df_in["Open"], df_in["Close"])
        df_in["BodyLow"] = np.minimum(df_in["Open"], df_in["Close"])

    idx_min = df_in["Low"].idxmin()
    idx_max = df_in["High"].idxmax()

    wick_low = float(df_in.loc[idx_min, "Low"])
    body_low = float(df_in.loc[idx_min, "BodyLow"])
    wick_high = float(df_in.loc[idx_max, "High"])
    body_high = float(df_in.loc[idx_max, "BodyHigh"])

    return {
        "wick_low": round(wick_low, 2),
        "body_low": round(body_low, 2),
        "wick_high": round(wick_high, 2),
        "body_high": round(body_high, 2),
    }


def build_fibo_levels(anchors: Dict[str, float], trend_direction: str) -> Dict[str, float]:
    """
    Calculate Fibonacci levels (0.000, 0.236, 0.382, 0.500, 0.618, 0.786, 1.000)
    according to trend direction.
    """
    if trend_direction == "DOWNTREND":
        # Retracement upward from yesterday's Wick Low (0.0) toward yesterday's Body High (1.0)
        p_low = anchors["wick_low"]
        p_high = anchors["body_high"]
        rng = max(0.01, p_high - p_low)
        return {
            "anchor_point_1_label": f"Wick Low (${p_low:.2f})",
            "anchor_point_2_label": f"Body High (${p_high:.2f})",
            "range_usd": round(rng, 2),
            "0.000": round(p_low, 2),
            "0.236": round(p_low + 0.236 * rng, 2),
            "0.382": round(p_low + 0.382 * rng, 2),
            "0.500": round(p_low + 0.500 * rng, 2),
            "0.618": round(p_low + 0.618 * rng, 2),
            "0.786": round(p_low + 0.786 * rng, 2),
            "1.000": round(p_high, 2),
        }
    else:
        # Uptrend: Retracement downward from yesterday's Wick High (0.0) toward yesterday's Body Low (1.0)
        p_high = anchors["wick_high"]
        p_low = anchors["body_low"]
        rng = max(0.01, p_high - p_low)
        return {
            "anchor_point_1_label": f"Wick High (${p_high:.2f})",
            "anchor_point_2_label": f"Body Low (${p_low:.2f})",
            "range_usd": round(rng, 2),
            "0.000": round(p_high, 2),
            "0.236": round(p_high - 0.236 * rng, 2),
            "0.382": round(p_high - 0.382 * rng, 2),
            "0.500": round(p_high - 0.500 * rng, 2),
            "0.618": round(p_high - 0.618 * rng, 2),
            "0.786": round(p_high - 0.786 * rng, 2),
            "1.000": round(p_low, 2),
        }


def detect_reversal_candle(
    curr_df: pd.DataFrame,
    trend_direction: str,
    level_price: float,
    rng: float,
) -> Dict:
    """
    Inspect the latest 2 bars to detect Fibonacci level touch + reversal candlestick confirmation.
    Requires `touched == True` before any reversal pattern name is emitted.
    """
    if curr_df is None or curr_df.empty:
        return {
            "touched": False,
            "confirmed": False,
            "pattern_name": "No Intraday Bars",
        }

    last_bar = curr_df.iloc[-1]
    prev_bar = curr_df.iloc[-2] if len(curr_df) >= 2 else last_bar

    c_vals = [last_bar.get("Open"), last_bar.get("High"), last_bar.get("Low"), last_bar.get("Close")]
    if any(pd.isna(v) or not np.isfinite(float(v)) for v in c_vals):
        return {
            "touched": False,
            "confirmed": False,
            "pattern_name": "Invalid Candle Data",
        }

    c_open = float(last_bar["Open"])
    c_high = float(last_bar["High"])
    c_low = float(last_bar["Low"])
    c_close = float(last_bar["Close"])

    p_vals = [prev_bar.get("Open"), prev_bar.get("High"), prev_bar.get("Low"), prev_bar.get("Close")]
    if any(pd.isna(v) or not np.isfinite(float(v)) for v in p_vals):
        p_open, p_high, p_low, p_close = c_open, c_high, c_low, c_close
    else:
        p_open = float(prev_bar["Open"])
        p_high = float(prev_bar["High"])
        p_low = float(prev_bar["Low"])
        p_close = float(prev_bar["Close"])

    tol = max(rng * 0.02, c_close * 0.0012)
    bar_range = max(0.001, c_high - c_low)

    if trend_direction == "DOWNTREND":
        # Resistance bounce at Fibo level -> looking for Bearish Rejection / Shooting Star / Bearish Engulfing
        touched = (
            (c_high >= level_price - tol and c_low <= level_price + tol * 1.5)
            or (p_high >= level_price - tol and p_low <= level_price + tol * 1.5)
        )
        upper_wick = c_high - max(c_open, c_close)
        is_bearish_body = c_close < c_open
        is_shooting_star = (upper_wick / bar_range) >= 0.45 and c_close <= level_price + tol * 0.5
        is_bearish_engulf = is_bearish_body and (c_close <= p_low or c_close <= level_price)

        confirmed = touched and (is_bearish_engulf or (is_shooting_star and is_bearish_body))
        if not touched:
            pattern_name = "Waiting for Pullback to Fibo (รอราคารีบาวด์แตะเส้น)"
        elif is_shooting_star and is_bearish_body:
            pattern_name = "Bearish Shooting Star Rejection (ไส้บนปฎิเสธราคา + ปิดแดง)"
        elif is_bearish_engulf:
            pattern_name = "Bearish Reversal Candle (แท่งแดงกลับตัวใต้แนว Fibo)"
        else:
            pattern_name = "Testing Fibo Resistance (กำลังทดสอบแนวต้าน รอแท่งแดงยืนยัน)"
    else:
        # Uptrend: Support bounce at Fibo level -> looking for Bullish Rejection / Hammer / Bullish Engulfing
        touched = (
            (c_low <= level_price + tol and c_high >= level_price - tol * 1.5)
            or (p_low <= level_price + tol and p_high >= level_price - tol * 1.5)
        )
        lower_wick = min(c_open, c_close) - c_low
        is_bullish_body = c_close > c_open
        is_hammer = (lower_wick / bar_range) >= 0.45 and c_close >= level_price - tol * 0.5
        is_bullish_engulf = is_bullish_body and (c_close >= p_high or c_close >= level_price)

        confirmed = touched and (is_bullish_engulf or (is_hammer and is_bullish_body))
        if not touched:
            pattern_name = "Waiting for Pullback to Fibo (รอราคาย่อตัวแตะเส้น)"
        elif is_hammer and is_bullish_body:
            pattern_name = "Bullish Hammer Rejection (ไส้ล่างดึงกลับ + ปิดเขียว)"
        elif is_bullish_engulf:
            pattern_name = "Bullish Reversal Candle (แท่งเขียวกลับตัวยืนเหนือแนว Fibo)"
        else:
            pattern_name = "Testing Fibo Support (กำลังทดสอบแนวรับ รอแท่งเขียวยืนยัน)"

    return {
        "touched": bool(touched),
        "confirmed": bool(confirmed),
        "pattern_name": pattern_name,
    }


def analyze_single_ticker(ticker: str, config: Dict) -> Dict:
    """Run full Day Option analysis on a single ticker."""
    rules = config.get("strategy_rules", {})
    ema_period = int(rules.get("ema_period", 200))

    vol_cfg = rules.get("volume_filter", {})
    vol_filter_enabled = bool(vol_cfg.get("enabled", True))
    vol_ma_period = int(vol_cfg.get("vol_ma_period", 20))
    min_rvol = float(vol_cfg.get("min_rvol", 1.15))

    gap_cfg = rules.get("gap_filter", {})
    gap_filter_enabled = bool(gap_cfg.get("enabled", True))
    max_gap_pct = float(gap_cfg.get("max_morning_gap_pct", 2.0))

    earn_cfg = rules.get("earnings_filter", {})
    earn_enabled = bool(earn_cfg.get("enabled", True))
    earn_days_before = int(earn_cfg.get("blackout_days_before", earn_cfg.get("days_before", 2)))
    earn_days_after = int(earn_cfg.get("blackout_days_after", earn_cfg.get("days_after", 1)))

    routing_cfg = rules.get("smart_fibo_routing", {})
    etf_primary_num = float(routing_cfg.get("etf_primary_level", 0.5))
    stock_primary_num = float(routing_cfg.get("stock_primary_level", 0.618))
    allow_secondary = bool(routing_cfg.get("allow_secondary_with_high_rvol", True))
    secondary_min_rvol = float(routing_cfg.get("secondary_min_rvol", 1.5))

    tp_sl_cfg = rules.get("tp_sl_rules", {})
    lock_profit_ratio = float(tp_sl_cfg.get("lock_profit_ratio_at_tp1", 0.45))

    opt_rules = rules.get("option_contract_rules", {})
    min_dte = int(opt_rules.get("min_dte", 1))
    max_dte = int(opt_rules.get("max_dte", 7))
    target_abs_delta = float(opt_rules.get("target_abs_delta", 0.58))
    max_spread_pct = float(opt_rules.get("max_spread_pct", 4.5))
    default_c_under_80 = int(opt_rules.get("default_contracts_under_80", 2))
    default_c_over_80 = int(opt_rules.get("default_contracts_over_80", 1))

    min_prev_bars = int(rules.get("min_prev_day_bars", 10))
    min_fibo_range_pct = float(rules.get("min_fibo_range_pct", 0.2))

    etf_list = {
        str(x).upper().strip()
        for x in config.get("watchlist", {}).get("etf_index", ["SPY", "QQQ"])
    }
    is_etf = str(ticker).upper().strip() in etf_list

    # 1. Check Earnings Blackout (NY date reference)
    ny_today = today_ny()
    earnings_info = check_earnings_blackout(
        ticker=ticker,
        days_before=earn_days_before,
        days_after=earn_days_after,
        reference_date=ny_today,
        enabled=earn_enabled,
        is_etf=is_etf,
    )

    # 2. Fetch 2m Intraday Data & normalize to NY regular hours
    df_raw = fetch_intraday_2m(ticker)
    df_clean = ensure_ny_regular_hours(df_raw)
    if df_clean.empty or len(df_clean) < ema_period:
        return {
            "ticker": ticker,
            "status": "NO_DATA",
            "earnings": earnings_info,
            "reason": (
                f"ข้อมูลกราฟ 2m ไม่เพียงพอสำหรับ EMA {ema_period} "
                f"({len(df_clean)}/{ema_period} แท่ง)"
            ),
        }

    df = compute_intraday_indicators(
        df_clean, ema_period=ema_period, vol_ma_period=vol_ma_period
    )
    dates = sorted(df["Date"].unique())
    if len(dates) < 2:
        return {
            "ticker": ticker,
            "status": "NO_DATA",
            "earnings": earnings_info,
            "reason": "ข้อมูลวันทำการก่อนหน้าไม่เพียงพอสำหรับตี Fibonacci",
        }

    prev_date = dates[-2]
    curr_date = dates[-1]
    prev_df = df[df["Date"] == prev_date]
    curr_df = df[df["Date"] == curr_date]

    if len(prev_df) < min_prev_bars or curr_df.empty:
        return {
            "ticker": ticker,
            "status": "NO_DATA",
            "earnings": earnings_info,
            "reason": (
                f"ข้อมูลวันทำการก่อนหน้าไม่สมบูรณ์ ({len(prev_df)}/{min_prev_bars} แท่ง)"
            ),
        }

    is_current_session = bool(curr_date == ny_today)
    last_bar_et = (
        format_ny_timestamp(curr_df.index[-1])
        if isinstance(curr_df.index, pd.DatetimeIndex)
        else ""
    )

    last_row = curr_df.iloc[-1]
    prev_row = curr_df.iloc[-2] if len(curr_df) >= 2 else last_row

    if (
        pd.isna(last_row["Close"])
        or pd.isna(last_row["EMA200"])
        or not np.isfinite(float(last_row["Close"]))
        or not np.isfinite(float(last_row["EMA200"]))
    ):
        return {
            "ticker": ticker,
            "status": "NO_DATA",
            "earnings": earnings_info,
            "reason": "ค่า Spot หรือ EMA200 ไม่สมบูรณ์ (NaN)",
        }

    spot = round(float(last_row["Close"]), 2)
    ema200 = round(float(last_row["EMA200"]), 2)
    vwap = (
        round(float(last_row["VWAP"]), 2)
        if pd.notna(last_row["VWAP"]) and np.isfinite(float(last_row["VWAP"]))
        else spot
    )
    rvol_vals = [
        float(v)
        for v in (last_row.get("RVOL"), prev_row.get("RVOL"))
        if pd.notna(v) and np.isfinite(float(v))
    ]
    curr_rvol = round(max(rvol_vals), 2) if rvol_vals else 1.0

    # 3. Determine EMA 200 Trend Filter
    if spot < ema200:
        trend_direction = "DOWNTREND"
        option_action = "BUY PUT"
        trend_thai = "ขาลง (ราคาต่ำกว่า EMA 200 -> ดักซื้อ PUT เท่านั้น)"
    elif spot > ema200:
        trend_direction = "UPTREND"
        option_action = "BUY CALL"
        trend_thai = "ขาขึ้น (ราคาสูงกว่า EMA 200 -> ดักซื้อ CALL เท่านั้น)"
    else:
        return {
            "ticker": ticker,
            "is_etf": is_etf,
            "status": "NEUTRAL",
            "status_thai": "⚖️ ราคาเท่ากับ EMA 200 (NEUTRAL — ไม่เข้าเทรด)",
            "score": 0,
            "prev_trading_date": str(prev_date),
            "current_trading_date": str(curr_date),
            "is_current_session": is_current_session,
            "last_bar_et": last_bar_et,
            "spot_price": spot,
            "ema200": ema200,
            "vwap": vwap,
            "rvol": curr_rvol,
            "rvol_passed": False,
            "trend_direction": "NEUTRAL",
            "option_action": "NEUTRAL",
            "trend_thai": "ราคาทับเส้น EMA 200 (NEUTRAL)",
            "earnings": earnings_info,
            "reason": "Spot price equals EMA200 (NEUTRAL)",
        }

    # 4. Compute Previous-Day Asymmetric Fibonacci Levels + Min-Range Guard
    anchors = extract_prev_day_fibo_anchors(prev_df)
    raw_rng = (
        (anchors["body_high"] - anchors["wick_low"])
        if trend_direction == "DOWNTREND"
        else (anchors["wick_high"] - anchors["body_low"])
    )
    if raw_rng <= max(0.05, spot * (min_fibo_range_pct / 100.0)):
        return {
            "ticker": ticker,
            "status": "NO_DATA",
            "earnings": earnings_info,
            "reason": f"กรอบราคาวันก่อนหน้าแคบเกินไป ({raw_rng:.2f} USD) สำหรับตี Fibonacci",
        }

    fibo = build_fibo_levels(anchors, trend_direction)
    rng = fibo["range_usd"]

    # Check Morning Gap
    first_open = float(curr_df.iloc[0]["Open"])
    prev_close = float(prev_df.iloc[-1]["Close"])
    morning_gap_pct = round(
        abs(first_open - prev_close) / max(0.01, prev_close) * 100.0, 2
    )

    # Smart Primary vs Secondary Fibo Level Routing
    primary_lvl_key = f"{etf_primary_num:.3f}" if is_etf else f"{stock_primary_num:.3f}"
    if primary_lvl_key not in fibo:
        primary_lvl_key = "0.500" if is_etf else "0.618"
    secondary_lvl_key = "0.618" if primary_lvl_key == "0.500" else "0.500"

    dist_primary_pct = abs(spot - fibo[primary_lvl_key]) / spot * 100.0
    dist_secondary_pct = abs(spot - fibo[secondary_lvl_key]) / spot * 100.0

    active_lvl_key = primary_lvl_key
    if allow_secondary and curr_rvol >= secondary_min_rvol:
        sec_candle = detect_reversal_candle(
            curr_df, trend_direction, fibo[secondary_lvl_key], rng
        )
        if dist_secondary_pct < dist_primary_pct and (
            dist_secondary_pct <= 0.65 or sec_candle["touched"]
        ):
            active_lvl_key = secondary_lvl_key

    active_lvl_price = fibo[active_lvl_key]
    dist_active_pct = round(abs(spot - active_lvl_price) / spot * 100.0, 2)

    # Map TP1 (immediate next Fibo line), TP2 (2nd Fibo line runner), and SL (immediate opposite Fibo line)
    if active_lvl_key == "0.618":
        tp1_price = fibo["0.500"]
        tp2_price = fibo["0.382"]
        sl_price = fibo["0.786"]
        tp1_label = "Fibo 0.500 (ล็อกกำไร/ขยับ SL บังทุน)"
        tp2_label = "Fibo 0.382 (เป้าถือรันกำไรคำใหญ่)"
        sl_label = "Fibo 0.786 (จุดคัทลอสเมื่อหลุดโซน 0.618)"
    else:
        tp1_price = fibo["0.382"]
        tp2_price = fibo["0.236"]
        sl_price = fibo["0.618"]
        tp1_label = "Fibo 0.382 (ล็อกกำไร/ขยับ SL บังทุน)"
        tp2_label = "Fibo 0.236 (เป้าถือรันกำไรคำใหญ่)"
        sl_label = "Fibo 0.618 (จุดคัทลอสเมื่อหลุดโซน 0.500)"

    # Lock-profit trailing stop anchored to entry_underlying (spot) so TP1_TRAIL_WIN >= breakeven
    if trend_direction == "DOWNTREND":
        lock_sl_at_tp1 = (
            round(spot - (spot - tp1_price) * lock_profit_ratio, 2)
            if spot > tp1_price
            else round(spot, 2)
        )
    else:
        lock_sl_at_tp1 = (
            round(spot + (tp1_price - spot) * lock_profit_ratio, 2)
            if spot < tp1_price
            else round(spot, 2)
        )

    # 5. Detect Reversal Candlestick & Volume RVOL
    candle_check = detect_reversal_candle(
        curr_df, trend_direction, active_lvl_price, rng
    )
    rvol_passed = (curr_rvol >= min_rvol) if vol_filter_enabled else True

    # 6. Price Real-Time Option Contract via Black-Scholes (Zero 15m Delay)
    opt_quote = select_and_price_realtime_option(
        ticker=ticker,
        direction=option_action,
        spot_realtime=spot,
        tp1_underlying=tp1_price,
        tp2_underlying=tp2_price,
        sl_underlying=sl_price,
        fibo_entry_underlying=active_lvl_price,
        min_dte=min_dte,
        max_dte=max_dte,
        target_abs_delta=target_abs_delta,
        max_spread_pct=max_spread_pct,
        reference_date=ny_today,
        default_contracts_under_80=default_c_under_80,
        default_contracts_over_80=default_c_over_80,
        is_etf=is_etf,
    )
    if opt_quote is None:
        opt_quote_dict: Dict = {}
    elif isinstance(opt_quote, dict):
        opt_quote_dict = opt_quote
    else:
        opt_quote_dict = opt_quote.__dict__

    # 7. Determine Final Signal Status & Score
    score = 50
    if active_lvl_key == primary_lvl_key:
        score += 15
    if candle_check["touched"]:
        score += 15
    if candle_check["confirmed"]:
        score += 15
    if rvol_passed:
        score += 10
    if earnings_info["is_blackout"]:
        score = 0

    if earnings_info["is_blackout"]:
        status = "EARNINGS_BLACKOUT"
        status_thai = earnings_info["status_label"]
    elif gap_filter_enabled and morning_gap_pct > max_gap_pct and dist_active_pct > 1.5:
        status = "GAP_SKIP"
        status_thai = f"⚠️ เปิดกระโดด Gap {morning_gap_pct:.2f}% ห่างกรอบ Fibo เมื่อวาน (แนะนำข้าม)"
    elif candle_check["confirmed"] and rvol_passed and opt_quote_dict:
        status = "ACTIONABLE"
        status_thai = (
            f"🔥 สัญญาณเข้าเทรด {option_action} ที่ Fibo {active_lvl_key} "
            f"(แท่งกลับตัว + RVOL {curr_rvol:.2f}x)"
        )
    elif candle_check["touched"] or dist_active_pct <= 0.45:
        status = "NEAR_SETUP"
        status_thai = (
            f"👀 ราคาเข้าใกล้/กำลังทดสอบ Fibo {active_lvl_key} "
            f"(ห่าง {dist_active_pct:.2f}%) รอแท่งกลับตัว+Volume"
        )
    else:
        status = "WATCHING"
        status_thai = (
            f"⏳ รอราคาย่อ/เด้งมาหา Fibo {active_lvl_key} "
            f"(${active_lvl_price:.2f} | ห่าง {dist_active_pct:.2f}%)"
        )

    return {
        "ticker": ticker,
        "is_etf": is_etf,
        "status": status,
        "status_thai": status_thai,
        "score": min(100, score),
        "prev_trading_date": str(prev_date),
        "current_trading_date": str(curr_date),
        "is_current_session": is_current_session,
        "last_bar_et": last_bar_et,
        "spot_price": spot,
        "ema200": ema200,
        "vwap": vwap,
        "rvol": curr_rvol,
        "rvol_passed": rvol_passed,
        "morning_gap_pct": morning_gap_pct,
        "trend_direction": trend_direction,
        "option_action": option_action,
        "trend_thai": trend_thai,
        "primary_fibo_level": primary_lvl_key,
        "active_fibo_level": active_lvl_key,
        "active_fibo_price": active_lvl_price,
        "distance_to_active_fibo_pct": dist_active_pct,
        "fibo_levels": fibo,
        "candle_reversal": candle_check,
        "trade_plan": {
            "entry_underlying": spot,
            "fibo_entry_level": active_lvl_key,
            "fibo_entry_price": active_lvl_price,
            "tp1_underlying": tp1_price,
            "tp1_label": tp1_label,
            "lock_sl_after_tp1": lock_sl_at_tp1,
            "tp2_underlying": tp2_price,
            "tp2_label": tp2_label,
            "sl_underlying": sl_price,
            "sl_label": sl_label,
        },
        "option_quote": opt_quote_dict,
        "earnings": earnings_info,
    }


def run_full_day_option_scan(config: Dict) -> Dict:
    """Scan all tickers in the watchlist and return structured results."""
    wl = config.get("watchlist", {})
    tickers: List[str] = list(wl.get("etf_index", [])) + list(wl.get("golden_stocks", []))

    results = []
    for tk in tickers:
        res = analyze_single_ticker(tk, config)
        results.append(res)

    # Sort: ACTIONABLE first, then NEAR_SETUP, then by score descending
    priority_map = {
        "ACTIONABLE": 0,
        "NEAR_SETUP": 1,
        "WATCHING": 2,
        "GAP_SKIP": 3,
        "EARNINGS_BLACKOUT": 4,
        "NEUTRAL": 5,
        "NO_DATA": 6,
    }
    results.sort(
        key=lambda x: (priority_map.get(x.get("status", "WATCHING"), 9), -x.get("score", 0))
    )

    actionable = [r for r in results if r.get("status") == "ACTIONABLE"]
    near_setup = [r for r in results if r.get("status") == "NEAR_SETUP"]
    blackout = [r for r in results if r.get("status") == "EARNINGS_BLACKOUT"]

    return {
        "timestamp": format_ny_timestamp(),
        "total_scanned": len(results),
        "actionable_count": len(actionable),
        "near_setup_count": len(near_setup),
        "blackout_count": len(blackout),
        "results": results,
    }
