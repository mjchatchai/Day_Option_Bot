"""
Core Fibonacci + EMA 200 + Volume RVOL + Reversal Candle Scanner for Day Option Bot.
Implements:
1. TF 2m EMA 200 Trend Filter:
   - Price < EMA 200 -> DOWNTREND (BUY PUT ONLY)
   - Price > EMA 200 -> UPTREND (BUY CALL ONLY)
2. Previous-Day Asymmetric Fibonacci Auto-Drawing (TF 2m of Previous Trading Day):
   - Downtrend: Point 1 (Low) = Wick Low of lowest bar; Point 2 (High) = Body High of highest bar
   - Uptrend:   Point 1 (High) = Wick High of highest bar; Point 2 (Low) = Body Low of lowest bar
3. Smart Fibo Level Selection (0.500 for SPY/QQQ; 0.618 Golden Zone primary for Individual Stocks)
4. Reversal Candle + Volume Spike (RVOL >= 1.15x) + Morning Gap & Earnings Blackout Filters
5. TP1 (Next Fibo Line + Lock Profit), TP2 (2nd Fibo Line Runner), SL (Opposite Fibo Line)
"""

from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import yfinance as yf

from .earnings_filter import check_earnings_blackout
from .options_pricing import select_and_price_realtime_option


def fetch_intraday_2m(ticker: str, period: str = "5d") -> pd.DataFrame:
    """Fetch 2-minute intraday OHLCV data for EMA 200 and Previous-Day Fibonacci calculation."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval="2m")
        if df is None or df.empty:
            df = t.history(period="5d", interval="5m")
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def compute_intraday_indicators(df: pd.DataFrame, ema_period: int = 200, vol_ma_period: int = 20) -> pd.DataFrame:
    """Compute EMA 200, Volume MA 20, RVOL, BodyHigh/BodyLow, and Session VWAP."""
    df = df.copy()
    df["EMA200"] = df["Close"].ewm(span=ema_period, adjust=False).mean()
    df["VolMA"] = df["Volume"].rolling(vol_ma_period, min_periods=5).mean()
    df["RVOL"] = np.where(df["VolMA"] > 0, df["Volume"] / df["VolMA"], 1.0)
    df["BodyHigh"] = np.maximum(df["Open"], df["Close"])
    df["BodyLow"] = np.minimum(df["Open"], df["Close"])
    df["Date"] = df.index.date

    typ_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    df["TPV"] = typ_price * df["Volume"]
    df["CumTPV"] = df.groupby("Date")["TPV"].cumsum()
    df["CumVol"] = df.groupby("Date")["Volume"].cumsum()
    df["VWAP"] = np.where(df["CumVol"] > 0, df["CumTPV"] / df["CumVol"], df["Close"])
    return df


def extract_prev_day_fibo_anchors(prev_day_df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract previous trading day's asymmetric Fibonacci anchors per user specification:
    - Downtrend Fibo: Point 1 = Wick Low of lowest bar, Point 2 = Body High of highest bar
    - Uptrend Fibo:   Point 1 = Wick High of highest bar, Point 2 = Body Low of lowest bar
    """
    idx_min = prev_day_df["Low"].idxmin()
    idx_max = prev_day_df["High"].idxmax()

    wick_low = float(prev_day_df.loc[idx_min, "Low"])
    body_low = float(prev_day_df.loc[idx_min, "BodyLow"])
    wick_high = float(prev_day_df.loc[idx_max, "High"])
    body_high = float(prev_day_df.loc[idx_max, "BodyHigh"])

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


def detect_reversal_candle(curr_df: pd.DataFrame, trend_direction: str, level_price: float, rng: float) -> Dict:
    """
    Inspect the latest 2 bars to detect Fibonacci level touch + reversal candlestick confirmation.
    """
    last_bar = curr_df.iloc[-1]
    prev_bar = curr_df.iloc[-2] if len(curr_df) >= 2 else last_bar

    c_open = float(last_bar["Open"])
    c_high = float(last_bar["High"])
    c_low = float(last_bar["Low"])
    c_close = float(last_bar["Close"])
    p_open = float(prev_bar["Open"])
    p_high = float(prev_bar["High"])
    p_low = float(prev_bar["Low"])
    p_close = float(prev_bar["Close"])

    tol = max(rng * 0.02, c_close * 0.0012)
    bar_range = max(0.001, c_high - c_low)

    if trend_direction == "DOWNTREND":
        # Resistance bounce at Fibo level -> looking for Bearish Rejection / Shooting Star / Bearish Engulfing
        touched = (max(c_high, p_high) >= level_price - tol) and (min(c_low, p_low) <= level_price + tol * 1.5)
        upper_wick = c_high - max(c_open, c_close)
        is_bearish_body = c_close < c_open
        is_shooting_star = (upper_wick / bar_range) >= 0.45 and c_close <= level_price + tol * 0.5
        is_bearish_engulf = is_bearish_body and (c_close <= p_low or c_close <= level_price)

        confirmed = touched and (is_bearish_engulf or is_shooting_star)
        if is_shooting_star and is_bearish_body:
            pattern_name = "Bearish Shooting Star Rejection (ไส้บนปฎิเสธราคา + ปิดแดง)"
        elif is_bearish_engulf:
            pattern_name = "Bearish Reversal Candle (แท่งแดงกลับตัวใต้แนว Fibo)"
        elif touched:
            pattern_name = "Testing Fibo Resistance (กำลังทดสอบแนวต้าน รอแท่งแดงยืนยัน)"
        else:
            pattern_name = "Waiting for Pullback to Fibo (รอราคารีบาวด์แตะเส้น)"
    else:
        # Uptrend: Support bounce at Fibo level -> looking for Bullish Rejection / Hammer / Bullish Engulfing
        touched = (min(c_low, p_low) <= level_price + tol) and (max(c_high, p_high) >= level_price - tol * 1.5)
        lower_wick = min(c_open, c_close) - c_low
        is_bullish_body = c_close > c_open
        is_hammer = (lower_wick / bar_range) >= 0.45 and c_close >= level_price - tol * 0.5
        is_bullish_engulf = is_bullish_body and (c_close >= p_high or c_close >= level_price)

        confirmed = touched and (is_bullish_engulf or is_hammer)
        if is_hammer and is_bullish_body:
            pattern_name = "Bullish Hammer Rejection (ไส้ล่างดึงกลับ + ปิดเขียว)"
        elif is_bullish_engulf:
            pattern_name = "Bullish Reversal Candle (แท่งเขียวกลับตัวยืนเหนือแนว Fibo)"
        elif touched:
            pattern_name = "Testing Fibo Support (กำลังทดสอบแนวรับ รอแท่งเขียวยืนยัน)"
        else:
            pattern_name = "Waiting for Pullback to Fibo (รอราคาย่อตัวแตะเส้น)"

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
    min_rvol = float(vol_cfg.get("min_rvol", 1.15))
    max_gap_pct = float(rules.get("gap_filter", {}).get("max_morning_gap_pct", 2.0))
    etf_list = set(config.get("watchlist", {}).get("etf_index", ["SPY", "QQQ"]))

    # 1. Check Earnings Blackout
    earnings_info = check_earnings_blackout(ticker)

    # 2. Fetch 2m Intraday Data
    df_raw = fetch_intraday_2m(ticker)
    if df_raw.empty or len(df_raw) < 30:
        return {
            "ticker": ticker,
            "status": "NO_DATA",
            "earnings": earnings_info,
            "reason": "ไม่สามารถดึงข้อมูลกราฟ Intraday 2m ได้",
        }

    df = compute_intraday_indicators(df_raw, ema_period=ema_period)
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

    last_row = curr_df.iloc[-1]
    prev_row = curr_df.iloc[-2] if len(curr_df) >= 2 else last_row

    spot = round(float(last_row["Close"]), 2)
    ema200 = round(float(last_row["EMA200"]), 2)
    vwap = round(float(last_row["VWAP"]), 2)
    curr_rvol = round(float(max(last_row["RVOL"], prev_row["RVOL"])), 2)

    # 3. Determine EMA 200 Trend Filter
    if spot < ema200:
        trend_direction = "DOWNTREND"
        option_action = "BUY PUT"
        trend_thai = "ขาลง (ราคาต่ำกว่า EMA 200 -> ดักซื้อ PUT เท่านั้น)"
    else:
        trend_direction = "UPTREND"
        option_action = "BUY CALL"
        trend_thai = "ขาขึ้น (ราคาสูงกว่า EMA 200 -> ดักซื้อ CALL เท่านั้น)"

    # 4. Compute Previous-Day Asymmetric Fibonacci Levels
    anchors = extract_prev_day_fibo_anchors(prev_df)
    fibo = build_fibo_levels(anchors, trend_direction)
    rng = fibo["range_usd"]

    # Check Morning Gap
    first_open = float(curr_df.iloc[0]["Open"])
    prev_close = float(prev_df.iloc[-1]["Close"])
    morning_gap_pct = round(abs(first_open - prev_close) / max(0.01, prev_close) * 100.0, 2)

    # Smart Primary vs Secondary Fibo Level Routing (from 90-Day Backtest)
    is_etf = ticker.upper() in etf_list
    primary_lvl_key = "0.500" if is_etf else "0.618"
    secondary_lvl_key = "0.618" if is_etf else "0.500"

    lvl_500 = fibo["0.500"]
    lvl_618 = fibo["0.618"]
    dist_500_pct = abs(spot - lvl_500) / spot * 100.0
    dist_618_pct = abs(spot - lvl_618) / spot * 100.0

    # Choose active evaluation level (closest or primary)
    if dist_618_pct <= dist_500_pct or (not is_etf and dist_618_pct <= 0.65):
        active_lvl_key = "0.618"
    elif dist_500_pct <= 0.45:
        active_lvl_key = "0.500"
    else:
        active_lvl_key = primary_lvl_key

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

    # Lock-profit trailing stop price once TP1 is reached (measured from Fibo entry price)
    if trend_direction == "DOWNTREND":
        lock_sl_at_tp1 = round(active_lvl_price - (active_lvl_price - tp1_price) * 0.45, 2)
    else:
        lock_sl_at_tp1 = round(active_lvl_price + (tp1_price - active_lvl_price) * 0.45, 2)

    # 5. Detect Reversal Candlestick & Volume RVOL
    candle_check = detect_reversal_candle(curr_df, trend_direction, active_lvl_price, rng)
    rvol_passed = curr_rvol >= min_rvol

    # 6. Price Real-Time Option Contract via Black-Scholes (Zero 15m Delay)
    opt_quote = select_and_price_realtime_option(
        ticker=ticker,
        direction=option_action,
        spot_realtime=spot,
        tp1_underlying=tp1_price,
        tp2_underlying=tp2_price,
        sl_underlying=sl_price,
        fibo_entry_underlying=active_lvl_price,
    )

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
    elif morning_gap_pct > max_gap_pct and dist_active_pct > 1.5:
        status = "GAP_SKIP"
        status_thai = f"⚠️ เปิดกระโดด Gap {morning_gap_pct:.2f}% ห่างกรอบ Fibo เมื่อวาน (แนะนำข้าม)"
    elif candle_check["confirmed"] and rvol_passed:
        status = "ACTIONABLE"
        status_thai = f"🔥 สัญญาณเข้าเทรด {option_action} ที่ Fibo {active_lvl_key} (แท่งกลับตัว + RVOL {curr_rvol:.2f}x)"
    elif candle_check["touched"] or dist_active_pct <= 0.45:
        status = "NEAR_SETUP"
        status_thai = f"👀 ราคาเข้าใกล้/กำลังทดสอบ Fibo {active_lvl_key} (ห่าง {dist_active_pct:.2f}%) รอแท่งกลับตัว+Volume"
    else:
        status = "WATCHING"
        status_thai = f"⏳ รอราคาย่อ/เด้งมาหา Fibo {primary_lvl_key} (${fibo[primary_lvl_key]:.2f} | ห่าง {dist_active_pct:.2f}%)"

    return {
        "ticker": ticker,
        "is_etf": is_etf,
        "status": status,
        "status_thai": status_thai,
        "score": min(100, score),
        "prev_trading_date": str(prev_date),
        "current_trading_date": str(curr_date),
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
        "option_quote": opt_quote.__dict__,
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
        "NO_DATA": 5,
    }
    results.sort(key=lambda x: (priority_map.get(x.get("status", "WATCHING"), 9), -x.get("score", 0)))

    actionable = [r for r in results if r.get("status") == "ACTIONABLE"]
    near_setup = [r for r in results if r.get("status") == "NEAR_SETUP"]
    blackout = [r for r in results if r.get("status") == "EARNINGS_BLACKOUT"]

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_scanned": len(results),
        "actionable_count": len(actionable),
        "near_setup_count": len(near_setup),
        "blackout_count": len(blackout),
        "results": results,
    }
