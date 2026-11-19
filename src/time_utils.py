"""
Shared America/New_York (ET) timezone, DST, NYSE trading-window,
and regular-hours DataFrame normalization utilities for Day Option Bot.
"""

from datetime import date, datetime, time as dt_time, timedelta
from typing import Optional, Tuple
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd


NY_TZ = ZoneInfo("America/New_York")


def now_ny() -> datetime:
    """Return the current timezone-aware datetime in America/New_York (EST/EDT)."""
    return datetime.now(NY_TZ)


def today_ny() -> date:
    """Return the current calendar date in America/New_York."""
    return now_ny().date()


def _to_ny_datetime(dt: Optional[datetime]) -> datetime:
    """Convert a datetime to America/New_York; treat naive datetimes as NY wall-clock."""
    if dt is None:
        return now_ny()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=NY_TZ)
    return dt.astimezone(NY_TZ)


def format_ny_timestamp(dt: Optional[datetime] = None) -> str:
    """Format a datetime (or current NY time) as 'YYYY-MM-DD HH:MM:SS ET'."""
    return _to_ny_datetime(dt).strftime("%Y-%m-%d %H:%M:%S ET")


def _easter_sunday(year: int) -> date:
    """Compute Gregorian Easter Sunday date for a given year."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _observed_holiday(d: date) -> date:
    """Apply US market observed rule: Saturday -> Friday, Sunday -> Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """Return the n-th (1-indexed) occurrence of weekday (0=Mon..6=Sun) in month."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + (n - 1) * 7)


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """Return the last occurrence of weekday (0=Mon..6=Sun) in month."""
    if month == 12:
        next_first = date(year + 1, 1, 1)
    else:
        next_first = date(year, month + 1, 1)
    last_day = next_first - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def is_us_market_holiday(d: date) -> bool:
    """Return True if date d is a standard NYSE full-day market holiday."""
    y = d.year
    holidays = {
        _observed_holiday(date(y, 1, 1)),                    # New Year's Day
        _observed_holiday(date(y + 1, 1, 1)),                # Next New Year's observed on Dec 31
        _nth_weekday_of_month(y, 1, 0, 3),                   # MLK Day (3rd Mon Jan)
        _nth_weekday_of_month(y, 2, 0, 3),                   # Presidents' Day (3rd Mon Feb)
        _easter_sunday(y) - timedelta(days=2),               # Good Friday
        _last_weekday_of_month(y, 5, 0),                     # Memorial Day (Last Mon May)
        _observed_holiday(date(y, 7, 4)),                    # Independence Day
        _nth_weekday_of_month(y, 9, 0, 1),                   # Labor Day (1st Mon Sep)
        _nth_weekday_of_month(y, 11, 3, 4),                  # Thanksgiving Day (4th Thu Nov)
        _observed_holiday(date(y, 12, 25)),                  # Christmas Day
    }
    if y >= 2021:
        holidays.add(_observed_holiday(date(y, 6, 19)))      # Juneteenth
    return d in holidays


def is_within_2h_window_ny(
    dt: Optional[datetime] = None,
    start_hm: Tuple[int, int] = (9, 35),
    end_hm: Tuple[int, int] = (11, 30),
    check_holidays: bool = True,
) -> bool:
    """
    Return True if dt (default: current NY time) falls on a US market weekday
    inside the half-open [start_hm, end_hm) America/New_York window (default 09:35-11:30 ET).
    """
    target = _to_ny_datetime(dt)
    if target.weekday() >= 5:
        return False
    if check_holidays and is_us_market_holiday(target.date()):
        return False

    cur_t = target.timetz().replace(tzinfo=None)
    start_t = dt_time(int(start_hm[0]), int(start_hm[1]))
    end_t = dt_time(int(end_hm[0]), int(end_hm[1]))
    return start_t <= cur_t < end_t


def ensure_ny_regular_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a yfinance or synthetic OHLCV intraday DataFrame to sorted, deduplicated
    America/New_York regular market hours (09:30-15:59 ET, Mon-Fri) with valid numeric OHLCV.
    Handles MultiIndex columns, naive/tz-aware/mixed-offset indexes, DST boundaries,
    NaN/non-positive OHLC prices, and duplicate/unsorted timestamps.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Flatten MultiIndex columns if yfinance returned (Price, Ticker) or (Ticker, Price)
    if isinstance(out.columns, pd.MultiIndex):
        ohlcv = {"Open", "High", "Low", "Close"}
        chosen_lvl = 0
        for lvl in range(out.columns.nlevels):
            vals = set(out.columns.get_level_values(lvl))
            if ohlcv.issubset(vals):
                chosen_lvl = lvl
                break
        out.columns = out.columns.get_level_values(chosen_lvl)

    out = out.loc[:, ~out.columns.duplicated(keep="first")]

    # Normalize index to tz-aware America/New_York DatetimeIndex
    if not isinstance(out.index, pd.DatetimeIndex):
        parsed = []
        for item in out.index:
            try:
                parsed.append(pd.to_datetime(item, errors="coerce"))
            except Exception:
                parsed.append(pd.NaT)

        has_aware = any(
            (not pd.isna(ts)) and getattr(ts, "tzinfo", None) is not None
            for ts in parsed
        )
        if not has_aware:
            out.index = pd.DatetimeIndex(parsed)
        else:
            converted = []
            for ts in parsed:
                if pd.isna(ts):
                    converted.append(pd.NaT)
                elif getattr(ts, "tzinfo", None) is not None:
                    converted.append(ts.tz_convert(NY_TZ))
                else:
                    try:
                        converted.append(
                            ts.tz_localize(
                                NY_TZ, nonexistent="shift_forward", ambiguous="NaT"
                            )
                        )
                    except Exception:
                        converted.append(pd.NaT)
            out.index = pd.DatetimeIndex(converted, tz=NY_TZ)

    out = out[~out.index.isna()]
    if out.empty:
        return out

    if out.index.tz is None:
        try:
            out.index = out.index.tz_localize(
                NY_TZ, nonexistent="shift_forward", ambiguous="infer"
            )
        except Exception:
            out.index = out.index.tz_localize(
                NY_TZ, nonexistent="shift_forward", ambiguous="NaT"
            )
            out = out[~out.index.isna()]
    else:
        out.index = out.index.tz_convert(NY_TZ)

    if out.empty:
        return out

    # Sort chronologically and keep latest bar on duplicate timestamps
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    # Filter to regular US market hours (Mon-Fri, 09:30 <= t <= 15:59 ET)
    out = out[out.index.dayofweek < 5]
    if out.empty:
        return out
    out = out.between_time("09:30", "15:59")
    if out.empty:
        return out

    # Coerce OHLCV to finite positive prices / non-negative volume
    price_cols = [c for c in ("Open", "High", "Low", "Close") if c in out.columns]
    for col in price_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)

    if price_cols:
        out = out.dropna(subset=price_cols)
        for col in price_cols:
            out = out[out[col] > 0]

    if "Volume" in out.columns:
        vol = pd.to_numeric(out["Volume"], errors="coerce")
        vol = vol.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        out["Volume"] = vol

    return out
