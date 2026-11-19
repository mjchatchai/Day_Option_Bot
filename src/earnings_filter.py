"""
Earnings Announcement Blackout Filter for Day Option Bot.
Checks upcoming/recent earnings dates for each ticker and blocks trading when an earnings
report is within the blackout window (default: 2 days before through 1 day after), protecting
against IV Crush, wide spreads, and binary price jumps.
"""

from datetime import datetime, date
import re
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import yfinance as yf

from .time_utils import NY_TZ, today_ny


ETF_TICKERS = {"SPY", "QQQ", "IWM", "DIA", "SOXL", "TQQQ"}
NON_EQUITY_QUOTE_TYPES = {
    "ETF",
    "INDEX",
    "MUTUALFUND",
    "CRYPTOCURRENCY",
    "CURRENCY",
    "FUTURE",
}


def _coerce_to_date(val: Any) -> Optional[date]:
    """Safely coerce date/datetime/Timestamp/np.datetime64/ISO string to date, rejecting numbers."""
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float, np.number)):
        return None

    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass

    # Check pd.Timestamp before datetime because pd.Timestamp subclasses datetime
    if isinstance(val, pd.Timestamp):
        if val.tzinfo is not None:
            return val.tz_convert(NY_TZ).date()
        return val.date()

    if isinstance(val, datetime):
        if val.tzinfo is not None:
            return val.astimezone(NY_TZ).date()
        return val.date()

    if isinstance(val, date):
        return val

    if isinstance(val, np.datetime64):
        ts = pd.Timestamp(val)
        if pd.isna(ts):
            return None
        return ts.date()

    if isinstance(val, str):
        s = val.strip()
        if not s or s.lower() in {"nat", "nan", "none", "null", "n/a"}:
            return None
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            try:
                return date.fromisoformat(s)
            except ValueError:
                return None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is not None:
                return dt.astimezone(NY_TZ).date()
            return dt.date()
        except ValueError:
            pass
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            return None
        if ts.tzinfo is not None:
            return ts.tz_convert(NY_TZ).date()
        return ts.date()

    return None


def _extract_dates_from_container(raw: Any) -> List[date]:
    """Extract all valid dates from a scalar or sequence/Series/Index container."""
    dates: List[date] = []
    if isinstance(raw, (list, tuple, set, pd.Series, pd.Index, np.ndarray)):
        for item in raw:
            d = _coerce_to_date(item)
            if d is not None:
                dates.append(d)
    else:
        d = _coerce_to_date(raw)
        if d is not None:
            dates.append(d)
    return dates


def _extract_earnings_dates_from_calendar(cal: Any) -> List[date]:
    """
    Parse earnings dates from yfinance .calendar (dict or DataFrame) or .get_earnings_dates()
    without misinterpreting numeric EPS keys such as 'Earnings High'.
    """
    if cal is None:
        return []

    dates: List[date] = []
    target_keys = {"earnings date", "earningsdate", "earnings call date"}

    if isinstance(cal, dict):
        for k, v in cal.items():
            norm_k = str(k).strip().lower().replace("_", " ")
            if norm_k in target_keys:
                dates.extend(_extract_dates_from_container(v))
    elif isinstance(cal, pd.DataFrame):
        if cal.empty:
            return []
        # Subcase A: row-oriented calendar DataFrame (index labels contain 'Earnings Date')
        for idx in cal.index:
            norm_idx = str(idx).strip().lower().replace("_", " ")
            if norm_idx in target_keys:
                row_slice = cal.loc[idx]
                if isinstance(row_slice, pd.DataFrame):
                    for col in row_slice.columns:
                        dates.extend(_extract_dates_from_container(row_slice[col]))
                else:
                    dates.extend(_extract_dates_from_container(row_slice))
        # Subcase B: column-oriented DataFrame
        for col in cal.columns:
            norm_col = str(col).strip().lower().replace("_", " ")
            if norm_col in target_keys:
                dates.extend(_extract_dates_from_container(cal[col]))
        # Subcase C: get_earnings_dates() DataFrame whose DatetimeIndex holds earnings dates
        if isinstance(cal.index, pd.DatetimeIndex):
            dates.extend(_extract_dates_from_container(cal.index))
    elif isinstance(cal, (pd.Series, pd.DatetimeIndex, list, tuple)):
        dates.extend(_extract_dates_from_container(cal))
    else:
        d = _coerce_to_date(cal)
        if d is not None:
            dates.append(d)

    return list(dict.fromkeys(dates))


def _select_relevant_earnings_date(
    candidates: List[date],
    ref_dt: date,
    days_before: int,
    days_after: int,
) -> Optional[date]:
    """Choose active blackout date first, else nearest upcoming date, else most recent past date."""
    if not candidates:
        return None

    in_blackout = [
        d for d in candidates if (-days_after) <= (d - ref_dt).days <= days_before
    ]
    if in_blackout:
        return min(
            in_blackout,
            key=lambda d: (abs((d - ref_dt).days), 0 if d >= ref_dt else 1),
        )

    upcoming = [d for d in candidates if d > ref_dt]
    if upcoming:
        return min(upcoming)

    return max(candidates)


def check_earnings_blackout(
    ticker: str,
    days_before: int = 2,
    days_after: int = 1,
    reference_date: Optional[Union[date, datetime, pd.Timestamp, str]] = None,
    enabled: bool = True,
    blackout_days_before: Optional[int] = None,
    blackout_days_after: Optional[int] = None,
    is_etf: Optional[bool] = None,
) -> Dict:
    """
    Returns earnings status dictionary:
    {
      "ticker": str,
      "is_blackout": bool,
      "earnings_date": Optional[str],
      "days_until_earnings": Optional[int],
      "status_label": str
    }
    """
    tk = str(ticker or "").upper().strip()
    eff_before = int(
        blackout_days_before if blackout_days_before is not None else days_before
    )
    eff_after = int(
        blackout_days_after if blackout_days_after is not None else days_after
    )
    ref_dt = _coerce_to_date(reference_date) or today_ny()

    if not enabled:
        return {
            "ticker": tk,
            "is_blackout": False,
            "earnings_date": None,
            "days_until_earnings": None,
            "status_label": "SAFE (Earnings Filter Disabled)",
        }

    if is_etf is True or tk in ETF_TICKERS:
        return {
            "ticker": tk,
            "is_blackout": False,
            "earnings_date": None,
            "days_until_earnings": None,
            "status_label": "SAFE (ETF Index)",
        }

    candidate_dates: List[date] = []
    try:
        yf_obj = yf.Ticker(tk)
        # Generic non-equity/ETF check before .calendar to avoid 404 errors on ETFs
        try:
            fast_info = getattr(yf_obj, "fast_info", None)
            qt = (
                fast_info.get("quoteType")
                if (fast_info is not None and hasattr(fast_info, "get"))
                else None
            )
            if isinstance(qt, str) and qt.upper().strip() in NON_EQUITY_QUOTE_TYPES:
                return {
                    "ticker": tk,
                    "is_blackout": False,
                    "earnings_date": None,
                    "days_until_earnings": None,
                    "status_label": f"SAFE ({qt.upper().strip()} / No Earnings)",
                }
        except Exception:
            pass

        try:
            cal = yf_obj.calendar
        except Exception:
            cal = None
        candidate_dates = _extract_earnings_dates_from_calendar(cal)

        if not candidate_dates:
            get_ed = getattr(yf_obj, "get_earnings_dates", None)
            if callable(get_ed):
                try:
                    ed_df = get_ed(limit=4)
                    if isinstance(ed_df, pd.DataFrame) and not ed_df.empty:
                        candidate_dates = _extract_earnings_dates_from_calendar(ed_df)
                except Exception:
                    pass
    except Exception:
        candidate_dates = []

    next_earn_date = _select_relevant_earnings_date(
        candidate_dates, ref_dt, eff_before, eff_after
    )

    if next_earn_date is None:
        return {
            "ticker": tk,
            "is_blackout": False,
            "earnings_date": None,
            "days_until_earnings": None,
            "status_label": "SAFE (No Upcoming Earnings)",
        }

    diff_days = (next_earn_date - ref_dt).days
    is_blocked = (-eff_after) <= diff_days <= eff_before

    if is_blocked:
        if diff_days == 0:
            label = f"🚫 BLACKOUT: ประกาศงบวันนี้ ({next_earn_date})"
        elif diff_days > 0:
            label = f"🚫 BLACKOUT: ประกาศงบในอีก {diff_days} วัน ({next_earn_date})"
        else:
            label = f"🚫 BLACKOUT: เพิ่งประกาศงบเมื่อ {abs(diff_days)} วันที่แล้ว ({next_earn_date})"
    else:
        if diff_days > 0:
            label = f"SAFE (งบถัดไป: {next_earn_date} / อีก {diff_days} วัน)"
        elif diff_days == 0:
            label = f"SAFE (วันประกาศงบ: {next_earn_date} / วันนี้)"
        else:
            label = f"SAFE (งบล่าสุด: {next_earn_date} / ผ่านมาแล้ว {abs(diff_days)} วัน)"

    return {
        "ticker": tk,
        "is_blackout": is_blocked,
        "earnings_date": str(next_earn_date),
        "days_until_earnings": diff_days,
        "status_label": label,
    }
