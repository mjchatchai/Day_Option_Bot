"""
Earnings Announcement Blackout Filter for Day Option Bot.
Checks upcoming earnings dates for each ticker and blocks trading when an earnings report
is within the blackout window (default: 2 days before through 1 day after), protecting against
IV Crush, wide spreads, and binary price jumps.
"""

from datetime import datetime, date
from typing import Dict, Optional
import yfinance as yf


ETF_TICKERS = {"SPY", "QQQ", "IWM", "DIA", "SOXL", "TQQQ"}


def check_earnings_blackout(
    ticker: str,
    days_before: int = 2,
    days_after: int = 1,
    reference_date: Optional[date] = None,
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
    tk = ticker.upper().strip()
    ref_dt = reference_date or date.today()

    if tk in ETF_TICKERS:
        return {
            "ticker": tk,
            "is_blackout": False,
            "earnings_date": None,
            "days_until_earnings": None,
            "status_label": "SAFE (ETF Index)",
        }

    next_earn_date: Optional[date] = None
    try:
        yf_obj = yf.Ticker(tk)
        cal = yf_obj.calendar
        if isinstance(cal, dict):
            ed_list = cal.get("Earnings Date") or cal.get("Earnings High")
            if isinstance(ed_list, list) and ed_list:
                first_dt = ed_list[0]
                if isinstance(first_dt, datetime):
                    next_earn_date = first_dt.date()
                elif isinstance(first_dt, date):
                    next_earn_date = first_dt
    except Exception:
        pass

    if next_earn_date is None:
        return {
            "ticker": tk,
            "is_blackout": False,
            "earnings_date": None,
            "days_until_earnings": None,
            "status_label": "SAFE (No Upcoming Earnings)",
        }

    diff_days = (next_earn_date - ref_dt).days
    is_blocked = (-days_after) <= diff_days <= days_before

    if is_blocked:
        if diff_days == 0:
            label = f"🚫 BLACKOUT: ประกาศงบวันนี้ ({next_earn_date})"
        elif diff_days > 0:
            label = f"🚫 BLACKOUT: ประกาศงบในอีก {diff_days} วัน ({next_earn_date})"
        else:
            label = f"🚫 BLACKOUT: เพิ่งประกาศงบเมื่อ {abs(diff_days)} วันที่แล้ว ({next_earn_date})"
    else:
        label = f"SAFE (งบถัดไป: {next_earn_date} / อีก {diff_days} วัน)"

    return {
        "ticker": tk,
        "is_blackout": is_blocked,
        "earnings_date": str(next_earn_date),
        "days_until_earnings": diff_days,
        "status_label": label,
    }
