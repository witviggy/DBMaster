from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Optional


def _to_iso(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def _month_end(d: date) -> date:
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


def _add_months(d: date, months: int) -> date:
    year = d.year + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    return date(year, month, 1)


def _clamp_range(start: date, end: date, min_date: date, max_date: date, *, clamp_end: bool = True) -> tuple[date, date]:
    start = max(start, min_date)
    if clamp_end:
        end = min(end, max_date + timedelta(days=1))
    if end <= start:
        end = start + timedelta(days=1)
    return start, end


def derive_relative_date_filter(
    user_query: str,
    *,
    anchor_date: date,
    min_date: date,
    max_date: date,
) -> Optional[str]:
    """Return a deterministic half-open date predicate for common relative phrases."""
    text = (user_query or "").lower()

    if re.search(r"\b(current|this)\s+month\b", text):
        start = _month_start(anchor_date)
        end = _month_end(start)
        start, end = _clamp_range(start, end, min_date, max_date, clamp_end=False)
        return f"transaction_date >= '{_to_iso(start)}' AND transaction_date < '{_to_iso(end)}'"

    if re.search(r"\blast\s+month\b", text):
        start = _add_months(_month_start(anchor_date), -1)
        end = _month_start(anchor_date)  # exclusive: last month ends the day before this month starts
        start, end = _clamp_range(start, end, min_date, max_date, clamp_end=False)
        return f"transaction_date >= '{_to_iso(start)}' AND transaction_date < '{_to_iso(end)}'"

    months_match = re.search(r"\blast\s+(\d+)\s+months?\b", text)
    if months_match:
        n = int(months_match.group(1))
        start = _add_months(_month_start(anchor_date), -n)
        end = _month_start(anchor_date)  # exclusive: range ends at the start of the current month
        start, end = _clamp_range(start, end, min_date, max_date, clamp_end=False)
        return f"transaction_date >= '{_to_iso(start)}' AND transaction_date < '{_to_iso(end)}'"

    days_match = re.search(r"\blast\s+(\d+)\s+days?\b", text)
    if days_match:
        n = int(days_match.group(1))
        start = anchor_date - timedelta(days=n - 1)
        end = anchor_date + timedelta(days=1)
        start, end = _clamp_range(start, end, min_date, max_date)
        return f"transaction_date >= '{_to_iso(start)}' AND transaction_date <= '{_to_iso(end - timedelta(days=1))}'"

    if re.search(r"\bthis year\b|\byear to date\b", text):
        start = date(anchor_date.year, 1, 1)
        end = date(anchor_date.year + 1, 1, 1)
        start, end = _clamp_range(start, end, min_date, max_date)
        return f"transaction_date >= '{_to_iso(start)}' AND transaction_date < '{_to_iso(end)}'"

    if re.search(r"\blast year\b", text):
        start = date(anchor_date.year - 1, 1, 1)
        end = date(anchor_date.year, 1, 1)
        start, end = _clamp_range(start, end, min_date, max_date)
        return f"transaction_date >= '{_to_iso(start)}' AND transaction_date < '{_to_iso(end)}'"

    return None


def resolve_posting_period_hint(
    user_query: str,
    *,
    anchor_date: date,
    max_date: date,
) -> Optional[str]:
    """
    Resolve relative time phrases to concrete posting_period label(s) for kpi_base_metrics.
    Clamps to max_date so queries for periods beyond the data window use the latest available month.
    Returns a hint string for LLM injection, e.g. 'Jul 2026' or 'Feb 2026 to Jul 2026'.
    """
    text = (user_query or "").lower()
    latest = _month_start(max_date)  # latest month with data

    if re.search(r"\b(current|this)\s+month\b", text):
        target = _month_start(anchor_date)
        # clamp forward: if current month has no data yet, use the latest available month
        if target > latest:
            target = latest
        return target.strftime("%b %Y")

    if re.search(r"\blast\s+month\b", text):
        target = _add_months(_month_start(anchor_date), -1)
        if target > latest:
            target = latest
        return target.strftime("%b %Y")

    months_match = re.search(r"\blast\s+(\d+)\s+months?\b", text)
    if months_match:
        n = int(months_match.group(1))
        end_month = _month_start(anchor_date)
        if end_month > latest:
            end_month = _add_months(latest, 1)  # exclusive upper bound
        start_month = _add_months(end_month, -n)
        last_month = _add_months(end_month, -1)
        return f"{start_month.strftime('%b %Y')} to {last_month.strftime('%b %Y')}"

    if re.search(r"\b(this year|year to date)\b", text):
        start_month = date(anchor_date.year, 1, 1)
        end_month = min(_month_start(anchor_date), latest)
        return f"{start_month.strftime('%b %Y')} to {end_month.strftime('%b %Y')}"

    if re.search(r"\blast year\b", text):
        start_month = date(anchor_date.year - 1, 1, 1)
        end_month = date(anchor_date.year - 1, 12, 1)
        return f"{start_month.strftime('%b %Y')} to {end_month.strftime('%b %Y')}"

    return None


def apply_relative_date_filter(
    sql: str,
    user_query: str,
    *,
    anchor_date: date,
    min_date: date,
    max_date: date,
) -> str:
    """Inject a deterministic relative-date predicate into generated SQL when appropriate."""
    predicate = derive_relative_date_filter(
        user_query,
        anchor_date=anchor_date,
        min_date=min_date,
        max_date=max_date,
    )
    if not predicate or "transaction_date" not in sql.lower():
        return sql

    sql = sql.strip().rstrip(";")
    if re.search(r"\bwhere\b", sql, flags=re.IGNORECASE):
        where_match = re.search(r"\bwhere\b(?:(?!\b(group|order|limit|having|union|intersect|except)\b).)*", sql, flags=re.IGNORECASE | re.DOTALL)
        if where_match:
            prefix = sql[: where_match.end()]
            suffix = sql[where_match.end() :]
            return f"{prefix} AND {predicate}{suffix}"
        return f"{sql} AND {predicate}"

    return f"{sql} WHERE {predicate}"
