"""Normalisation used by the grounding check and the eval: whitespace, thousands separators, date formats."""
from __future__ import annotations

import datetime as dt
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

CENT = Decimal("0.01")
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
     "december"], start=1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
MONTHS["sept"] = 9


def money(x) -> Decimal | None:
    """A float, int, Decimal or string amount as a Decimal rounded to pence (None if not a number)."""
    if x is None or x == "":
        return None
    if isinstance(x, str):
        parsed = parse_amount_token(x)
        if parsed is None:
            return None
        return parsed
    try:
        return Decimal(str(x)).quantize(CENT, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return None


def parse_amount_token(tok: str) -> Decimal | None:
    """'1,234.50', '1.234,50', '1 234.50', '(12.00)', '-12.00', '12' -> Decimal (sign dropped)."""
    t = tok.strip().strip("()").replace(" ", " ")
    t = re.sub(r"^[^\d-]+|[^\d]+$", "", t).lstrip("-")
    if not t:
        return None
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+,\d{1,2}", t) or re.fullmatch(r"\d+,\d{2}", t):
        t = t.replace(".", "").replace(",", ".")  # European style
    elif re.fullmatch(r"\d{1,3}(?:[ ,]\d{3})+(?:\.\d+)?", t):
        t = t.replace(",", "").replace(" ", "")
    elif not re.fullmatch(r"\d+(?:\.\d+)?", t):
        return None
    try:
        return Decimal(t).quantize(CENT, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return None


_NUM_RE = re.compile(r"\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?|\d{1,3}(?: \d{3})+(?:\.\d{1,2})?|\d+(?:[.,]\d{1,2})?(?!\d)")


def numbers_in(text: str) -> set[Decimal]:
    """Every amount-like number in the text, after removing thousands separators (both styles)."""
    out: set[Decimal] = set()
    for m in _NUM_RE.finditer(text or ""):
        v = parse_amount_token(m.group(0))
        if v is not None:
            out.add(v)
    return out


def parse_date(s: str | None) -> dt.date | None:
    """ISO 'YYYY-MM-DD' (what the model is asked for) or a UK-style printed date."""
    if not s:
        return None
    s = s.strip()
    try:
        return dt.date.fromisoformat(s[:10])
    except ValueError:
        pass
    if re.fullmatch(r"\d{8}", s):  # compact date: day first (UK), else year first
        return _mk(int(s[4:]), int(s[2:4]), int(s[:2])) if 1990 <= int(s[4:]) <= 2100 else \
            _mk(int(s[:4]), int(s[4:6]), int(s[6:]))
    found = dates_in(s)
    return min(found) if len(found) == 1 else None


_D_NUM = re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})\b")
_D_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_D_TXT = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?[ -]([A-Za-z]{3,9})\.?,?[ -](\d{4})\b")
_D_US_TXT = re.compile(r"\b([A-Za-z]{3,9})\.? (\d{1,2})(?:st|nd|rd|th)?,? (\d{4})\b")


def _mk(y: int, m: int, d: int) -> dt.date | None:
    if y < 100:
        y += 2000
    try:
        return dt.date(y, m, d)
    except ValueError:
        return None


def dates_in(text: str) -> set[dt.date]:
    """Dates printed in the text: dd/mm/yyyy (UK order), dd-mm-yy, yyyy-mm-dd, '3 Nov 2026', 'November 3, 2026'."""
    t = text or ""
    out: set[dt.date] = set()
    for d, m, y in _D_NUM.findall(t):
        if v := _mk(int(y), int(m), int(d)):
            out.add(v)
    for y, m, d in _D_ISO.findall(t):
        if v := _mk(int(y), int(m), int(d)):
            out.add(v)
    for d, mon, y in _D_TXT.findall(t):
        if mon.lower() in MONTHS and (v := _mk(int(y), MONTHS[mon.lower()], int(d))):
            out.add(v)
    for mon, d, y in _D_US_TXT.findall(t):
        if mon.lower() in MONTHS and (v := _mk(int(y), MONTHS[mon.lower()], int(d))):
            out.add(v)
    return out


def alnum(text: str) -> str:
    """Lowercase letters and digits only: removes whitespace, punctuation and separators."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower().replace("&", "and"))


def squash_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def supplier_key(name: str | None) -> str:
    """Stable key for a supplier name: lowercase alnum without common company suffixes."""
    n = (name or "").lower()
    n = re.sub(r"\b(ltd|limited|plc|llp|co|company|inc)\b\.?", " ", n)
    return alnum(n)


def doc_number_key(num: str | None) -> str:
    """'INV-0042' and 'inv 0042' compare equal; leading zeros in the numeric part are kept."""
    return alnum(num or "")


def vat_number_key(v: str | None) -> str:
    return re.sub(r"[^0-9A-Z]", "", (v or "").upper()).removeprefix("GB")
