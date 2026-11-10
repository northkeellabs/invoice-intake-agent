"""Code checks on an extraction. Each problem becomes a named flag with a reason shown in the review UI.

Nothing here asks the model anything. The checks decide routing; the model's own opinion of its
output is never used.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

from . import rules
from .config import LOW_OCR_CONFIDENCE
from .documents import PageText
from .normalize import CENT, alnum, dates_in, money, numbers_in, parse_date
from .schema import BILL_TYPES, Extraction

FLAG_TITLES = {
    "grounding": "Value not found in the document",
    "missing_field": "Required field missing",
    "arithmetic_total": "Net + VAT does not equal total",
    "arithmetic_lines": "Line items do not add up to the net total",
    "arithmetic_line_item": "Quantity x unit price does not equal the line amount",
    "arithmetic_vat_rate": "VAT amount does not match the VAT rate",
    "date_invalid": "Date not parseable",
    "date_in_future": "Date in the future",
    "due_before_issue": "Due date before issue date",
    "currency_invalid": "Not an ISO 4217 currency code",
    "foreign_currency": "Foreign currency",
    "vat_number_format": "VAT number format",
    "vat_invoice_incomplete": "VAT invoice details missing",
    "duplicate": "Possible duplicate",
    "bank_details_changed": "Bank details changed",
    "low_ocr_confidence": "Low OCR confidence",
    "not_a_bill": "Not exported as a bill",
    "instructions_found": "Instructions found in document",
    "account_code_invalid": "Account code not in chart of accounts",
    "extraction_failed": "Extraction failed",
}
REQUIRED_FIELDS = ("supplier_name", "document_number", "issue_date", "gross_total", "currency")
STRING_FIELDS = ("supplier_name", "supplier_address", "supplier_vat_number", "customer_name", "document_number",
                 "original_invoice_number", "bank_sort_code", "bank_account_number", "bank_iban")
AMOUNT_FIELDS = ("net_total", "vat_total", "gross_total")
CURRENCY_SYMBOLS = {"GBP": "£", "EUR": "€", "USD": "$"}
BANK_CHANGE_REASON = ("bank details on the document differ from the supplier record: possible invoice fraud: "
                      "confirm by phone using a number you already have")


def flag(name: str, reason: str) -> dict:
    return {"name": name, "title": FLAG_TITLES.get(name, name), "reason": reason[:400]}


@dataclass
class DocText:
    """Normalised views of the document text, computed once per document."""
    raw: str
    alnum: str
    numbers: set[Decimal]
    dates: set[dt.date]
    lower: str

    @classmethod
    def of(cls, pages: list[PageText]) -> "DocText":
        raw = "\n".join(p.text for p in pages)
        return cls(raw=raw, alnum=alnum(raw), numbers=numbers_in(raw), dates=dates_in(raw), lower=raw.lower())


# ------------------------------------------------------------------ grounding

def _string_grounded(value: str, text: DocText) -> bool:
    parts = [p for p in (value or "").replace("\n", ",").split(",") if alnum(p)]
    return bool(parts) and all(alnum(p) in text.alnum for p in parts)


def _amount_grounded(value, text: DocText) -> bool:
    m = money(value)
    return m is not None and abs(m) in text.numbers


def ground(ex: Extraction, text: DocText) -> dict[str, str]:
    """Per extracted value: ok | computed | not_found (absent values are not listed).

    A value is grounded if it appears in the document text after normalising whitespace, thousands
    separators and date formats. Totals that are not printed but follow from the other two (gross = net + VAT)
    and line amounts equal to quantity x unit price count as computed, not as missing.
    """
    g: dict[str, str] = {}
    for f in STRING_FIELDS:
        v = getattr(ex, f)
        if v:
            g[f] = "ok" if _string_grounded(v, text) else "not_found"
    for f in ("issue_date", "due_date"):
        v = getattr(ex, f)
        if v:
            d = parse_date(v)
            g[f] = "ok" if d and d in text.dates else "not_found"
    if ex.currency:
        code = ex.currency.strip().upper()
        sym = CURRENCY_SYMBOLS.get(code)
        hit = bool(re.search(rf"(?i)\b{re.escape(code)}\b", text.raw)) or bool(sym and sym in text.raw)
        g["currency"] = "ok" if hit else "not_found"
    for f in AMOUNT_FIELDS:
        v = getattr(ex, f)
        if v is not None:
            g[f] = "ok" if _amount_grounded(v, text) else "not_found"
    net, vat, gross = money(ex.net_total), money(ex.vat_total), money(ex.gross_total)
    if None not in (net, vat, gross) and abs(net + vat - gross) <= CENT:
        for f, others in (("net_total", ("vat_total", "gross_total")), ("vat_total", ("net_total", "gross_total")),
                          ("gross_total", ("net_total", "vat_total"))):
            if g.get(f) == "not_found" and all(g.get(o) == "ok" for o in others):
                g[f] = "computed"
    for i, r in enumerate(ex.vat_rates):
        g[f"vat_rate[{i}]"] = "ok" if _amount_grounded(r, text) else "not_found"
    for i, li in enumerate(ex.line_items):
        p = f"line[{i}]."
        if li.description:
            g[p + "description"] = "ok" if _string_grounded(li.description, text) else "not_found"
        for f in ("quantity", "vat_rate"):
            v = getattr(li, f)
            if v is not None:
                g[p + f] = "ok" if _amount_grounded(v, text) else "not_found"
        # A line amount or unit price that is not printed but equals quantity x unit price (or line / quantity)
        # counts as computed, provided the other two values are printed.
        q, u, n = money(li.quantity), money(li.unit_price), money(li.net_amount)
        consistent = None not in (q, u, n) and abs((q * u).quantize(CENT, ROUND_HALF_UP) - n) <= CENT
        u_ok = li.unit_price is not None and _amount_grounded(li.unit_price, text)
        n_ok = li.net_amount is not None and _amount_grounded(li.net_amount, text)
        if li.unit_price is not None:
            g[p + "unit_price"] = "ok" if u_ok else ("computed" if consistent and n_ok else "not_found")
        if li.net_amount is not None:
            g[p + "net_amount"] = "ok" if n_ok else ("computed" if consistent and u_ok else "not_found")
    return g


# ------------------------------------------------------------------ checks

def _round(x: Decimal) -> Decimal:
    return x.quantize(CENT, rounding=ROUND_HALF_UP)


def arithmetic(ex: Extraction) -> list[dict]:
    out = []
    net, vat, gross = money(ex.net_total), money(ex.vat_total), money(ex.gross_total)
    if None not in (net, vat, gross) and abs(net + vat - gross) > CENT:
        out.append(flag("arithmetic_total", f"net {net} + VAT {vat} = {net + vat}, but the total is {gross}"))
    lines = [li for li in ex.line_items if li.net_amount is not None]
    if lines:
        s = sum((money(li.net_amount) for li in lines), Decimal("0"))
        target = net if net is not None else (gross - vat if None not in (gross, vat) else None)
        vat_inclusive_lines = gross is not None and abs(s - gross) <= CENT  # e.g. till receipts
        if target is not None and abs(s - target) > CENT and not vat_inclusive_lines:
            out.append(flag("arithmetic_lines", f"line items add up to {s}, the net total is {target}"))
    for i, li in enumerate(ex.line_items):
        q, u, n = money(li.quantity), money(li.unit_price), money(li.net_amount)
        if None not in (q, u, n) and abs(_round(Decimal(str(li.quantity)) * Decimal(str(li.unit_price))) - n) > CENT:
            out.append(flag("arithmetic_line_item", f"line {i + 1}: {li.quantity} x {u} is not {n}"))
    if vat is not None and net is not None:
        header_rate = ex.vat_rates[0] if len(ex.vat_rates) == 1 else None
        candidates = []
        if header_rate is not None:
            candidates.append(_round(net * Decimal(str(header_rate)) / 100))
        if lines and all(li.vat_rate is not None or header_rate is not None for li in lines):
            candidates.append(sum((_round(money(li.net_amount) * Decimal(str(li.vat_rate if li.vat_rate is not None
                                                                              else header_rate)) / 100)
                                   for li in lines), Decimal("0")))
        tol = CENT * max(1, len(lines))
        if candidates and not any(abs(c - vat) <= tol for c in candidates):
            out.append(flag("arithmetic_vat_rate", f"VAT {vat} does not match the rate(s) shown "
                                                   f"(expected about {candidates[0]})"))
    return out


def dates(ex: Extraction, today: dt.date) -> list[dict]:
    out = []
    parsed = {}
    for f in ("issue_date", "due_date"):
        v = getattr(ex, f)
        if v:
            d = parse_date(v)
            if d is None:
                out.append(flag("date_invalid", f"{f} {v!r} is not a valid date"))
            else:
                parsed[f] = d
    if parsed.get("issue_date") and parsed["issue_date"] > today:
        out.append(flag("date_in_future", f"issue date {parsed['issue_date']} is after today ({today})"))
    if parsed.get("issue_date") and parsed.get("due_date") and parsed["due_date"] < parsed["issue_date"]:
        out.append(flag("due_before_issue", f"due date {parsed['due_date']} is before issue date {parsed['issue_date']}"))
    return out


def currency(ex: Extraction, iso_codes: set[str]) -> list[dict]:
    if not ex.currency:
        return []
    code = ex.currency.strip().upper()
    if code not in iso_codes:
        return [flag("currency_invalid", f"{ex.currency!r} is not an ISO 4217 code")]
    if code != "GBP":
        return [flag("foreign_currency", f"{code} document: check the exchange rate and VAT treatment before "
                                         "export (policy of this tool, not a tax rule)")]
    return []


def vat_number(ex: Extraction) -> list[dict]:
    if ex.supplier_vat_number and not rules.vat_number_format_ok(ex.supplier_vat_number):
        return [flag("vat_number_format", f"{ex.supplier_vat_number!r} is not 9 digits with an optional GB prefix "
                                          "(HMRC format; the check digits are not tested)")]
    return []


def _present(ex: Extraction, key: str) -> bool:
    lines = ex.line_items
    single_rate = len(ex.vat_rates) == 1
    if key == "line_description":
        return any(li.description for li in lines)
    if key == "line_quantity":
        return bool(lines) and all(li.quantity is not None for li in lines)
    if key == "line_vat_rate":
        return bool(lines) and (single_rate or all(li.vat_rate is not None for li in lines))
    if key == "line_net":
        return bool(lines) and all(li.net_amount is not None for li in lines)
    if key == "vat_rate":
        return bool(ex.vat_rates) or (bool(lines) and all(li.vat_rate is not None for li in lines))
    v = getattr(ex, key)
    return v is not None and v != ""


def vat_completeness(doc_type: str, ex: Extraction) -> list[dict]:
    """Required VAT-invoice details (VAT Notice 700 / 700/21), checked only when VAT is charged."""
    vat, gross = money(ex.vat_total), money(ex.gross_total)
    if doc_type not in BILL_TYPES or vat is None or vat <= 0:
        return []
    if doc_type == "credit_note":
        kind, req = "credit note", rules.CREDIT_NOTE_FIELDS
    elif gross is not None and gross <= rules.SIMPLIFIED_INVOICE_MAX_GROSS:
        kind, req = "simplified VAT invoice (total GBP 250 or less incl. VAT)", rules.SIMPLIFIED_INVOICE_FIELDS
    else:
        kind, req = "full VAT invoice (total over GBP 250 incl. VAT)", rules.FULL_INVOICE_FIELDS
    missing = [desc for key, desc in req.items() if not _present(ex, key)]
    if missing:
        return [flag("vat_invoice_incomplete", f"a {kind} must show: " + "; ".join(missing))]
    return []


def required(doc_type: str, ex: Extraction) -> list[dict]:
    if doc_type not in BILL_TYPES:
        return []
    missing = [f for f in REQUIRED_FIELDS if not _present(ex, f)]
    return [flag("missing_field", "missing: " + ", ".join(missing))] if missing else []


def ground_statement(st, text: DocText) -> dict[str, str]:
    """Statement lines are extracted values too: each reference and amount must be printed on the statement."""
    g: dict[str, str] = {}
    if st.supplier_name:
        g["supplier_name"] = "ok" if _string_grounded(st.supplier_name, text) else "not_found"
    for i, ln in enumerate(st.lines):
        if ln.document_number:
            g[f"stmt[{i}].document_number"] = "ok" if alnum(ln.document_number) in text.alnum else "not_found"
        if ln.amount is not None:
            g[f"stmt[{i}].amount"] = "ok" if _amount_grounded(ln.amount, text) else "not_found"
    return g


def statement_grounding_flags(g: dict[str, str], st) -> list[dict]:
    out = []
    for k, v in g.items():
        if v == "not_found":
            if k.startswith("stmt["):
                i = int(k[5:k.index("]")])
                val = getattr(st.lines[i], k.split(".", 1)[1])
            else:
                val = getattr(st, k)
            out.append(flag("grounding", f"statement {k} = {repr(val)[:60]} not found in the document text "
                                         "(not used for missing-document claims)"))
    return out


def grounding_flags(g: dict[str, str], ex: Extraction) -> list[dict]:
    out = []
    for k, v in g.items():
        if v == "not_found":
            val = _value_for(ex, k)
            shown = "(bank detail hidden)" if k.startswith("bank_") else repr(val)[:80]
            out.append(flag("grounding", f"{k} = {shown} not found in the document text"))
    return out


def _value_for(ex: Extraction, key: str):
    if key.startswith("line["):
        i = int(key[5:key.index("]")])
        return getattr(ex.line_items[i], key.split(".", 1)[1])
    if key.startswith("vat_rate["):
        return ex.vat_rates[int(key[9:-1])]
    return getattr(ex, key)


def low_ocr(pages: list[PageText]) -> list[dict]:
    bad = [p for p in pages if p.source == "ocr" and p.confidence < LOW_OCR_CONFIDENCE]
    return [flag("low_ocr_confidence", f"page {p.page}: OCR confidence {p.confidence:.2f} < {LOW_OCR_CONFIDENCE}")
            for p in bad]


def extraction_confidence(pages: list[PageText], g: dict[str, str]) -> float:
    """Defined in code: lowest page text confidence x share of extracted values found in the text.

    Never uses the model's own rating. Text-layer pages count as confidence 1.0.
    """
    page_conf = min((p.confidence for p in pages), default=0.0)
    if not g:
        return 0.0
    grounded = sum(v in ("ok", "computed") for v in g.values()) / len(g)
    return round(page_conf * grounded, 4)


def required_grounded(ex: Extraction, g: dict[str, str]) -> bool:
    return all(_present(ex, f) and g.get(f) in ("ok", "computed") for f in REQUIRED_FIELDS)
