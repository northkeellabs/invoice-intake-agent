"""Output files (out/<batch_id>/): Xero-format bills CSV, JSON export, Markdown exception report, chaser draft.

These are files only. The agent never sends email, never pays anything and never writes to an accounting
system: a person imports the CSV and sends (or bins) the chaser.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
from decimal import Decimal
from pathlib import Path

from .coding import Chart
from .normalize import CENT, money, parse_date
from .reconcile import fmt_date
from .rules import XERO_BILL_COLUMNS, XERO_DATE_FORMAT
from .schema import Extraction

CHASER_MARK = "DRAFT - NOT SENT"


def _xdate(s: str | None) -> str:
    d = parse_date(s)
    return d.strftime(XERO_DATE_FORMAT) if d else ""


def bill_rows(doc: dict, chart: Chart) -> list[dict]:
    """One CSV row per line item (amounts tax exclusive). Credit notes get negative UnitAmount, as Xero asks.

    If the printed lines are VAT-inclusive (they add up to the gross, not the net) or there are no lines,
    one summary row per document is written with the net amount, so the import totals stay right.
    """
    ex = Extraction.model_validate(doc["extraction"] or {})
    sign = Decimal("-1") if doc["doc_type"] == "credit_note" else Decimal("1")
    vat = money(ex.vat_total) or Decimal("0")
    net = money(ex.net_total)
    if net is None and money(ex.gross_total) is not None:
        net = money(ex.gross_total) - vat
    base = {c: "" for c in XERO_BILL_COLUMNS}
    base.update({"ContactName": ex.supplier_name or "", "InvoiceNumber": ex.document_number or "",
                 "InvoiceDate": _xdate(ex.issue_date), "DueDate": _xdate(ex.due_date),
                 "AccountCode": doc.get("account_code") or "", "Currency": (ex.currency or "").upper()})
    lines = [li for li in ex.line_items if li.net_amount is not None]
    header_rate = ex.vat_rates[0] if len(ex.vat_rates) == 1 else None
    line_sum = sum((money(li.net_amount) for li in lines), Decimal("0"))
    rows = []
    if lines and net is not None and abs(line_sum - net) <= CENT:
        for li in lines:
            rate = li.vat_rate if li.vat_rate is not None else header_rate
            qty = Decimal(str(li.quantity)) if li.quantity else Decimal("1")
            unit = money(li.unit_price) if li.unit_price is not None and li.quantity else money(li.net_amount)
            rows.append({**base, "Description": (li.description or "")[:400], "Quantity": f"{qty.normalize():f}",
                         "UnitAmount": f"{(sign * unit):.2f}", "TaxType": chart.tax_type(rate, vat > 0)})
    elif net is not None:
        label = {"credit_note": "Credit note", "receipt": "Receipt"}.get(doc["doc_type"], "Invoice")
        rows.append({**base, "Description": f"{label} {ex.document_number or ''}".strip(), "Quantity": "1",
                     "UnitAmount": f"{(sign * net):.2f}", "TaxType": chart.tax_type(header_rate, vat > 0)})
    return rows


def write_bills_csv(path: Path, docs: list[dict], chart: Chart) -> int:
    rows = [r for d in docs for r in bill_rows(d, chart)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=XERO_BILL_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def public_doc(d: dict) -> dict:
    """Document as exported to JSON / shown in reports: bank details masked, no document text."""
    ex = dict(d.get("extraction") or {})
    for k in ("bank_sort_code", "bank_account_number", "bank_iban"):
        ex.pop(k, None)
    return {"id": d["id"], "file": d["file_name"], "type": d["doc_type"], "status": d["status"], "route": d["route"],
            "account_code": d.get("account_code"), "account_source": d.get("account_source"),
            "confidence": d.get("confidence"), "fields": ex,
            "bank_details": f"account ending {d['bank_last4']}" if d.get("bank_last4") else None,
            "flags": [f["name"] for f in d.get("flags") or []], "edited": bool(d.get("edited"))}


def write_json(path: Path, batch_id: str, docs: list[dict]) -> None:
    path.write_text(json.dumps({"batch_id": batch_id, "documents": [public_doc(d) for d in docs]}, indent=1,
                               default=str), encoding="utf-8")


def chaser_text(client: str, missing: list[dict]) -> str:
    lines = [CHASER_MARK, "", f"To: {client}", "Subject: Missing supplier invoices for your books", "", "Hello,", "",
             "While reconciling your supplier statements we could not find the following documents in what you "
             "sent us. Could you send us a copy of each (a photo or PDF is fine)?", ""]
    for m in missing:
        amt = f", {m['amount']}" if m.get("amount") else ""
        lines.append(f"- {m['supplier']}: {m['document_number']} dated {fmt_date(m.get('date'))}{amt} "
                     f"(on the statement dated {fmt_date(m.get('statement_date'))})")
    lines += ["", "Thank you.", "", "[Your name]", "",
              f"{CHASER_MARK}: prepared by invoice-intake-agent for a person to review and send. Nothing was sent."]
    return "\n".join(lines) + "\n"


def exception_report(batch_id: str, client: str, docs: list[dict], recon: list[dict], chasers: list[str],
                     status: str, live: bool, today: dt.date) -> str:
    def row(d):
        return f"| {d['file_name']} | {d['doc_type'] or '-'} | {d['status']} | " + \
            "; ".join(f"{f['name']}: {f['reason']}" for f in d.get("flags") or []).replace("|", "/") + " |"

    by = lambda st: [d for d in docs if d["status"] == st]
    flagged = lambda name: [d for d in docs if any(f["name"] == name for f in d.get("flags") or [])]
    missing = [m for r in recon for m in r["missing"]]
    mism = [m for r in recon for m in r["amount_mismatch"]]
    out = [f"# Exception report: {batch_id}", "",
           f"Client: {client}. Generated {today.isoformat()}. Batch status: {status}. "
           f"Mode: {'live' if live else 'dry run (no CRM writes)'}.",
           "All documents in this demo are fictional. Bank details are never shown in full.", "",
           "## Summary", "",
           f"- Documents: {len(docs)}",
           f"- Approved: {len(by('approved'))}; in review queue: {len(by('review'))}; auto-approve eligible, "
           f"not yet approved: {len(by('auto_eligible'))}; rejected: {len(by('rejected'))}; invalid files: "
           f"{len(by('invalid'))}",
           f"- Missing documents from statements: {len(missing)}; amount mismatches: {len(mism)}",
           f"- Chasers drafted (not sent): {len(chasers)}", ""]
    sections = [("In review", by("review")), ("Rejected", by("rejected")), ("Invalid files", by("invalid")),
                ("Possible duplicates", flagged("duplicate")), ("Bank-detail changes", flagged("bank_details_changed")),
                ("Instructions found in document", flagged("instructions_found"))]
    for title, items in sections:
        out += [f"## {title}", ""]
        if items:
            out += ["| file | type | status | flags |", "|---|---|---|---|"] + [row(d) for d in items]
        else:
            out.append("None.")
        out.append("")
    failed = [d for d in docs if any(f["name"] not in ("not_a_bill",) for f in d.get("flags") or [])]
    out += ["## Failed checks (all flags)", ""]
    out += (["| file | type | status | flags |", "|---|---|---|---|"] + [row(d) for d in failed]) if failed else ["None."]
    out += ["", "## Missing documents from statements", ""]
    if missing:
        out += [f"- {m['supplier']}: {m['document_number']} dated {fmt_date(m.get('date'))}, {m.get('amount')}"
                for m in missing]
    else:
        out.append("None.")
    unver = [m for r in recon for m in r.get("unverified", [])]
    if unver:
        out += ["", "## Statement lines to check by hand", "",
                "The extracted reference is not printed on the statement, so it was not used to claim a missing document:"]
        out += [f"- {m['supplier']}: {m['document_number']} ({m.get('amount')})" for m in unver]
    if mism:
        out += ["", "## Statement amount mismatches", ""]
        out += [f"- {m['supplier']}: {m['document_number']} statement {m['amount']}, document {m['document_amount']}"
                for m in mism]
    out += ["", "## Chasers drafted", ""] + ([f"- {c} ({CHASER_MARK})" for c in chasers] or ["None."])
    return "\n".join(out) + "\n"
