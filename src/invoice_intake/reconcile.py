"""Supplier statement reconciliation: every invoice listed on a statement must match a processed document
from that supplier (same number, same amount within 0.01). Unmatched lines are missing documents."""
from __future__ import annotations

from .normalize import CENT, doc_number_key, money, parse_date
from .store import Store


def reconcile_statement(statement_doc: dict, store: Store) -> dict:
    st = statement_doc.get("statement") or {}
    supplier = st.get("supplier_name") or "(unknown supplier)"
    key = statement_doc.get("supplier_key") or ""
    known = store.documents_from_supplier(key) if key else []
    by_number = {d["doc_number_key"]: d for d in known if d.get("doc_number_key")}
    g = statement_doc.get("grounding") or {}
    matched, missing, mismatched, unverified = [], [], [], []
    for i, line in enumerate(st.get("lines") or []):
        num, amt = line.get("document_number"), money(line.get("amount"))
        if not num or line.get("kind") in ("payment", "other"):
            continue
        item = {"supplier": supplier, "document_number": num, "date": line.get("date"),
                "amount": str(amt) if amt is not None else None, "statement_doc": statement_doc["id"],
                "statement_date": st.get("statement_date")}
        if g.get(f"stmt[{i}].document_number") == "not_found":
            unverified.append(item)  # the model's reading is not printed on the statement: a person checks it
            continue
        d = by_number.get(doc_number_key(num))
        if d is None:
            missing.append(item)
        elif amt is not None and d.get("gross") and abs(money(d["gross"]) - amt) > CENT:
            mismatched.append({**item, "doc_id": d["id"], "document_amount": d["gross"]})
        else:
            matched.append({**item, "doc_id": d["id"]})
    return {"statement_doc": statement_doc["id"], "supplier": supplier, "supplier_key": key, "matched": matched,
            "missing": missing, "amount_mismatch": mismatched, "unverified": unverified}


def fmt_date(s: str | None) -> str:
    d = parse_date(s)
    return d.strftime("%d/%m/%Y") if d else (s or "?")
