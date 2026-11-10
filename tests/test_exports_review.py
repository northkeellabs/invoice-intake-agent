import csv
import datetime as dt
import io

import pytest
from pydantic import ValidationError

from conftest import CLIENT, CLIENT_ID, SUP, invoice_extraction
from invoice_intake.api import apply_review, straight_through
from invoice_intake.coding import Chart, decide
from invoice_intake.config import ROOT
from invoice_intake.exports import CHASER_MARK, bill_rows, chaser_text, public_doc, write_bills_csv
from invoice_intake.graph import build_batch_plan
from invoice_intake.rules import XERO_BILL_COLUMNS
from invoice_intake.schema import BatchPlan
from invoice_intake.store import Store

CHART = Chart.load(ROOT / "config" / "coa.yaml")


def doc(doc_id="d1", doc_type="invoice", status="approved", **ex):
    return {"id": doc_id, "batch_id": "b1", "file_name": f"{doc_id}.pdf", "doc_type": doc_type, "status": status,
            "route": "auto", "extraction": invoice_extraction(**ex), "account_code": "5000", "account_source": "rule",
            "supplier_key": "brackenfoldmillsdemo", "flags": [], "bank_hash": "h", "bank_last4": "4217",
            "confidence": 1.0, "edited": 0}


# ------------------------------------------------------------------ CSV export

def test_csv_columns_and_rows(tmp_path):
    p = tmp_path / "b.csv"
    n = write_bills_csv(p, [doc()], CHART)
    rows = list(csv.DictReader(io.StringIO(p.read_text(encoding="utf-8"))))
    assert n == 1 and list(rows[0]) == XERO_BILL_COLUMNS
    r = rows[0]
    assert r["ContactName"] == SUP and r["InvoiceNumber"] == "BFM-1001" and r["InvoiceDate"] == "03/11/2026"
    assert r["DueDate"] == "03/12/2026" and r["Quantity"] == "50" and r["UnitAmount"] == "25.00"
    assert r["AccountCode"] == "5000" and r["TaxType"] == "20% (VAT on Expenses)" and r["Currency"] == "GBP"
    assert "00004217" not in p.read_text(encoding="utf-8")


def test_credit_note_amounts_are_negative():
    r = bill_rows(doc(doc_type="credit_note", due_date=None), CHART)[0]
    assert r["UnitAmount"] == "-25.00" and r["DueDate"] == ""


def test_vat_inclusive_lines_export_one_net_row():
    d = doc(doc_type="receipt", net_total=10.0, vat_total=2.0, gross_total=12.0, due_date=None,
            line_items=[{"description": "Tray", "quantity": 1, "unit_price": 12.0, "net_amount": 12.0}])
    rows = bill_rows(d, CHART)
    assert len(rows) == 1 and rows[0]["UnitAmount"] == "10.00" and rows[0]["Description"] == "Receipt BFM-1001"


def test_zero_rated_and_no_vat_tax_types():
    d = doc(vat_total=0.0, gross_total=1250.0, vat_rates=[0],
            line_items=[{"description": "Flour", "quantity": 50, "unit_price": 25.0, "net_amount": 1250.0, "vat_rate": 0}])
    assert bill_rows(d, CHART)[0]["TaxType"] == "No VAT"


def test_json_export_masks_bank_details():
    pd = public_doc(doc())
    assert "bank_sort_code" not in pd["fields"] and "bank_account_number" not in pd["fields"]
    assert pd["bank_details"] == "account ending 4217"


def test_chaser_is_marked_not_sent():
    t = chaser_text(CLIENT, [{"supplier": SUP, "document_number": "BFM-0950", "date": "2026-10-01", "amount": "99.00",
                              "statement_date": "2026-11-07"}])
    assert t.startswith(CHASER_MARK) and t.count(CHASER_MARK) == 2 and "01/10/2026" in t


# ------------------------------------------------------------------ coding

def test_rule_beats_llm_and_llm_code_is_validated():
    assert decide({"account_code": "6100"}, "5000", CHART).code == "6100"
    assert decide(None, "6200", CHART).source == "llm"
    bad = decide(None, "0", CHART)
    assert bad.code is None and not bad.valid


# ------------------------------------------------------------------ review UI logic

def store_with(tmp_path, *docs):
    s = Store(tmp_path / "s.sqlite")
    for d in docs:
        s.add_document(d)
    return s


def test_edit_records_before_after_and_approve_creates_rule(tmp_path):
    s = store_with(tmp_path, doc(status="review", account_code=None) | {"account_code": None})
    apply_review(s, s.get("d1"), {"gross_total": "1,500.00", "document_number": "BFM-1002", "account_code": "5000"},
                 "approve", CHART)
    d = s.get("d1")
    assert d["status"] == "approved" and d["edited"] == 1 and d["extraction"]["document_number"] == "BFM-1002"
    edits = {e["field"]: (e["before"], e["after"]) for e in s.edits("d1")}
    assert edits["document_number"] == ('"BFM-1001"', '"BFM-1002"') and "gross_total" not in edits
    assert s.supplier("brackenfoldmillsdemo")["account_code"] == "5000"
    assert straight_through(s) == {"approved_bills": 1, "approved_without_edit": 0, "straight_through_rate": 0.0}


def test_reject_needs_reason(tmp_path):
    s = store_with(tmp_path, doc(status="review"))
    with pytest.raises(ValueError):
        apply_review(s, s.get("d1"), {}, "reject", CHART, reason=" ")
    apply_review(s, s.get("d1"), {}, "reject", CHART, reason="not ours")
    assert s.get("d1")["status"] == "rejected" and s.suppliers() == []


def test_statement_cannot_be_approved_and_invalid_code_refused(tmp_path):
    s = store_with(tmp_path, doc(doc_type="statement", status="review"), doc("d2", status="review"))
    with pytest.raises(ValueError):
        apply_review(s, s.get("d1"), {}, "approve", CHART)
    with pytest.raises(ValueError):
        apply_review(s, s.get("d2"), {"account_code": "0"}, "approve", CHART)
    with pytest.raises(ValueError):
        apply_review(s, s.get("d2"), {"issue_date": "not a date"}, "save", CHART)


def test_approving_new_bank_details_does_not_change_supplier_record(tmp_path):
    s = store_with(tmp_path, doc(status="review") | {"bank_hash": "NEW", "bank_last4": "9133"})
    s.create_supplier_rule("brackenfoldmillsdemo", SUP, None, "5000", "OLD", "4217", "seed")
    apply_review(s, s.get("d1"), {}, "approve", CHART)
    assert s.supplier("brackenfoldmillsdemo")["bank_hash"] == "OLD"


# ------------------------------------------------------------------ plan validation and caps

def plan_for(docs, recon=(), contact=CLIENT_ID):
    return build_batch_plan("b1", CLIENT, contact, list(docs), list(recon), dt.date(2026, 11, 10))


MISSING = [{"missing": [{"supplier": SUP, "document_number": "BFM-0950"}], "amount_mismatch": [], "matched": []}]


def test_plan_shapes():
    p = plan_for([doc(status="auto_eligible")], MISSING)
    assert [a.action for a in p.crm.actions] == ["add_note", "create_task"] and p.approve_doc_ids == ["d1"]
    assert plan_for([doc(status="auto_eligible")]).crm.actions[0].action == "add_note"
    assert plan_for([doc()], contact=None).crm.actions == []


def bad(p: BatchPlan, **change):
    data = p.model_dump(mode="json")
    data.update(change)
    with pytest.raises(ValidationError):
        BatchPlan.model_validate(data)


def test_plan_validation_rejects_bad_shapes():
    p = plan_for([doc(status="auto_eligible")], MISSING)
    note, task = [a.model_dump(mode="json") for a in p.crm.actions]
    bad(p, crm={"actions": [note, task, note]})                                    # too many actions
    bad(p, crm={"actions": [task, note]})                                          # task first
    bad(p, crm={"actions": [note, {**task, "name": "Pay supplier today"}]})        # task name fixed
    bad(p, crm={"actions": [note, {**task, "name": "Chase 2 missing documents"}]})  # N must match
    bad(p, crm={"actions": [note, {**task, "contact_id": "777"}]})                 # one record only
    bad(p, crm={"actions": [{**note, "note": "x" * 2001}, task]})                  # note length cap
    bad(p, crm={"actions": [{**note, "note": "pay sort code 00-00-77 account 00009133"}, task]})  # bank details
    bad(p, crm={"actions": [{**note, "extra": 1}, task]})                          # unknown field
    bad(p, missing_count=0)                                                        # task without missing docs
    bad(p, approve_doc_ids=["d1", "zz"])                                           # approved but not exported
    bad(p, crm={"actions": [note]})                                                # missing docs need the task


def test_check_plan_targets():
    from invoice_intake.graph import check_plan_targets
    p = plan_for([doc(status="auto_eligible")], MISSING)
    check_plan_targets(p, CLIENT_ID, {"d1"})
    with pytest.raises(ValueError):
        check_plan_targets(p, "other", {"d1"})
    with pytest.raises(ValueError):
        check_plan_targets(p, CLIENT_ID, {"d9"})
