import datetime as dt
from decimal import Decimal

import pytest

from conftest import TODAY, invoice_extraction, invoice_text
from invoice_intake import checks, rules
from invoice_intake.documents import PageText
from invoice_intake.graph import route_document
from invoice_intake.normalize import dates_in, numbers_in, parse_amount_token
from invoice_intake.rules import iso4217_codes
from invoice_intake.schema import Extraction

ISO = iso4217_codes()


def X(**over):
    return Extraction.model_validate(invoice_extraction(**over))


def T(text):
    return checks.DocText.of([PageText(page=1, text=text, source="text_layer", confidence=1.0)])


def names(flags):
    return [f["name"] for f in flags]


# ------------------------------------------------------------------ normalisation

@pytest.mark.parametrize("tok,val", [("1,234.50", "1234.50"), ("1.234,50", "1234.50"), ("1 234.50", "1234.50"),
                                     ("(12.00)", "12.00"), ("-12.00", "12.00"), ("£249.14", "249.14"),
                                     ("12,50", "12.50"), ("1500", "1500.00")])
def test_amount_tokens(tok, val):
    assert parse_amount_token(tok) == Decimal(val)


def test_numbers_in_handles_both_thousands_styles():
    nums = numbers_in("Total £1,500.00 / Total TTC 1.299,39 € / qty 50")
    assert {Decimal("1500.00"), Decimal("1299.39"), Decimal("50.00")} <= nums


def test_compact_dates_from_the_model():
    from invoice_intake.normalize import parse_date
    assert parse_date("20102026") == dt.date(2026, 10, 20) and parse_date("20261020") == dt.date(2026, 10, 20)
    assert parse_date("07 Oct 2026") == dt.date(2026, 10, 7)


def test_ocr_rows_are_deskewed():
    from invoice_intake.documents import ocr_page
    import math

    class Tilted:  # two rows, each tilted 3 degrees: the right-hand value sits lower than the label
        def lines(self, image):
            a, t = 0.0524, math.tan(0.0524)
            return [("TOTAL", 0.99, 10, 100, 20, 40, a), ("44.02", 0.99, 500, 100 + 490 * t, 20, 530, a),
                    ("CARD", 0.99, 10, 126, 20, 40, a), ("44.02", 0.99, 500, 126 + 490 * t, 20, 530, a)]
    p = ocr_page(Tilted(), None, 1)
    assert p.text.splitlines() == ["TOTAL   44.02", "CARD   44.02"]


def test_dates_in_all_formats():
    d = dt.date(2026, 11, 3)
    for text in ("03/11/2026", "3 Nov 2026", "3rd November 2026", "2026-11-03", "03.11.2026", "03-11-2026",
                 "November 3, 2026"):
        assert d in dates_in(text), text


def test_grounding_normalises_whitespace_separators_and_dates():
    text = invoice_text("k").replace("GB 123 4567 81", "GB123  4567\n81")
    g = checks.ground(X(), T(text))
    assert all(v in ("ok", "computed") for v in g.values()), {k: v for k, v in g.items() if v == "not_found"}


def test_grounding_flags_values_not_in_text():
    g = checks.ground(X(gross_total=1600.0, document_number="BFM-9999", issue_date="2026-11-04"), T(invoice_text("k")))
    assert g["gross_total"] == "not_found" and g["document_number"] == "not_found" and g["issue_date"] == "not_found"
    flags = checks.grounding_flags(g, X())
    assert names(flags).count("grounding") == 3


def test_grounding_allows_computed_totals_and_line_amounts():
    text = invoice_text("k").replace("Net total 1,250.00\n", "")  # net only printed in the line
    g = checks.ground(X(), T(text.replace("Strong white flour 16kg 50 25.00 20% 1,250.00", "Strong white flour 16kg 50 25.00 20%")))
    assert g["net_total"] == "computed"
    assert g["line[0].net_amount"] == "computed"


def test_grounding_hides_bank_values_in_reasons():
    g = {"bank_account_number": "not_found"}
    f = checks.grounding_flags(g, X(bank_account_number="99999999"))
    assert "99999999" not in f[0]["reason"]


def test_currency_symbol_grounds_gbp_but_not_other_text():
    assert checks.ground(X(), T("Total £5.00"))["currency"] == "ok"
    assert checks.ground(X(currency="EUR"), T("Neuro Total £5.00"))["currency"] == "not_found"


# ------------------------------------------------------------------ arithmetic

def test_arithmetic_ok_on_clean_invoice():
    assert checks.arithmetic(X()) == []


def test_net_plus_vat_not_total():
    assert "arithmetic_total" in names(checks.arithmetic(X(gross_total=1510.0)))


def test_total_within_one_penny_passes():
    assert checks.arithmetic(X(gross_total=1500.01)) == []


def test_lines_do_not_sum_to_net():
    assert "arithmetic_lines" in names(checks.arithmetic(X(net_total=1300.0, gross_total=1550.0, vat_total=250.0)))


def test_vat_inclusive_receipt_lines_are_accepted():
    ex = X(net_total=10.0, vat_total=2.0, gross_total=12.0,
           line_items=[{"description": "Tray", "quantity": 1, "unit_price": 12.0, "net_amount": 12.0}])
    assert "arithmetic_lines" not in names(checks.arithmetic(ex))


def test_line_quantity_times_price():
    ex = X(line_items=[{"description": "Flour", "quantity": 50, "unit_price": 24.0, "net_amount": 1250.0, "vat_rate": 20}])
    assert "arithmetic_line_item" in names(checks.arithmetic(ex))


def test_vat_does_not_match_rate():
    assert "arithmetic_vat_rate" in names(checks.arithmetic(X(vat_total=200.0, gross_total=1450.0)))


def test_vat_rounding_per_line_is_tolerated():
    lines = [{"description": f"x{i}", "quantity": 1, "unit_price": 0.33, "net_amount": 0.33, "vat_rate": 20} for i in range(3)]
    ex = X(net_total=0.99, vat_total=0.21, gross_total=1.20, line_items=lines)  # 3 x round(0.066) = 0.21
    assert "arithmetic_vat_rate" not in names(checks.arithmetic(ex))


def test_mixed_rates_use_line_rates():
    lines = [{"description": "Butter", "quantity": 1, "unit_price": 100.0, "net_amount": 100.0, "vat_rate": 0},
             {"description": "Delivery", "quantity": 1, "unit_price": 10.0, "net_amount": 10.0, "vat_rate": 20}]
    ex = X(net_total=110.0, vat_total=2.0, gross_total=112.0, vat_rates=[0, 20], line_items=lines)
    assert checks.arithmetic(ex) == []


# ------------------------------------------------------------------ dates, currency, VAT number

def test_dates_checks():
    assert names(checks.dates(X(issue_date="31/02/2026"), TODAY)) == ["date_invalid"]
    assert names(checks.dates(X(issue_date="2026-11-11", due_date="2026-12-11"), TODAY)) == ["date_in_future"]
    assert names(checks.dates(X(due_date="2026-11-01"), TODAY)) == ["due_before_issue"]
    assert checks.dates(X(issue_date="2026-11-10", due_date="2026-11-10"), TODAY) == []


def test_currency_checks():
    assert checks.currency(X(), ISO) == []
    assert names(checks.currency(X(currency="XYZ1"), ISO)) == ["currency_invalid"]
    assert names(checks.currency(X(currency="UKP"), ISO)) == ["currency_invalid"]
    assert names(checks.currency(X(currency="eur"), ISO)) == ["foreign_currency"]


@pytest.mark.parametrize("v,ok", [("GB 123 4567 81", True), ("123456781", True), ("gb123456781", True),
                                  ("GB12345678", False), ("GB1234567890", False), ("FR12345678901", False),
                                  ("GB 123 4567 8X", False)])
def test_vat_number_format(v, ok):
    assert rules.vat_number_format_ok(v) is ok
    assert (checks.vat_number(X(supplier_vat_number=v)) == []) is ok


# ------------------------------------------------------------------ VAT invoice completeness

def test_full_invoice_complete():
    assert checks.vat_completeness("invoice", X()) == []


def test_full_invoice_missing_vat_number_and_customer():
    f = checks.vat_completeness("invoice", X(supplier_vat_number=None, customer_name=None))
    assert names(f) == ["vat_invoice_incomplete"]
    assert "VAT registration number" in f[0]["reason"] and "customer's name" in f[0]["reason"]


def test_simplified_threshold_is_250_including_vat():
    small = dict(net_total=208.33, vat_total=41.67, gross_total=250.0, customer_name=None, document_number=None,
                 line_items=[{"description": "Trays", "quantity": None, "unit_price": None, "net_amount": None}])
    assert checks.vat_completeness("receipt", X(**small)) == []  # simplified: no customer, no number needed
    over = {**small, "net_total": 208.34, "vat_total": 41.67, "gross_total": 250.01}
    assert names(checks.vat_completeness("receipt", X(**over))) == ["vat_invoice_incomplete"]


def test_no_vat_charged_means_no_vat_invoice_check():
    assert checks.vat_completeness("invoice", X(vat_total=0.0, supplier_vat_number=None)) == []


def test_credit_note_needs_original_invoice_number():
    f = checks.vat_completeness("credit_note", X(original_invoice_number=None))
    assert "original VAT invoice" in f[0]["reason"]
    assert checks.vat_completeness("credit_note", X(original_invoice_number="BFM-0999")) == []


def test_required_fields():
    assert names(checks.required("invoice", X(document_number=None))) == ["missing_field"]
    assert checks.required("statement", X(document_number=None)) == []


# ------------------------------------------------------------------ OCR confidence and routing

def test_low_ocr_confidence_only_for_ocr_pages():
    pages = [PageText(1, "x", "ocr", 0.6), PageText(2, "y", "text_layer", 1.0), PageText(3, "z", "ocr", 0.95)]
    assert names(checks.low_ocr(pages)) == ["low_ocr_confidence"]


def test_extraction_confidence_is_code_defined():
    pages = [PageText(1, "x", "ocr", 0.9)]
    assert checks.extraction_confidence(pages, {"a": "ok", "b": "computed", "c": "not_found", "d": "ok"}) == 0.675
    assert checks.extraction_confidence(pages, {}) == 0.0


@pytest.mark.parametrize("doc_type,flags,req,conf,route", [
    ("invoice", [], True, 0.95, "auto"), ("receipt", [], True, 0.95, "auto"), ("credit_note", [], True, 0.95, "auto"),
    ("invoice", [{"name": "grounding"}], True, 0.95, "review"), ("invoice", [], False, 0.95, "review"),
    ("invoice", [], True, 0.89, "review"), ("statement", [], True, 1.0, "review"), ("other", [], True, 1.0, "review"),
    (None, [], True, 1.0, "review")])
def test_routing_rules(doc_type, flags, req, conf, route):
    assert route_document(doc_type, flags, req, conf, 0.9) == route
