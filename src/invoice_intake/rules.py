"""Tax and export rules used in code, each with the official source it was checked against (accessed 2026-11-10).

Only rules confirmed on an official page are implemented. Not implemented, because no official source
publishes them: the UK VAT number check-digit algorithm (HMRC only says the last two digits come from
"a standard formula"), and any format beyond "9 numbers, sometimes with GB at the start".
"""
from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path

from .config import ROOT

SOURCES = {
    "vat_invoice_content": "https://www.gov.uk/guidance/vat-guide-notice-700 (section 16.3.1) and "
                           "https://www.gov.uk/guidance/record-keeping-for-vat-notice-70021 (section 4.1)",
    "simplified_invoice": "https://www.gov.uk/guidance/record-keeping-for-vat-notice-70021 (section 4.5) and "
                          "https://www.gov.uk/guidance/vat-guide-notice-700 (section 16.6.1)",
    "credit_note_content": "https://www.gov.uk/guidance/vat-guide-notice-700 (section 18.2.3)",
    "vat_number_format": "https://design.tax.service.gov.uk/hmrc-design-patterns/vat-registration-number/ and "
                         "https://www.gov.uk/hmrc-internal-manuals/enquiry-manual/em3058",
    "xero_bill_import": "https://central.xero.com/s/article/Import-bills-and-credit-notes",
    "iso4217": "https://www.six-group.com/dam/download/financial-information/data-center/iso-currrency/lists/list-one.xml",
}

# VAT Notice 700/21 s4.5: a simplified invoice may be used for supplies of "£250 or less (including VAT)".
SIMPLIFIED_INVOICE_MAX_GROSS = Decimal("250.00")

# Fields we extract that map to the VAT Notice 700 s16.3.1 list for a full VAT invoice. Listed items we do
# not check: the tax point when it differs from the issue date, the rate of any cash discount offered
# ("only if offered"), and the unit price (s16.3.2 allows it to be left out in some cases).
FULL_INVOICE_FIELDS = {
    "document_number": "a sequential number which uniquely identifies the document",
    "issue_date": "the time of supply / date of issue",
    "supplier_name": "the supplier's name",
    "supplier_address": "the supplier's address",
    "supplier_vat_number": "the supplier's VAT registration number",
    "customer_name": "the customer's name",
    "line_description": "a description of the goods or services",
    "line_quantity": "the quantity of goods or extent of services per line",
    "line_vat_rate": "the rate of VAT per line",
    "line_net": "the amount payable excluding VAT per line",
    "net_total": "the total amount payable excluding VAT",
    "vat_total": "the total amount of VAT chargeable",
}

# Notice 700 s16.6.1 "The simplified invoice must include": name, address and VAT number; time of supply;
# a description; for each VAT rate the total including VAT and the rate.
SIMPLIFIED_INVOICE_FIELDS = {
    "supplier_name": "the supplier's name",
    "supplier_address": "the supplier's address",
    "supplier_vat_number": "the supplier's VAT registration number",
    "issue_date": "the time of supply",
    "line_description": "a description of the goods or services",
    "gross_total": "the total amount payable including VAT",
    "vat_rate": "the VAT rate",
}

# Notice 700 s18.2.3, valid credit notes ("show clearly all the following details").
CREDIT_NOTE_FIELDS = {
    "document_number": "the identifying number",
    "issue_date": "the date of issue",
    "supplier_name": "the supplier's name",
    "supplier_address": "the supplier's address",
    "supplier_vat_number": "the supplier's VAT registration number",
    "customer_name": "the customer's name",
    "line_description": "a description of what is credited",
    "line_quantity": "the quantity per description",
    "line_net": "the amount per description",
    "net_total": "the total amount credited, excluding VAT",
    "vat_rate": "the rate of VAT credited",
    "vat_total": "the amount of VAT credited",
    "original_invoice_number": "the number of the original VAT invoice",
}

_VAT_RE = re.compile(r"^(?:GB)?\d{9}$")


def vat_number_format_ok(v: str) -> bool:
    """HMRC design pattern: '9 numbers, sometimes with GB at the start'; spaces are removed before validating.

    No check-digit test: HMRC does not publish the formula (EM3058 only says one exists).
    """
    return bool(_VAT_RE.match(re.sub(r"\s+", "", v or "").upper()))


# Xero Central, "Import bills and credit notes": columns named on that page. The page names
# EmailAddress, POAddressLine1-4, POCity, PORegion, POPostalCode, POCountry, InventoryItemCode, Description,
# TrackingName1/2, TrackingOption1/2, Currency, UnitAmount, TaxType and the optional TaxAmount by their
# header. Contact Name, Invoice Number, Invoice Date, Due Date, Quantity and Account Code are described in
# words; the header spellings used here (no spaces) follow the same style but were not visible on the page,
# and the column order is ours. Check them against the template you download from Xero before importing.
XERO_BILL_COLUMNS = [
    "ContactName", "EmailAddress", "POAddressLine1", "POAddressLine2", "POAddressLine3", "POAddressLine4",
    "POCity", "PORegion", "POPostalCode", "POCountry", "InvoiceNumber", "InvoiceDate", "DueDate",
    "InventoryItemCode", "Description", "Quantity", "UnitAmount", "AccountCode", "TaxType", "TaxAmount",
    "TrackingName1", "TrackingOption1", "TrackingName2", "TrackingOption2", "Currency",
]
XERO_DATE_FORMAT = "%d/%m/%Y"  # "Use the format DD/MM/YYYY."
XERO_MAX_ROWS_HINT = 500       # "If you're importing more than 500 items, we recommend you split up the file."


def iso4217_codes(path: Path | None = None) -> set[str]:
    p = path or ROOT / "config" / "iso4217.txt"
    return {ln.strip() for ln in p.read_text(encoding="ascii").splitlines() if ln.strip() and not ln.startswith("#")}
