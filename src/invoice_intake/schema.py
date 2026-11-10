"""Pydantic models: the LLM's structured outputs and the batch plan.

The LLM outputs are data to be checked, never instructions. The batch plan is the only thing the
execute node acts on. It is built in code and strict: two CRM action types, at most one of each,
capped field lengths, no unknown fields, a fixed task name, and no bank-like numbers in the note.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

DOC_TYPES = ("invoice", "receipt", "credit_note", "statement", "other")
DocType = Literal["invoice", "receipt", "credit_note", "statement", "other"]
BILL_TYPES = ("invoice", "receipt", "credit_note")
NOTE_MAX = 2000
TASK_RE = r"^Chase [1-9][0-9]{0,2} missing documents$"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class LlmOut(BaseModel):
    """LLM output: unknown keys are ignored (they are never read), every string is capped."""
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


Str200 = Annotated[str, Field(max_length=200)]
Str400 = Annotated[str, Field(max_length=400)]


# --------------------------------------------------------- LLM outputs

class Classification(LlmOut):
    document_type: DocType
    reason: str = Field(default="", max_length=300)


class LineItem(LlmOut):
    description: Str400 | None = None
    quantity: float | None = None
    unit_price: float | None = None
    net_amount: float | None = None
    vat_rate: float | None = Field(default=None, description="VAT rate in percent, e.g. 20")


class Extraction(LlmOut):
    supplier_name: Str200 | None = None
    supplier_address: Str400 | None = None
    supplier_vat_number: Str200 | None = None
    customer_name: Str200 | None = None
    document_number: Str200 | None = Field(default=None, description="invoice, receipt or credit note number")
    original_invoice_number: Str200 | None = Field(default=None, description="credit notes only: the invoice it refers to")
    issue_date: Str200 | None = Field(default=None, description="YYYY-MM-DD")
    due_date: Str200 | None = Field(default=None, description="YYYY-MM-DD")
    currency: Str200 | None = Field(default=None, description="ISO 4217 code, e.g. GBP")
    net_total: float | None = None
    vat_total: float | None = None
    gross_total: float | None = None
    vat_rates: list[float] = Field(default_factory=list, max_length=5)
    bank_sort_code: Str200 | None = None
    bank_account_number: Str200 | None = None
    bank_iban: Str200 | None = None
    line_items: list[LineItem] = Field(default_factory=list, max_length=60)


class StatementLine(LlmOut):
    kind: Literal["invoice", "credit_note", "payment", "other"] | None = None
    document_number: Str200 | None = None
    date: Str200 | None = None
    amount: float | None = None


class StatementExtraction(LlmOut):
    supplier_name: Str200 | None = None
    statement_date: Str200 | None = None
    currency: Str200 | None = None
    lines: list[StatementLine] = Field(default_factory=list, max_length=100)
    closing_balance: float | None = None


class AccountSuggestion(LlmOut):
    account_code: str = Field(max_length=20)
    reason: str = Field(default="", max_length=300)


# --------------------------------------------------------- batch plan

class AddNote(Strict):
    action: Literal["add_note"]
    contact_id: str = Field(min_length=1, max_length=40, pattern=r"^\d+$")
    note: str = Field(min_length=1, max_length=NOTE_MAX)

    @model_validator(mode="after")
    def _no_bank_details(self) -> "AddNote":
        from .security import redact
        if redact(self.note) != self.note:
            raise ValueError("note contains bank-detail-like or key-like values")
        return self


class CreateTask(Strict):
    action: Literal["create_task"]
    contact_id: str = Field(min_length=1, max_length=40, pattern=r"^\d+$")
    name: str = Field(max_length=60, pattern=TASK_RE)
    due_date: dt.date
    description: str | None = Field(default=None, max_length=500)


CrmAction = Annotated[Union[AddNote, CreateTask], Field(discriminator="action")]


class CrmPlan(Strict):
    actions: list[CrmAction] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def _shape(self) -> "CrmPlan":
        kinds = [a.action for a in self.actions]
        if any(kinds.count(k) > 1 for k in set(kinds)):
            raise ValueError("at most one add_note and one create_task per batch")
        if self.actions and kinds[0] != "add_note":
            raise ValueError("a non-empty CRM plan starts with add_note")
        if len({a.contact_id for a in self.actions}) > 1:
            raise ValueError("all CRM actions must target the same client record")
        return self


class BatchPlan(Strict):
    batch_id: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9-]+$")
    client_name: str = Field(min_length=1, max_length=120)
    approve_doc_ids: list[str] = Field(default_factory=list, max_length=500)
    export_doc_ids: list[str] = Field(default_factory=list, max_length=500)
    missing_count: int = Field(ge=0, le=999)
    crm: CrmPlan

    @model_validator(mode="after")
    def _task_matches(self) -> "BatchPlan":
        tasks = [a for a in self.crm.actions if isinstance(a, CreateTask)]
        if self.missing_count == 0 and tasks:
            raise ValueError("a chase task is only allowed when documents are missing")
        if self.missing_count and self.crm.actions and not tasks:
            raise ValueError("missing documents need the chase task")
        for t in tasks:
            if t.name != f"Chase {self.missing_count} missing documents":
                raise ValueError("task name must be 'Chase N missing documents' with N = missing documents")
        if not set(self.approve_doc_ids) <= set(self.export_doc_ids):
            raise ValueError("every document approved in this step must also be exported")
        return self

    def contact_ids(self) -> set[str]:
        return {a.contact_id for a in self.crm.actions}


def slug(text: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:n] or "batch"
