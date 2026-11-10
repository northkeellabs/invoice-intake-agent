"""Test doubles: a fake OCR engine, a scripted fake chat model and a mocked MCP client. No Ollama, no network."""
import datetime as dt
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from PIL import Image

from invoice_intake.audit import AuditLog
from invoice_intake.coding import Chart
from invoice_intake.config import ROOT
from invoice_intake.extract import LlmExtractor
from invoice_intake.graph import Deps, build_batch_graph, build_doc_graph
from invoice_intake.rules import iso4217_codes
from invoice_intake.security import BankHasher, last4, normalise_bank
from invoice_intake.store import Store

TODAY = dt.date(2026, 11, 10)
CLIENT = "Harbour Lane Bakery (Demo)"
CLIENT_ID = "4100"


class FakeOcr:
    """Returns fixed text per image width (each test image gets a unique width)."""
    name = "fake-ocr"

    def __init__(self, texts: dict[int, tuple[str, float]]):
        self.texts = texts

    def lines(self, image):
        text, score = self.texts[image.width]
        return [(ln, score, 10.0, 20.0 * (i + 1), 12.0) for i, ln in enumerate(text.splitlines()) if ln.strip()]


class FakeModel:
    """Scripted chat model. Picks the answer by prompt kind and by a DOC:<key> marker in the document text.

    Never bound to tools: it only ever receives messages, which tests inspect.
    """

    def __init__(self, docs: dict[str, dict], codes: dict[str, str] | None = None):
        self.docs, self.codes = docs, codes or {}
        self.calls: list[tuple[str, list]] = []

    async def ainvoke(self, messages, **kw):
        system, user = messages[0].content, messages[1].content
        user = user if isinstance(user, str) else " ".join(p.get("text", "") for p in user if isinstance(p, dict))
        kind = ("classify" if "You sort incoming" in system else "statement" if "statement of account" in system
                else "code" if "account code" in system and "chart of accounts" in system else "extract")
        self.calls.append((kind, messages))
        if kind == "code":
            for name, code in self.codes.items():
                if name in user:
                    return AIMessage(json.dumps({"account_code": code, "reason": "test"}))
            return AIMessage(json.dumps({"account_code": "7900", "reason": "default"}))
        key = next(k for k in self.docs if f"DOC:{k}" in user)
        d = self.docs[key]
        if kind == "classify":
            return AIMessage(json.dumps({"document_type": d["type"], "reason": "test"}))
        return AIMessage(json.dumps(d.get("statement" if kind == "statement" else "extraction") or {}))


class FakeWriter:
    def __init__(self, crm):
        self.crm = crm

    async def call(self, tool, args):
        self.crm.write_calls.append((tool, args))
        if self.crm.fail_on == tool:
            raise RuntimeError(f"boom with key {self.crm.secret}")
        return {"add_note": {"note_id": "N1"}, "create_task": {"task_id": "T1"}}[tool]


class FakeCrm:
    """Mocked MCP client: records every call; writes only through writer()."""

    def __init__(self, records=None, fail_on=None, secret=""):
        self.records = records if records is not None else [{"contact_id": CLIENT_ID, "name": CLIENT, "is_company": True}]
        self.search_calls, self.write_calls = [], []
        self.writer_opened = 0
        self.fail_on, self.secret = fail_on, secret

    async def search(self, query):
        self.search_calls.append(query)
        return list(self.records)

    @asynccontextmanager
    async def writer(self):
        self.writer_opened += 1
        yield FakeWriter(self)


# ------------------------------------------------------------------ document fixtures

SUP = "Brackenfold Mills (Demo) Ltd"


def invoice_text(key: str, number="BFM-1001", date="03/11/2026", due="03/12/2026", net="1,250.00", vat="250.00",
                 total="1,500.00", vat_no="GB 123 4567 81", sort="00-00-42", acct="00004217", extra="") -> str:
    return (f"DOC:{key}\n{SUP}\nUnit 7, Brackenfold Works, Exampleton, ZZ8 5FD\nVAT Reg No: {vat_no}\nINVOICE\n"
            f"Invoice No. {number}\nInvoice Date {date}\nDue Date {due}\nInvoice to: Harbour Lane Bakery (Demo)\n"
            f"Description Qty Unit price VAT % Net\nStrong white flour 16kg 50 25.00 20% {net}\n"
            f"Net total {net}\nVAT @ 20% {vat}\nTotal due £{total}\n"
            f"Bank: Demo Bank plc Sort code {sort} Account {acct}\n{extra}\nDEMO DOCUMENT - FICTIONAL")


def invoice_extraction(**over) -> dict:
    ex = {"supplier_name": SUP, "supplier_address": "Unit 7, Brackenfold Works, Exampleton, ZZ8 5FD",
          "supplier_vat_number": "GB 123 4567 81", "customer_name": "Harbour Lane Bakery (Demo)",
          "document_number": "BFM-1001", "issue_date": "2026-11-03", "due_date": "2026-12-03", "currency": "GBP",
          "net_total": 1250.0, "vat_total": 250.0, "gross_total": 1500.0, "vat_rates": [20],
          "bank_sort_code": "00-00-42", "bank_account_number": "00004217", "bank_iban": None,
          "line_items": [{"description": "Strong white flour 16kg", "quantity": 50, "unit_price": 25.0,
                          "net_amount": 1250.0, "vat_rate": 20}]}
    ex.update(over)
    return ex


class Env:
    """A folder of fake documents + matching fake OCR and model answers."""

    def __init__(self, tmp: Path):
        self.tmp, self.folder = tmp, tmp / "in"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.texts: dict[int, tuple[str, float]] = {}
        self.docs: dict[str, dict] = {}
        self._w = 200

    def add(self, name: str, text: str, doc_type: str, extraction=None, statement=None, ocr_conf=0.97) -> Path:
        self._w += 1
        Image.new("RGB", (self._w, 100), "white").save(self.folder / name)
        key = Path(name).stem
        self.texts[self._w] = (text.replace("DOC:KEY", f"DOC:{key}"), ocr_conf)
        self.docs[key] = {"type": doc_type, "extraction": extraction, "statement": statement}
        return self.folder / name


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


@pytest.fixture
def make_deps(tmp_path):
    def _make(env: Env, crm=None, codes=None, secrets=(), min_conf=0.9, audit_dir=None):
        model = FakeModel(env.docs, codes)
        deps = Deps(extractor=LlmExtractor("ocr", model, "fake"), ocr=FakeOcr(env.texts), store=Store(tmp_path / "docs.sqlite"),
                    chart=Chart.load(ROOT / "config" / "coa.yaml"), crm=crm or FakeCrm(),
                    audit=AuditLog(audit_dir or tmp_path / "audit", list(secrets)), hasher=BankHasher(None),
                    files_dir=tmp_path / "files", out_dir=tmp_path / "out", min_confidence=min_conf,
                    iso_codes=iso4217_codes(), today=lambda: TODAY)
        return deps, model
    return _make


def seed_supplier(deps, name=SUP, code="5000", sort="00-00-42", acct="00004217", vat="GB 123 4567 81"):
    from invoice_intake.normalize import supplier_key
    bank = normalise_bank(sort, acct, None)
    deps.store.create_supplier_rule(supplier_key(name), name, vat, code, deps.hasher.hash(bank), last4(bank), "seed")


async def run_batch(deps, env: Env, live=False, decision="approve", thread="b1"):
    g = build_batch_graph(deps, InMemorySaver())
    cfg = {"configurable": {"thread_id": thread}}
    s = await g.ainvoke({"run_id": thread, "batch_id": thread, "client_name": CLIENT, "folder": str(env.folder),
                         "live": live, "timings": {}}, cfg)
    first = s
    if decision and s.get("__interrupt__"):
        s = await g.ainvoke(__import__("langgraph.types", fromlist=["Command"]).Command(resume={"decision": decision}), cfg)
    return first, s, g, cfg


def doc_graph(deps):
    return build_doc_graph(deps)
