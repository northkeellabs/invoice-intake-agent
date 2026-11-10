"""The LangGraph workflows.

Per document:  validate -> text (text layer / OCR) -> classify -> extract -> check -> code -> route (stored)
Per batch:     ingest (document pipeline per file) -> reconcile statements -> lookup client (read-only MCP)
               -> plan (code) -> approval (interrupt) -> execute (dry run by default) -> report

The LLM is used for classify, extract and (for suppliers without a rule) an account-code suggestion. It has
no tools and cannot change routing, approval, bank details or supplier rules: all of that is code.
"""
from __future__ import annotations

import datetime as dt
import functools
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import ValidationError

from . import checks
from .audit import AuditLog
from .coding import Chart, decide
from .crm import Crm, CrmError
from .documents import FileRejected, OcrEngine, PageText, get_text, render_pages, validate_file
from .exports import chaser_text, exception_report, write_bills_csv, write_json
from .extract import LlmExtractor
from .llm import StructuredOutputError
from .normalize import doc_number_key, money, parse_date, supplier_key, vat_number_key
from .reconcile import reconcile_statement
from .schema import (BILL_TYPES, AddNote, BatchPlan, CreateTask, CrmPlan, Extraction, StatementExtraction, slug)
from .security import BankHasher, file_sha256, find_instructions, last4, normalise_bank, redact, sha256
from .store import Store

ALLOWED_EXT = (".pdf", ".png", ".jpg", ".jpeg")
MAX_APPROVAL_ROUNDS = 3
TASK_DUE_DAYS = 3


def _merge(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


@dataclass
class Deps:
    extractor: LlmExtractor
    ocr: OcrEngine
    store: Store
    chart: Chart
    crm: Crm
    audit: AuditLog
    hasher: BankHasher
    files_dir: Path
    out_dir: Path
    min_confidence: float
    iso_codes: set[str]
    today: Callable[[], dt.date] = field(default=dt.date.today)
    images: dict = field(default_factory=dict)  # doc_id -> rendered pages, only while a document is processed


def _timed(name: str, deps: Deps):
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(state: dict) -> dict:
            t0 = time.perf_counter()
            out = await fn(state) or {}
            secs = round(time.perf_counter() - t0, 3)
            deps.audit.write(state.get("run_id", "?"), "node", node=name, doc_id=state.get("doc_id"), seconds=secs)
            return {**out, "timings": {name: secs}}
        return wrapper
    return deco


# ====================================================================== document pipeline

class DocState(TypedDict, total=False):
    run_id: str
    batch_id: str
    doc_id: str
    path: str
    file_name: str
    status: str
    reason: str
    kind: str
    pages_count: int
    sha: str
    stored_path: str
    pages: list[dict]
    instructions: list[str]
    doc_type: str | None
    extraction: dict | None
    statement: dict | None
    llm_errors: list[str]
    grounding: dict
    flags: list[dict]
    confidence: float
    coding: dict
    route: str
    timings: Annotated[dict, _merge]


def _pages(state: DocState) -> list[PageText]:
    return [PageText(page=p["page"], text=p["text"], source=p["source"], confidence=p["confidence"])
            for p in state.get("pages") or []]


def route_document(doc_type: str | None, flags: list[dict], required_ok: bool, confidence: float,
                   min_confidence: float) -> str:
    """auto (auto-approve eligible, still approved by a person in the batch step) or review."""
    if doc_type in BILL_TYPES and not flags and required_ok and confidence >= min_confidence:
        return "auto"
    return "review"


def build_doc_graph(deps: Deps):
    audit, store = deps.audit, deps.store

    @_timed("validate", deps)
    async def validate(state: DocState) -> dict:
        p = Path(state["path"])
        try:
            kind, pages = validate_file(p)
        except (FileRejected, OSError) as exc:
            audit.write(state["run_id"], "file_rejected", doc_id=state["doc_id"], name_hash=sha256(p.name),
                        reason=str(exc))
            return {"status": "invalid", "reason": str(exc)}
        sha = file_sha256(p)
        stored = deps.files_dir / f"{sha}{p.suffix.lower()}"
        if not stored.exists():
            deps.files_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, stored)
        audit.write(state["run_id"], "input", doc_id=state["doc_id"], file_sha256=sha, kind=kind, pages=pages,
                    size=p.stat().st_size, name_hash=sha256(p.name))
        return {"kind": kind, "pages_count": pages, "sha": sha, "stored_path": str(stored), "status": "running"}

    @_timed("text", deps)
    async def text(state: DocState) -> dict:
        p = Path(state["stored_path"])
        images = render_pages(p, state["kind"])
        deps.images[state["doc_id"]] = images
        pages = get_text(p, state["kind"], deps.ocr, images)
        found = find_instructions("\n".join(pg.text for pg in pages))
        audit.write(state["run_id"], "text", doc_id=state["doc_id"],
                    pages=[{"page": pg.page, "source": pg.source, "confidence": pg.confidence, "chars": len(pg.text)}
                           for pg in pages], instruction_phrases=len(found))
        return {"pages": [{"page": pg.page, "text": pg.text, "source": pg.source, "confidence": pg.confidence}
                          for pg in pages], "instructions": found}

    @_timed("classify", deps)
    async def classify(state: DocState) -> dict:
        try:
            c, attempts = await deps.extractor.classify(_pages(state), deps.images.get(state["doc_id"], []))
        except StructuredOutputError as exc:
            audit.write(state["run_id"], "llm_invalid_output", doc_id=state["doc_id"], node="classify", error=str(exc)[:300])
            return {"doc_type": None, "llm_errors": ["classification failed (invalid model output twice)"]}
        audit.write(state["run_id"], "decision", doc_id=state["doc_id"], node="classify", doc_type=c.document_type,
                    attempts=attempts)
        return {"doc_type": c.document_type, "llm_errors": []}

    @_timed("extract", deps)
    async def extract(state: DocState) -> dict:
        dt_ = state.get("doc_type")
        imgs = deps.images.get(state["doc_id"], [])
        errors = list(state.get("llm_errors") or [])
        try:
            if dt_ in BILL_TYPES:
                ex, attempts = await deps.extractor.extract(_pages(state), imgs)
                audit.write(state["run_id"], "decision", doc_id=state["doc_id"], node="extract", attempts=attempts,
                            lines=len(ex.line_items))
                return {"extraction": ex.model_dump(), "statement": None}
            if dt_ == "statement":
                st, attempts = await deps.extractor.extract_statement(_pages(state), imgs)
                audit.write(state["run_id"], "decision", doc_id=state["doc_id"], node="extract_statement",
                            attempts=attempts, lines=len(st.lines))
                return {"statement": st.model_dump(), "extraction": None}
        except StructuredOutputError as exc:
            audit.write(state["run_id"], "llm_invalid_output", doc_id=state["doc_id"], node="extract", error=str(exc)[:300])
            errors.append("extraction failed (invalid model output twice)")
        return {"extraction": None, "statement": None, "llm_errors": errors}

    @_timed("check", deps)
    async def check(state: DocState) -> dict:
        pages = _pages(state)
        dtype = state.get("doc_type")
        text = checks.DocText.of(pages)
        flags: list[dict] = [checks.flag("extraction_failed", e) for e in state.get("llm_errors") or []]
        g: dict[str, str] = {}
        if state.get("instructions"):
            flags.append(checks.flag("instructions_found", "text that reads like instructions to the processor: "
                                     + "; ".join(f'"{s}"' for s in state["instructions"][:5])
                                     + ". Ignored; a person must review this document"))
        if dtype == "statement":
            flags.append(checks.flag("not_a_bill", "supplier statement: used for reconciliation, never exported as a bill"))
        elif dtype == "other" or dtype is None:
            flags.append(checks.flag("not_a_bill", "not an invoice, receipt or credit note; never exported as a bill"))
        flags += checks.low_ocr(pages)
        extra: dict[str, Any] = {}
        if state.get("extraction") is not None:
            ex = Extraction.model_validate(state["extraction"])
            g = checks.ground(ex, text)
            flags += checks.grounding_flags(g, ex)
            flags += checks.required(dtype, ex)
            flags += checks.arithmetic(ex)
            flags += checks.dates(ex, deps.today())
            flags += checks.currency(ex, deps.iso_codes)
            flags += checks.vat_number(ex)
            flags += checks.vat_completeness(dtype, ex)
            skey = supplier_key(ex.supplier_name)
            gross = money(ex.gross_total)
            issue = parse_date(ex.issue_date)
            dups = store.find_duplicates(state["doc_id"], skey, doc_number_key(ex.document_number),
                                         str(gross) if gross is not None else None, issue.isoformat() if issue else None)
            if dups:
                flags.append(checks.flag("duplicate", "; ".join(f"{d['match']} as {d['file_name']} ({d['status']})"
                                                                for d in dups[:3])))
            rule = store.supplier(skey) or store.supplier_by_vat(vat_number_key(ex.supplier_vat_number))
            bank = normalise_bank(ex.bank_sort_code, ex.bank_account_number, ex.bank_iban)
            bank_hash = deps.hasher.hash(bank)
            if rule and rule.get("bank_hash") and bank and bank_hash != rule["bank_hash"]:
                flags.append(checks.flag("bank_details_changed",
                                         checks.BANK_CHANGE_REASON + f" (record: account ending {rule.get('bank_last4')}; "
                                         f"document: account ending {last4(bank)})"))
            extra = {"supplier_key": rule["supplier_key"] if rule else skey, "gross": str(gross) if gross is not None else None,
                     "issue_date": issue.isoformat() if issue else None, "doc_number_key": doc_number_key(ex.document_number),
                     "bank_hash": bank_hash, "bank_last4": last4(bank), "rule": rule}
        elif state.get("statement") is not None:
            st = StatementExtraction.model_validate(state["statement"])
            g = checks.ground_statement(st, text)
            flags += checks.statement_grounding_flags(g, st)
            skey = supplier_key(st.supplier_name)
            extra = {"supplier_key": skey}
        conf = checks.extraction_confidence(pages, g) if state.get("extraction") is not None else 0.0
        audit.write(state["run_id"], "checks", doc_id=state["doc_id"], flags=[f["name"] for f in flags],
                    grounding_not_found=[k for k, v in g.items() if v == "not_found"], confidence=conf)
        return {"grounding": g, "flags": flags, "confidence": conf, "coding": extra}

    @_timed("code", deps)
    async def code(state: DocState) -> dict:
        info = dict(state.get("coding") or {})
        if state.get("doc_type") not in BILL_TYPES or state.get("extraction") is None:
            return {"coding": {**info, "code": None, "source": "none", "reason": "not a bill"}}
        ex = Extraction.model_validate(state["extraction"])
        rule = info.pop("rule", None)
        flags = list(state.get("flags") or [])
        suggestion, reason = None, ""
        if not rule:
            try:
                s, _ = await deps.extractor.suggest_code(ex.supplier_name or "(unknown)",
                                                         [li.description or "" for li in ex.line_items], deps.chart)
                suggestion, reason = s.account_code, s.reason
            except StructuredOutputError as exc:
                reason = f"no suggestion ({str(exc)[:80]})"
        r = decide(rule, suggestion, deps.chart, reason)
        if not r.valid:
            flags.append(checks.flag("account_code_invalid", r.reason))
        audit.write(state["run_id"], "decision", doc_id=state["doc_id"], node="code", source=r.source, code=r.code,
                    valid=r.valid)
        return {"coding": {**info, "code": r.code, "source": r.source, "reason": r.reason}, "flags": flags}

    @_timed("route", deps)
    async def route(state: DocState) -> dict:
        deps.images.pop(state["doc_id"], None)
        base = {"id": state["doc_id"], "batch_id": state["batch_id"], "file_name": state["file_name"],
                "file_path": state.get("stored_path"), "file_sha256": state.get("sha"), "kind": state.get("kind"),
                "pages": state.get("pages_count"), "timings": state.get("timings")}
        if state.get("status") == "invalid":
            store.add_document({**base, "status": "invalid", "route": "review", "error": state.get("reason"),
                                "flags": [checks.flag("not_a_bill", "file rejected: " + (state.get("reason") or ""))]})
            return {"route": "review"}
        ex = Extraction.model_validate(state["extraction"]) if state.get("extraction") else None
        flags = state.get("flags") or []
        req_ok = checks.required_grounded(ex, state.get("grounding") or {}) if ex else False
        r = route_document(state.get("doc_type"), flags, req_ok, state.get("confidence", 0.0), deps.min_confidence)
        c = state.get("coding") or {}
        store.add_document({**base, "doc_type": state.get("doc_type"), "extraction": state.get("extraction"),
                            "statement": state.get("statement"), "flags": flags, "grounding": state.get("grounding"),
                            "ocr": [{"page": p["page"], "source": p["source"], "confidence": p["confidence"]}
                                    for p in state.get("pages") or []],
                            "confidence": state.get("confidence"), "route": r,
                            "status": "auto_eligible" if r == "auto" else "review",
                            "account_code": c.get("code"), "account_source": c.get("source"),
                            "supplier_key": c.get("supplier_key"), "doc_number_key": c.get("doc_number_key"),
                            "gross": c.get("gross"), "issue_date": c.get("issue_date"), "bank_hash": c.get("bank_hash"),
                            "bank_last4": c.get("bank_last4"),
                            "text": "\n\n".join(p["text"] for p in state.get("pages") or [])})
        audit.write(state["run_id"], "decision", doc_id=state["doc_id"], node="route", route=r,
                    flags=[f["name"] for f in flags], confidence=state.get("confidence"),
                    threshold=deps.min_confidence)
        return {"route": r}

    def after_validate(s: DocState) -> str:
        return "route" if s.get("status") == "invalid" else "text"

    g = StateGraph(DocState)
    for name, fn in [("validate", validate), ("text", text), ("classify", classify), ("extract", extract),
                     ("check", check), ("code", code), ("route", route)]:
        g.add_node(name, fn)
    g.add_edge(START, "validate")
    g.add_conditional_edges("validate", after_validate, ["text", "route"])
    for a, b in [("text", "classify"), ("classify", "extract"), ("extract", "check"), ("check", "code"),
                 ("code", "route"), ("route", END)]:
        g.add_edge(a, b)
    return g.compile()


# ====================================================================== batch graph

class BatchState(TypedDict, total=False):
    run_id: str
    batch_id: str
    client_name: str
    folder: str | None
    live: bool
    status: str  # running | pending_approval | rejected | dry_run | executed | partial | failed | no_documents
    reason: str
    doc_ids: list[str]
    doc_latency: dict
    reconciliation: list[dict]
    contact_id: str | None
    lookup_note: str
    plan: dict
    plan_summary: dict
    decision: str
    results: list[dict]
    outputs: dict
    timings: Annotated[dict, _merge]


def batch_counts(docs: list[dict]) -> dict:
    st = lambda s: sum(d["status"] == s for d in docs)
    fl = lambda n: sum(any(f["name"] == n for f in d.get("flags") or []) for d in docs)
    return {"documents": len(docs), "auto_eligible": st("auto_eligible"), "approved": st("approved"),
            "review": st("review"), "rejected": st("rejected"), "invalid": st("invalid"),
            "statements": sum(d["doc_type"] == "statement" for d in docs),
            "other": sum(d["doc_type"] in ("other", None) and d["status"] != "invalid" for d in docs),
            "duplicates": fl("duplicate"), "bank_changes": fl("bank_details_changed"),
            "instructions": fl("instructions_found"),
            "failed_checks": sum(any(f["name"] != "not_a_bill" for f in d.get("flags") or []) for d in docs)}


def build_batch_plan(batch_id: str, client: str, contact_id: str | None, docs: list[dict], recon: list[dict],
                     today: dt.date) -> BatchPlan:
    approve = [d["id"] for d in docs if d["status"] == "auto_eligible"]
    export = approve + [d["id"] for d in docs if d["status"] == "approved" and d["doc_type"] in BILL_TYPES]
    missing = [m for r in recon for m in r["missing"]]
    c = batch_counts(docs)
    after = {**c, "approved": c["approved"] + c["auto_eligible"], "auto_eligible": 0}
    actions: list = []
    if contact_id:
        lines = [f"Document intake batch {batch_id} for {client} (invoice-intake-agent, logged after human approval).",
                 f"Documents received: {after['documents']}. Bills approved and exported: {len(export)}. "
                 f"In review queue: {after['review']}. Rejected: {after['rejected']}. Invalid files: {after['invalid']}. "
                 f"Statements: {after['statements']}. Other documents: {after['other']}.",
                 f"Exceptions: possible duplicates {after['duplicates']}; bank-detail changes {after['bank_changes']} "
                 f"(confirm by phone before paying); documents with failed checks {after['failed_checks']}; "
                 f"instructions found in documents {after['instructions']}; missing documents from statements "
                 f"{len(missing)}."]
        if missing:
            lines.append("Missing: " + "; ".join(f"{m['supplier']} {m['document_number']}" for m in missing[:10]))
        lines.append(f"Chasers drafted: {1 if missing else 0} (draft only, not sent).")
        lines.append("Nothing was paid, emailed or written to an accounting system by the agent.")
        note = redact("\n".join(lines))[:2000]
        actions.append(AddNote(action="add_note", contact_id=contact_id, note=note))
        if missing:
            desc = (f"Ask {client} for: " + "; ".join(f"{m['supplier']} {m['document_number']}" for m in missing[:10])
                    + f". Chaser draft (not sent): out/{batch_id}/chaser-to-client.txt")
            actions.append(CreateTask(action="create_task", contact_id=contact_id,
                                      name=f"Chase {len(missing)} missing documents",
                                      due_date=today + dt.timedelta(days=TASK_DUE_DAYS), description=redact(desc)[:500]))
    return BatchPlan(batch_id=batch_id, client_name=client, approve_doc_ids=approve, export_doc_ids=export,
                     missing_count=len(missing), crm=CrmPlan(actions=actions))


def check_plan_targets(p: BatchPlan, contact_id: str | None, batch_docs: set[str]) -> None:
    if p.contact_ids() - ({contact_id} if contact_id else set()):
        raise ValueError("plan targets a CRM record other than this client's")
    if not set(p.export_doc_ids) <= batch_docs:
        raise ValueError("plan exports documents from another batch")


def build_batch_graph(deps: Deps, checkpointer=None):
    audit, store = deps.audit, deps.store
    doc_graph = build_doc_graph(deps)

    @_timed("ingest", deps)
    async def ingest(state: BatchState) -> dict:
        ids, lat = list(state.get("doc_ids") or []), {}
        folder = state.get("folder")
        if folder:
            files = sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXT)
            for i, f in enumerate(files, start=1):
                doc_id = f"{state['batch_id']}-{i:03d}"
                t0 = time.perf_counter()
                await doc_graph.ainvoke({"run_id": state["run_id"], "batch_id": state["batch_id"], "doc_id": doc_id,
                                         "path": str(f), "file_name": f.name, "timings": {}})
                lat[doc_id] = round(time.perf_counter() - t0, 3)
                ids.append(doc_id)
        else:
            ids = [d["id"] for d in store.list(batch_id=state["batch_id"])]
        status = "running" if ids else "no_documents"
        return {"doc_ids": ids, "doc_latency": lat, "status": status}

    @_timed("reconcile", deps)
    async def reconcile(state: BatchState) -> dict:
        recon = [reconcile_statement(d, store) for d in store.list(batch_id=state["batch_id"])
                 if d["doc_type"] == "statement" and d.get("statement")]
        audit.write(state["run_id"], "reconcile", statements=len(recon),
                    missing=sum(len(r["missing"]) for r in recon),
                    mismatched=sum(len(r["amount_mismatch"]) for r in recon))
        return {"reconciliation": recon}

    @_timed("lookup", deps)
    async def lookup(state: BatchState) -> dict:
        client = state["client_name"]
        contact, note = None, "client record not found; the plan has no CRM actions"
        try:
            for r in await deps.crm.search(client):
                if (r.get("name") or "").strip().lower() == client.strip().lower():
                    contact = str(r.get("contact_id"))
                    kind = "company" if r.get("is_company") else "person"
                    note = f"client record found ({kind}); the batch note goes on it"
                    break
            audit.write(state["run_id"], "tool_call", tool="search_contacts", mode="read-only", found=bool(contact))
        except Exception as exc:  # a lookup failure must not lose the batch
            note = f"CRM lookup failed ({type(exc).__name__}); no CRM actions planned"
            audit.write(state["run_id"], "tool_error", tool="search_contacts", error=type(exc).__name__)
        return {"contact_id": contact, "lookup_note": note}

    def _plan(state: BatchState) -> BatchPlan:
        docs = store.list(batch_id=state["batch_id"])
        p = build_batch_plan(state["batch_id"], state["client_name"], state.get("contact_id"), docs,
                             state.get("reconciliation") or [], deps.today())
        check_plan_targets(p, state.get("contact_id"), {d["id"] for d in docs})
        return p

    @_timed("plan", deps)
    async def plan(state: BatchState) -> dict:
        p = _plan(state)
        docs = store.list(batch_id=state["batch_id"])
        audit.write(state["run_id"], "plan", approve=len(p.approve_doc_ids), export=len(p.export_doc_ids),
                    crm=[a.action for a in p.crm.actions], missing=p.missing_count)
        return {"plan": p.model_dump(mode="json"), "plan_summary": batch_counts(docs), "status": "pending_approval"}

    async def approval(state: BatchState) -> dict:
        # No @_timed: waiting for a person is not processing time.
        shown = state["plan"]
        error = None
        for _ in range(MAX_APPROVAL_ROUNDS):
            resp = interrupt({"run_id": state["run_id"], "batch_id": state["batch_id"], "client": state["client_name"],
                              "plan": shown, "summary": batch_counts(store.list(batch_id=state["batch_id"])),
                              "reconciliation": state.get("reconciliation"), "lookup_note": state.get("lookup_note"),
                              "live": state.get("live", False), "error": error})
            decision = (resp or {}).get("decision") if isinstance(resp, dict) else resp
            if decision == "reject":
                audit.write(state["run_id"], "approval", decision="reject")
                return {"decision": "reject", "status": "rejected"}
            if decision == "approve":
                current = _plan(state).model_dump(mode="json")
                if current != shown:  # documents changed in the review UI since the plan was shown
                    error = "documents changed since this plan was shown; review the updated plan"
                    audit.write(state["run_id"], "approval", decision="stale_plan")
                    shown = current
                    continue
                audit.write(state["run_id"], "approval", decision="approve")
                return {"decision": "approve", "plan": current}
            error = f"unknown decision {decision!r}; use approve or reject"
        audit.write(state["run_id"], "approval", decision="reject", reason="too many rounds")
        return {"decision": "reject", "status": "rejected", "reason": "too many approval rounds"}

    @_timed("execute", deps)
    async def execute(state: BatchState) -> dict:
        p = BatchPlan.model_validate(state["plan"])  # re-validate right before acting
        docs = {d["id"]: d for d in store.list(batch_id=state["batch_id"])}
        check_plan_targets(p, state.get("contact_id"), set(docs))
        for doc_id in p.approve_doc_ids:
            if docs[doc_id]["status"] == "auto_eligible":
                store.update_document(doc_id, status="approved", approved_by="batch approval")
        rules_created = 0
        for doc_id in p.export_doc_ids:
            d = store.get(doc_id)
            ex = Extraction.model_validate(d["extraction"] or {})
            if d.get("account_code") and store.create_supplier_rule(
                    d["supplier_key"], ex.supplier_name or "", ex.supplier_vat_number, d["account_code"],
                    d.get("bank_hash") or "", d.get("bank_last4") or "", doc_id):
                rules_created += 1
        out = deps.out_dir / state["batch_id"]
        out.mkdir(parents=True, exist_ok=True)
        exported = [store.get(i) for i in p.export_doc_ids]
        n_rows = write_bills_csv(out / "bills-xero.csv", exported, deps.chart)
        write_json(out / "bills.json", state["batch_id"], exported)
        audit.write(state["run_id"], "export", documents=len(exported), rows=n_rows, rules_created=rules_created)
        outputs = {"bills_csv": str(out / "bills-xero.csv"), "json": str(out / "bills.json"), "rows": n_rows,
                   "rules_created": rules_created}
        if not state.get("live"):
            audit.write(state["run_id"], "execute", mode="dry-run", write_calls=0,
                        actions=[a.action for a in p.crm.actions])
            return {"results": [{"action": a.action, "dry_run": True} for a in p.crm.actions], "status": "dry_run",
                    "outputs": outputs}
        results: list[dict] = []
        try:
            if p.crm.actions:
                async with deps.crm.writer() as w:
                    for a in p.crm.actions:
                        if isinstance(a, AddNote):
                            r = await w.call("add_note", {"contact_id": a.contact_id, "note": a.note})
                        else:
                            args = {"name": a.name, "due_date": a.due_date.isoformat(), "contact_id": a.contact_id}
                            if a.description:
                                args["description"] = a.description
                            r = await w.call("create_task", args)
                        results.append({"action": a.action, "ok": True, **{k: v for k, v in r.items() if k.endswith("_id")}})
                        audit.write(state["run_id"], "tool_call", tool=a.action, mode="write", ok=True)
        except Exception as exc:  # do not retry non-idempotent writes
            msg = str(exc)[:300] if isinstance(exc, CrmError) else type(exc).__name__
            audit.write(state["run_id"], "tool_error", node="execute", error=msg, completed=len(results))
            return {"results": results + [{"error": msg}], "status": "partial" if results else "failed",
                    "reason": msg, "outputs": outputs}
        return {"results": results, "status": "executed", "outputs": outputs}

    @_timed("report", deps)
    async def report(state: BatchState) -> dict:
        outputs = dict(state.get("outputs") or {})
        if state.get("status") != "no_documents":
            out = deps.out_dir / state["batch_id"]
            out.mkdir(parents=True, exist_ok=True)
            recon = state.get("reconciliation") or []
            missing = [m for r in recon for m in r["missing"]]
            chasers = []
            if missing:
                (out / "chaser-to-client.txt").write_text(chaser_text(state["client_name"], missing), encoding="utf-8")
                chasers.append("chaser-to-client.txt")
            docs = store.list(batch_id=state["batch_id"])
            (out / "exceptions.md").write_text(exception_report(state["batch_id"], state["client_name"], docs, recon,
                                                                chasers, state.get("status", "?"), state.get("live", False),
                                                                deps.today()), encoding="utf-8")
            outputs.update({"exceptions": str(out / "exceptions.md"), "chasers": chasers})
        audit.write(state["run_id"], "report", status=state.get("status"), timings=state.get("timings"),
                    documents=len(state.get("doc_ids") or []))
        return {"outputs": outputs}

    def after_ingest(s: BatchState) -> str:
        return "report" if s.get("status") == "no_documents" else "reconcile"

    def after_approval(s: BatchState) -> str:
        return "execute" if s.get("decision") == "approve" else "report"

    g = StateGraph(BatchState)
    for name, fn in [("ingest", ingest), ("reconcile", reconcile), ("lookup", lookup), ("plan", plan),
                     ("approval", approval), ("execute", execute), ("report", report)]:
        g.add_node(name, fn)
    g.add_edge(START, "ingest")
    g.add_conditional_edges("ingest", after_ingest, ["reconcile", "report"])
    g.add_edge("reconcile", "lookup")
    g.add_edge("lookup", "plan")
    g.add_edge("plan", "approval")
    g.add_conditional_edges("approval", after_approval, ["execute", "report"])
    g.add_edge("execute", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)


def new_batch_id(client: str) -> str:
    import uuid
    # Under 40 characters and a 6-digit date, so the key and account-number redactors never mask it in logs/notes.
    return f"{slug(client, 18).strip('-')}-{dt.date.today():%y%m%d}-{uuid.uuid4().hex[:5]}"
