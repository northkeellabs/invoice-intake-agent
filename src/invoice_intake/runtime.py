"""Shared runner for the CLI, the webhook/review server and the eval: SQLite checkpointer + run index."""
from __future__ import annotations

import datetime as dt
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from .audit import AuditLog
from .coding import Chart
from .config import Settings, known_secrets
from .crm import Crm, McpCrm
from .documents import OcrEngine, RapidOcrEngine
from .extract import LlmExtractor
from .graph import Deps, build_batch_graph, new_batch_id
from .llm import make_model
from .rules import iso4217_codes
from .security import BankHasher
from .store import Store


class RunIndex:
    """Tiny table of batch runs so pending approvals can be listed after a restart."""

    def __init__(self, path: Path):
        self.path = path
        with sqlite3.connect(path) as c:
            c.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, batch_id TEXT, status TEXT, "
                      "created TEXT DEFAULT CURRENT_TIMESTAMP)")

    def set(self, run_id: str, batch_id: str, status: str) -> None:
        with sqlite3.connect(self.path) as c:
            c.execute("INSERT INTO runs (run_id, batch_id, status) VALUES (?,?,?) "
                      "ON CONFLICT(run_id) DO UPDATE SET status=excluded.status", (run_id, batch_id, status))

    def pending(self) -> list[tuple[str, str, str]]:
        with sqlite3.connect(self.path) as c:
            return c.execute("SELECT run_id, batch_id, created FROM runs WHERE status='pending_approval' "
                             "ORDER BY created").fetchall()


class Runner:
    def __init__(self, graph, index: RunIndex | None, deps: Deps, settings: Settings):
        self.graph, self.index, self.deps, self.settings = graph, index, deps, settings

    async def start(self, folder: str | None, client: str, live: bool = False, batch_id: str | None = None) -> dict:
        batch_id = batch_id or new_batch_id(client)
        run_id = batch_id
        state = await self.graph.ainvoke({"run_id": run_id, "batch_id": batch_id, "client_name": client,
                                          "folder": folder, "live": live, "timings": {}},
                                         {"configurable": {"thread_id": run_id}})
        return await self._result(run_id, state)

    async def resume(self, run_id: str, response: dict) -> dict:
        state = await self.graph.ainvoke(Command(resume=response), {"configurable": {"thread_id": run_id}})
        return await self._result(run_id, state)

    async def pending_payload(self, run_id: str) -> dict | None:
        snap = await self.graph.aget_state({"configurable": {"thread_id": run_id}})
        if snap.interrupts:
            return snap.interrupts[0].value
        return None

    async def _result(self, run_id: str, state: dict) -> dict:
        interrupts = state.get("__interrupt__") if isinstance(state, dict) else None
        status = "pending_approval" if interrupts else state.get("status", "unknown")
        if self.index:
            self.index.set(run_id, state.get("batch_id", ""), status)
        return {"run_id": run_id, "status": status, "state": state,
                "approval": interrupts[0].value if interrupts else None}


def make_deps(settings: Settings, *, crm: Crm | None = None, model=None, ocr: OcrEngine | None = None,
              store: Store | None = None, audit: AuditLog | None = None,
              today: Callable[[], dt.date] | None = None) -> Deps:
    settings.ensure_dirs()
    vision = settings.extractor == "vision"
    extractor = LlmExtractor(settings.extractor, model or make_model(settings, vision=vision), settings.model_provider)
    return Deps(extractor=extractor, ocr=ocr or RapidOcrEngine(), store=store or Store(settings.store_db),
                chart=Chart.load(settings.coa_path), crm=crm or McpCrm(settings),
                audit=audit or AuditLog(settings.data_dir / "audit", known_secrets()),
                hasher=BankHasher(settings.data_dir / "bank-salt"), files_dir=settings.data_dir / "files",
                out_dir=settings.out_dir, min_confidence=settings.min_confidence, iso_codes=iso4217_codes(),
                today=today or dt.date.today)


@asynccontextmanager
async def open_runner(settings: Settings, **kw: Any) -> AsyncIterator[Runner]:
    deps = make_deps(settings, **kw)
    async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_db)) as saver:
        yield Runner(build_batch_graph(deps, saver), RunIndex(settings.data_dir / "runs.sqlite"), deps, settings)


def summarize(result: dict[str, Any], store: Store | None = None) -> str:
    """Human-readable summary of a batch run (CLI and saved outputs). Bank details are never shown."""
    st = result.get("state") or {}
    ap = result.get("approval") or {}
    batch = st.get("batch_id") or ap.get("batch_id")
    lines = [f"run_id: {result['run_id']}", f"status: {result['status']}", f"batch: {batch}",
             f"client: {st.get('client_name') or ap.get('client')}"]
    if st.get("lookup_note") or ap.get("lookup_note"):
        lines.append(f"crm: {st.get('lookup_note') or ap.get('lookup_note')}")
    if store and batch:
        lines.append("documents:")
        for d in store.list(batch_id=batch):
            ex = d.get("extraction") or {}
            what = (f"{ex.get('supplier_name')} {ex.get('document_number')} {ex.get('currency') or ''} "
                    f"{ex.get('gross_total')}" if ex else (d.get("statement") or {}).get("supplier_name") or "")
            lines.append(f"  - {d['id']} {d['file_name']}: {d['doc_type']} [{d['status']}] {what} "
                         f"code={d.get('account_code')} ({d.get('account_source')}) conf={d.get('confidence')}")
            for f in d.get("flags") or []:
                lines.append(f"      ! {f['name']}: {f['reason']}")
    rec = st.get("reconciliation") or ap.get("reconciliation") or []
    for r in rec:
        lines.append(f"statement {r['statement_doc']} ({r['supplier']}): matched {len(r['matched'])}, "
                     f"missing {len(r['missing'])}, amount mismatch {len(r['amount_mismatch'])}")
        for m in r["missing"]:
            lines.append(f"      missing: {m['document_number']} {m.get('date')} {m.get('amount')}")
    plan = st.get("plan") or ap.get("plan")
    if plan:
        lines.append(f"plan: approve {len(plan['approve_doc_ids'])} auto-eligible document(s), export "
                     f"{len(plan['export_doc_ids'])} bill(s), missing documents {plan['missing_count']}")
        for a in plan["crm"]["actions"]:
            lines.append("  - " + ", ".join(f"{k}={v}" for k, v in a.items() if k != "note"))
            if "note" in a:
                lines += ["      | " + l for l in a["note"].splitlines()]
    if ap.get("error"):
        lines.append(f"!! {ap['error']}")
    if st.get("results"):
        lines.append("execution:")
        lines += [f"  - {r}" for r in st["results"]]
    if st.get("outputs"):
        lines.append("outputs: " + ", ".join(f"{k}={v}" for k, v in st["outputs"].items()))
    if st.get("timings"):
        lines.append("timings (s): " + ", ".join(f"{k}={v}" for k, v in st["timings"].items()))
    return "\n".join(lines)
