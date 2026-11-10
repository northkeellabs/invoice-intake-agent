"""FastAPI app: the review queue UI (server-rendered HTML, no JS framework) and the document webhook.

- GET /review, GET /review/{id}, POST /review/{id}: loopback clients only; the server binds to 127.0.0.1.
- POST /documents: multipart upload, X-Intake-Secret header, 10 MB cap, extension + magic-byte allowlist
  (pdf/png/jpg), 10-page cap, per-IP rate limit. The document runs through the pipeline into an inbox batch;
  `intake close <batch_id>` then builds the batch plan for approval.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import hmac
import io
import json
import re
import secrets
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Form, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from . import checks
from .config import Settings, webhook_secret
from .documents import MAX_BYTES, FileRejected, render_pages, validate_bytes
from .graph import build_doc_graph
from .normalize import doc_number_key, money, parse_date, supplier_key
from .schema import BILL_TYPES, Extraction, slug

RATE_PER_MIN = 30
LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}
EDITABLE = ("supplier_name", "supplier_vat_number", "document_number", "issue_date", "due_date", "currency",
            "net_total", "vat_total", "gross_total")
NUMERIC = ("net_total", "vat_total", "gross_total")
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _clean(value: str, field: str):
    v = (value or "").strip()[:200]
    if v == "":
        return None
    if field in NUMERIC:
        m = money(v)
        if m is None:
            raise ValueError(f"{field}: {v!r} is not an amount")
        return float(m)
    if field in ("issue_date", "due_date"):
        d = parse_date(v)
        if d is None:
            raise ValueError(f"{field}: {v!r} is not a date (use YYYY-MM-DD)")
        return d.isoformat()
    return v


def apply_review(store, doc: dict, form: dict, action: str, chart, reason: str = "") -> dict:
    """Record edits (before/after), then approve or reject. Shared by the UI and the tests."""
    ex = dict(doc.get("extraction") or {})
    changes: list[tuple[str, object, object]] = []
    for f in EDITABLE:
        if f in form:
            new = _clean(form[f], f)
            if new != ex.get(f):
                changes.append((f, ex.get(f), new))
                ex[f] = new
    code = (form.get("account_code") or "").strip() or doc.get("account_code")
    if code and not chart.valid(code):
        raise ValueError(f"account code {code!r} is not in the chart of accounts")
    if code != doc.get("account_code"):
        changes.append(("account_code", doc.get("account_code"), code))
    for f, before, after in changes:
        store.record_edit(doc["id"], f, before, after)
    upd: dict = {}
    if changes:
        e = Extraction.model_validate(ex)
        gross, issue = money(e.gross_total), parse_date(e.issue_date)
        upd = {"extraction": e.model_dump(), "account_code": code,
               "account_source": "human" if any(c[0] == "account_code" for c in changes) else doc.get("account_source"),
               "supplier_key": doc.get("supplier_key") or supplier_key(e.supplier_name),
               "doc_number_key": doc_number_key(e.document_number), "gross": str(gross) if gross is not None else None,
               "issue_date": issue.isoformat() if issue else None}
    if action == "approve":
        if doc["doc_type"] not in BILL_TYPES:
            raise ValueError("only invoices, receipts and credit notes can be approved as bills")
        if not code:
            raise ValueError("choose an account code before approving")
        upd.update({"status": "approved", "approved_by": "reviewer"})
        store.update_document(doc["id"], **upd)
        e = Extraction.model_validate(ex)
        # A rule is created only from a human-approved document, and never overwrites an existing rule
        # (so approving a document with new bank details does not change the supplier's bank record).
        store.create_supplier_rule(upd.get("supplier_key") or doc.get("supplier_key") or "", e.supplier_name or "",
                                   e.supplier_vat_number, code, doc.get("bank_hash") or "", doc.get("bank_last4") or "",
                                   doc["id"])
    elif action == "reject":
        if not reason.strip():
            raise ValueError("give a reason for rejecting")
        upd.update({"status": "rejected", "reject_reason": reason.strip()[:300]})
        store.update_document(doc["id"], **upd)
    elif upd:
        store.update_document(doc["id"], **upd)
    return {"changes": len(changes)}


def create_app(settings: Settings | None = None, deps_factory=None) -> FastAPI:
    settings = settings or Settings()
    hits: dict[str, deque] = defaultdict(deque)
    sem = asyncio.Semaphore(1)  # one local-LLM document at a time
    tasks: set[asyncio.Task] = set()
    form_token = secrets.token_urlsafe(24)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from .runtime import make_deps
        app.state.deps = (deps_factory or (lambda: make_deps(settings)))()
        app.state.doc_graph = build_doc_graph(app.state.deps)
        yield
        for t in list(tasks):
            await t

    app = FastAPI(title="invoice-intake-agent", version="0.1.0", lifespan=lifespan)

    def loopback_only(request: Request) -> None:
        host = request.client.host if request.client else ""
        if host not in LOOPBACK:
            raise HTTPException(403, "the review queue is only served to this machine")

    def rate(request: Request) -> None:
        ip = request.client.host if request.client else "?"
        q, now = hits[ip], time.monotonic()
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_PER_MIN:
            raise HTTPException(429, "rate limit")
        q.append(now)

    def guard(request: Request, x_intake_secret: str = Header(default="")) -> None:
        secret = webhook_secret()
        if not secret:
            raise HTTPException(503, "INTAKE_WEBHOOK_SECRET is not configured")
        if not hmac.compare_digest(x_intake_secret.encode(), secret.encode()):
            raise HTTPException(401, "bad or missing X-Intake-Secret")
        rate(request)

    # ------------------------------------------------------------------ webhook

    @app.post("/documents", status_code=202, dependencies=[Depends(guard)])
    async def post_document(request: Request) -> dict:
        clen = request.headers.get("content-length")
        if clen and clen.isdigit() and int(clen) > MAX_BYTES + 64 * 1024:
            raise HTTPException(413, "file too large (10 MB limit)")
        form = await request.form(max_files=1, max_fields=5)
        up = form.get("file")
        if up is None or not hasattr(up, "read"):
            raise HTTPException(422, "multipart field 'file' is required")
        data = await up.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "file too large (10 MB limit)")
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(up.filename or "upload").name)[-80:]
        try:
            validate_bytes(data, name)
        except FileRejected as exc:
            raise HTTPException(415, str(exc)) from None
        client = str(form.get("client") or settings.client_name)[:120]
        batch_id = f"inbox-{slug(client, 14).strip('-')}-{dt.date.today():%y%m%d}"
        folder = settings.inbox_dir / slug(client, 30)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{dt.datetime.now():%Y%m%d%H%M%S}-{secrets.token_hex(3)}-{name}"
        path.write_bytes(data)
        doc_id = f"{batch_id}-{secrets.token_hex(3)}"

        async def work():
            async with sem:
                await app.state.doc_graph.ainvoke({"run_id": batch_id, "batch_id": batch_id, "doc_id": doc_id,
                                                   "path": str(path), "file_name": name, "timings": {}})

        t = asyncio.create_task(work())
        tasks.add(t)
        t.add_done_callback(tasks.discard)
        return {"doc_id": doc_id, "batch_id": batch_id, "status": "accepted"}

    @app.get("/documents/{doc_id}", dependencies=[Depends(guard)])
    async def get_document(doc_id: str) -> dict:
        from .exports import public_doc
        d = app.state.deps.store.get(doc_id)
        if not d:
            raise HTTPException(404, "not found (or still processing)")
        return public_doc(d)

    # ------------------------------------------------------------------ review UI

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse("/review")

    @app.get("/review", response_class=HTMLResponse, dependencies=[Depends(loopback_only)])
    async def review_list(request: Request, batch: str = "", status: str = ""):
        store = app.state.deps.store
        docs = store.list(batch_id=batch or None, status=status or None)
        return TEMPLATES.TemplateResponse(request, "list.html", {"docs": docs, "batches": store.batches(),
                                                                 "batch": batch, "status": status})

    @app.get("/review/{doc_id}", response_class=HTMLResponse, dependencies=[Depends(loopback_only)])
    async def review_detail(request: Request, doc_id: str, error: str = ""):
        d = app.state.deps.store.get(doc_id)
        if not d:
            raise HTTPException(404, "document not found")
        rule = app.state.deps.store.supplier(d.get("supplier_key") or "") if d.get("supplier_key") else None
        return TEMPLATES.TemplateResponse(request, "detail.html", {
            "d": d, "ex": d.get("extraction") or {}, "st": d.get("statement") or {}, "editable": EDITABLE,
            "accounts": app.state.deps.chart.accounts, "token": form_token, "error": error[:300],
            "edits": app.state.deps.store.edits(doc_id), "rule": rule, "bill_types": BILL_TYPES,
            "titles": checks.FLAG_TITLES, "grounding": d.get("grounding") or {}})

    @app.get("/review/{doc_id}/page/{n}.png", dependencies=[Depends(loopback_only)])
    async def page_image(doc_id: str, n: int):
        d = app.state.deps.store.get(doc_id)
        if not d or not d.get("file_path"):
            raise HTTPException(404, "no page image")
        imgs = render_pages(Path(d["file_path"]), d["kind"], dpi=110)
        if not 1 <= n <= len(imgs):
            raise HTTPException(404, "no such page")
        buf = io.BytesIO()
        imgs[n - 1].save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png")

    @app.post("/review/{doc_id}", dependencies=[Depends(loopback_only)])
    async def review_post(request: Request, doc_id: str, action: str = Form(...), token: str = Form(...),
                          reason: str = Form(default="")):
        if not hmac.compare_digest(token, form_token):
            raise HTTPException(403, "stale form; reload the page")
        store = app.state.deps.store
        d = store.get(doc_id)
        if not d:
            raise HTTPException(404, "document not found")
        form = {k: str(v) for k, v in (await request.form()).items()}
        try:
            apply_review(store, d, form, action, app.state.deps.chart, reason)
        except ValueError as exc:
            from urllib.parse import quote
            return RedirectResponse(f"/review/{doc_id}?error={quote(str(exc))}", status_code=303)
        app.state.deps.audit.write(d["batch_id"], "review", doc_id=doc_id, action=action)
        return RedirectResponse(f"/review/{doc_id}", status_code=303)

    @app.get("/stats", dependencies=[Depends(loopback_only)])
    async def stats() -> dict:
        return straight_through(app.state.deps.store)

    return app


def straight_through(store) -> dict:
    """Straight-through rate from recorded reviews: approved bills that needed no human edit."""
    approved = [d for d in store.list(status="approved") if d["doc_type"] in BILL_TYPES]
    no_edit = [d for d in approved if not d.get("edited")]
    return {"approved_bills": len(approved), "approved_without_edit": len(no_edit),
            "straight_through_rate": round(len(no_edit) / len(approved), 3) if approved else None}
