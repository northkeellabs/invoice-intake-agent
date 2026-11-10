"""Command line: intake run | close | pending | approve | serve | eval | calibrate | supplier | stats."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .config import ROOT, Settings, known_secrets, lacrm_api_key
from .runtime import open_runner, summarize
from .security import redact


def _emit(text: str, out: Path | None) -> None:
    text = redact(text.replace(str(ROOT) + "\\", "").replace(str(ROOT) + "/", "").replace(str(ROOT), "."),
                  known_secrets())
    print(text)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def _ask(prompt: dict) -> dict:
    if prompt.get("error"):
        print(f"!! {prompt['error']}")
    mode = "LIVE: approving writes the CRM note/task" if prompt.get("live") else "dry-run: no CRM writes"
    while True:
        ans = input(f"[{mode}] approve / reject the batch? [a/r]: ").strip().lower()
        if ans in ("a", "approve"):
            return {"decision": "approve"}
        if ans in ("r", "reject"):
            return {"decision": "reject"}


async def _drive(runner, res, args, out) -> None:
    _emit("=" * 70 + "\n" + summarize(res, runner.deps.store), out)
    while res["status"] == "pending_approval" and not args.no_prompt:
        res = await runner.resume(res["run_id"], _ask(res["approval"]))
        _emit("-" * 70 + "\n" + summarize(res, runner.deps.store), out)
    if res["status"] == "pending_approval":
        _emit(f"pending: review documents at http://127.0.0.1:8090/review, then "
              f"`intake approve {res['run_id']} --decision approve|reject`", out)


async def _run(args, settings: Settings) -> int:
    if args.live and not lacrm_api_key():
        print("LACRM_API_KEY is not set; cannot run --live.", file=sys.stderr)
        return 2
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"{folder} is not a folder", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else None
    async with open_runner(settings) as runner:
        res = await runner.start(str(folder), args.client or settings.client_name, live=args.live)
        await _drive(runner, res, args, out)
    return 0


async def _close(args, settings: Settings) -> int:
    out = Path(args.out) if args.out else None
    async with open_runner(settings) as runner:
        if not runner.deps.store.list(batch_id=args.batch_id):
            print(f"no documents in batch {args.batch_id}", file=sys.stderr)
            return 1
        res = await runner.start(None, args.client or settings.client_name, live=args.live, batch_id=args.batch_id)
        await _drive(runner, res, args, out)
    return 0


async def _pending(args, settings: Settings) -> int:
    async with open_runner(settings) as runner:
        rows = runner.index.pending()
        if not rows:
            print("no pending approvals")
        for run_id, batch_id, created in rows:
            p = await runner.pending_payload(run_id) or {}
            print(f"{run_id}  summary={p.get('summary')}  live={p.get('live')}  created={created}")
    return 0


async def _approve(args, settings: Settings) -> int:
    out = Path(args.out) if args.out else None
    async with open_runner(settings) as runner:
        payload = await runner.pending_payload(args.run_id)
        if payload is None:
            print(f"{args.run_id} is not waiting for approval", file=sys.stderr)
            return 1
        if payload.get("live") and not lacrm_api_key():
            print("LACRM_API_KEY is not set; cannot execute a live plan.", file=sys.stderr)
            return 2
        res = await runner.resume(args.run_id, {"decision": args.decision})
        _emit("-" * 70 + "\n" + summarize(res, runner.deps.store), out)
    return 0


def _supplier(args, settings: Settings) -> int:
    from .security import BankHasher, last4, normalise_bank
    from .store import Store
    store = Store(settings.store_db)
    if args.sup_cmd == "list":
        for s in store.suppliers():
            print(f"{s['supplier_key']:30s} {s['name']:40s} code={s['account_code']} vat={s['vat_number']} "
                  f"bank=ending {s['bank_last4'] or '-'} (from {s['created_from_doc']})")
        return 0
    d = store.get(args.doc_id)
    if not d or not d.get("bank_hash"):
        print("document not found or shows no bank details", file=sys.stderr)
        return 1
    if not store.supplier(d["supplier_key"]):
        print("no supplier record yet; approve a document from this supplier first", file=sys.stderr)
        return 1
    print("Only do this after confirming the new bank details by phone, using a number you already had "
          "(not one printed on the document).")
    if input(f"Set the bank details for {d['supplier_key']} to account ending {d['bank_last4']}? type YES: ") != "YES":
        print("unchanged")
        return 1
    store.confirm_bank_details(d["supplier_key"], d["bank_hash"], d["bank_last4"])
    print("supplier bank details updated by a person")
    return 0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="intake", description="Invoice intake agent (drafts and files only; never "
                                                            "sends email, pays, or writes to accounting systems).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process every PDF/PNG/JPG in a folder as one batch")
    r.add_argument("folder")
    r.add_argument("--client", help="client name (default INTAKE_CLIENT or 'Harbour Lane Bakery (Demo)')")
    r.add_argument("--live", action="store_true", help="write the CRM note/task after approval (default: dry run)")
    r.add_argument("--dry-run", action="store_true", help="explicit dry run (the default)")
    r.add_argument("--no-prompt", action="store_true", help="leave the batch pending instead of asking now")
    r.add_argument("--out", help="also append redacted output to this file")
    c = sub.add_parser("close", help="build the plan for documents already in a batch (e.g. from the webhook)")
    c.add_argument("batch_id")
    c.add_argument("--client")
    c.add_argument("--live", action="store_true")
    c.add_argument("--no-prompt", action="store_true")
    c.add_argument("--out")
    sub.add_parser("pending", help="list batches waiting for approval")
    a = sub.add_parser("approve", help="approve or reject a pending batch")
    a.add_argument("run_id")
    a.add_argument("--decision", choices=["approve", "reject"], required=True)
    a.add_argument("--out")
    s = sub.add_parser("serve", help="review queue + webhook on 127.0.0.1")
    s.add_argument("--port", type=int, default=8090)
    e = sub.add_parser("eval", help="run the eval set once per extractor")
    e.add_argument("--extractor", choices=["ocr", "vision", "both"], default="both")
    e.add_argument("--docs", default=str(ROOT / "evals" / "docs"))
    cal = sub.add_parser("calibrate", help="compute the auto-approve threshold on the calibration set")
    cal.add_argument("--extractor", choices=["ocr", "vision", "both"], default="both")
    cal.add_argument("--freeze", action="store_true", help="write the thresholds to config/thresholds.json")
    sp = sub.add_parser("supplier", help="supplier rules")
    ssub = sp.add_subparsers(dest="sup_cmd", required=True)
    ssub.add_parser("list")
    cb = ssub.add_parser("confirm-bank", help="a person confirms new bank details from a document (after a phone check)")
    cb.add_argument("doc_id")
    sub.add_parser("stats", help="straight-through rate from recorded reviews")
    args = ap.parse_args(argv)
    if getattr(args, "live", False) and getattr(args, "dry_run", False):
        ap.error("--live and --dry-run are mutually exclusive")
    settings = Settings()
    if args.cmd == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(settings), host="127.0.0.1", port=args.port)  # never another interface
        return
    if args.cmd == "eval":
        from .evals import run_eval
        raise SystemExit(asyncio.run(run_eval(settings, Path(args.docs), args.extractor)))
    if args.cmd == "calibrate":
        from .evals import run_calibration
        raise SystemExit(asyncio.run(run_calibration(settings, args.extractor, args.freeze)))
    if args.cmd == "supplier":
        raise SystemExit(_supplier(args, settings))
    if args.cmd == "stats":
        from .api import straight_through
        from .store import Store
        print(straight_through(Store(settings.store_db)))
        return
    fn = {"run": _run, "close": _close, "pending": _pending, "approve": _approve}[args.cmd]
    raise SystemExit(asyncio.run(fn(args, settings)))


if __name__ == "__main__":
    main()
