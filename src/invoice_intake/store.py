"""SQLite document store: processed documents, human edits, and supplier rules.

Supplier rules (supplier -> account code, VAT number, bank details as keyed hash + last 4) are created
only from human-approved documents, and an existing rule's bank details are never changed by the agent.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, batch_id TEXT, file_name TEXT, file_path TEXT, file_sha256 TEXT, kind TEXT, pages INTEGER,
  doc_type TEXT, extraction TEXT, statement TEXT, flags TEXT, grounding TEXT, ocr TEXT, confidence REAL,
  route TEXT, status TEXT, account_code TEXT, account_source TEXT, supplier_key TEXT, doc_number_key TEXT,
  gross TEXT, issue_date TEXT, bank_hash TEXT, bank_last4 TEXT, reject_reason TEXT, edited INTEGER DEFAULT 0,
  approved_by TEXT, text TEXT, timings TEXT, error TEXT,
  created TEXT DEFAULT CURRENT_TIMESTAMP, updated TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX IF NOT EXISTS ix_docs_batch ON documents(batch_id);
CREATE INDEX IF NOT EXISTS ix_docs_supplier ON documents(supplier_key);
CREATE TABLE IF NOT EXISTS edits (
  id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT, field TEXT, before TEXT, after TEXT, editor TEXT,
  ts TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS suppliers (
  supplier_key TEXT PRIMARY KEY, name TEXT, vat_number TEXT, account_code TEXT, bank_hash TEXT, bank_last4 TEXT,
  created_from_doc TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP, updated TEXT DEFAULT CURRENT_TIMESTAMP);
"""
JSON_COLS = ("extraction", "statement", "flags", "grounding", "ocr", "timings")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._mem = sqlite3.connect(":memory:", check_same_thread=False) if self.path == ":memory:" else None
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        if self._mem is not None:
            return self._mem
        c = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        return c

    def _q(self, sql: str, args: tuple = ()) -> list[dict]:
        c = self._conn()
        c.row_factory = sqlite3.Row
        try:
            rows = [dict(r) for r in c.execute(sql, args).fetchall()]
        finally:
            c.row_factory = None
            if self._mem is None:
                c.close()
        for r in rows:
            for k in JSON_COLS:
                if k in r and r[k]:
                    r[k] = json.loads(r[k])
        return rows

    def _x(self, sql: str, args: tuple = ()) -> None:
        c = self._conn()
        try:
            with c:
                c.execute(sql, args)
        finally:
            if self._mem is None:
                c.close()

    # -------------------------------------------------------------- documents

    def add_document(self, rec: dict[str, Any]) -> None:
        rec = {k: (json.dumps(v, default=str) if k in JSON_COLS and v is not None else v) for k, v in rec.items()}
        cols = ", ".join(rec)
        self._x(f"INSERT OR REPLACE INTO documents ({cols}) VALUES ({', '.join('?' * len(rec))})", tuple(rec.values()))

    def update_document(self, doc_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields = {k: (json.dumps(v, default=str) if k in JSON_COLS and v is not None else v) for k, v in fields.items()}
        fields["updated"] = _now()
        sets = ", ".join(f"{k}=?" for k in fields)
        self._x(f"UPDATE documents SET {sets} WHERE id=?", (*fields.values(), doc_id))

    def get(self, doc_id: str) -> dict | None:
        rows = self._q("SELECT * FROM documents WHERE id=?", (doc_id,))
        return rows[0] if rows else None

    def list(self, batch_id: str | None = None, status: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM documents WHERE 1=1", []
        if batch_id:
            sql += " AND batch_id=?"
            args.append(batch_id)
        if status:
            sql += " AND status=?"
            args.append(status)
        return self._q(sql + " ORDER BY created, id", tuple(args))

    def batches(self) -> list[str]:
        return [r["batch_id"] for r in self._q("SELECT DISTINCT batch_id FROM documents ORDER BY batch_id")]

    def find_duplicates(self, doc_id: str, supplier_key: str, doc_number_key: str, gross: str | None,
                        issue_date: str | None) -> list[dict]:
        """Earlier documents with the same supplier + document number, or same supplier + total + date."""
        if not supplier_key:
            return []
        rows = self._q("SELECT id, file_name, doc_number_key, gross, issue_date, status, doc_type FROM documents "
                       "WHERE supplier_key=? AND id<>? AND doc_type IN ('invoice','receipt','credit_note')",
                       (supplier_key, doc_id))
        out = []
        for r in rows:
            if doc_number_key and r["doc_number_key"] == doc_number_key:
                out.append({**r, "match": "same supplier and document number"})
            elif gross and issue_date and r["gross"] == gross and r["issue_date"] == issue_date:
                out.append({**r, "match": "same supplier, total and date"})
        return out

    def documents_from_supplier(self, supplier_key: str) -> list[dict]:
        return self._q("SELECT id, file_name, doc_type, doc_number_key, gross, status, extraction FROM documents "
                       "WHERE supplier_key=? AND doc_type IN ('invoice','receipt','credit_note')", (supplier_key,))

    # -------------------------------------------------------------- edits

    def record_edit(self, doc_id: str, field: str, before: Any, after: Any, editor: str = "reviewer") -> None:
        self._x("INSERT INTO edits (doc_id, field, before, after, editor) VALUES (?,?,?,?,?)",
                (doc_id, field, json.dumps(before, default=str), json.dumps(after, default=str), editor))
        self._x("UPDATE documents SET edited=1, updated=? WHERE id=?", (_now(), doc_id))

    def edits(self, doc_id: str | None = None) -> list[dict]:
        if doc_id:
            return self._q("SELECT * FROM edits WHERE doc_id=? ORDER BY id", (doc_id,))
        return self._q("SELECT * FROM edits ORDER BY id")

    # -------------------------------------------------------------- suppliers

    def supplier(self, supplier_key: str) -> dict | None:
        rows = self._q("SELECT * FROM suppliers WHERE supplier_key=?", (supplier_key,))
        return rows[0] if rows else None

    def supplier_by_vat(self, vat_key: str) -> dict | None:
        if not vat_key:
            return None
        for s in self._q("SELECT * FROM suppliers WHERE vat_number IS NOT NULL AND vat_number<>''"):
            from .normalize import vat_number_key
            if vat_number_key(s["vat_number"]) == vat_key:
                return s
        return None

    def suppliers(self) -> list[dict]:
        return self._q("SELECT * FROM suppliers ORDER BY name")

    def create_supplier_rule(self, supplier_key: str, name: str, vat_number: str | None, account_code: str,
                             bank_hash: str, bank_last4: str, doc_id: str) -> bool:
        """Create a rule if none exists (returns True). An existing rule is never overwritten here."""
        if not supplier_key or self.supplier(supplier_key):
            return False
        self._x("INSERT INTO suppliers (supplier_key, name, vat_number, account_code, bank_hash, bank_last4, "
                "created_from_doc) VALUES (?,?,?,?,?,?,?)",
                (supplier_key, name, vat_number, account_code, bank_hash, bank_last4, doc_id))
        return True

    def confirm_bank_details(self, supplier_key: str, bank_hash: str, bank_last4: str) -> None:
        """Only called by the explicit human command `intake supplier confirm-bank` (never by the agent)."""
        self._x("UPDATE suppliers SET bank_hash=?, bank_last4=?, updated=? WHERE supplier_key=?",
                (bank_hash, bank_last4, _now(), supplier_key))
