"""JSONL audit log: one file per run. File hashes, decisions, flags, tool calls and timings; secrets redacted."""
from __future__ import annotations

import datetime as dt
import json
import threading
from pathlib import Path
from typing import Any

from .security import redact


class AuditLog:
    def __init__(self, directory: Path | None, secrets: list[str] | None = None):
        self.directory = directory
        self.secrets = list(secrets or [])
        self.events: list[dict] = []  # in-memory copy (used by tests and eval)
        self._lock = threading.Lock()
        if directory:
            directory.mkdir(parents=True, exist_ok=True)

    def write(self, run_id: str, event: str, **data: Any) -> None:
        rec = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
               "run_id": run_id, "event": event, **data}
        rec = redact(rec, self.secrets)
        with self._lock:
            self.events.append(rec)
            if self.directory:
                with open(self.directory / f"{run_id}.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, default=str, ensure_ascii=True) + "\n")
