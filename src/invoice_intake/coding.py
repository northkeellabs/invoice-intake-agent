"""Account coding: a supplier rule learned from a human-approved document always wins; only suppliers
without a rule get an LLM suggestion, and that suggestion must be a code in the chart (checked here)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Chart:
    accounts: dict[str, dict]
    tax_types: dict[str, str]

    @classmethod
    def load(cls, path: Path) -> "Chart":
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        accounts = {str(a["code"]): {"name": str(a.get("name", "")), "hint": str(a.get("hint", ""))}
                    for a in data.get("accounts", [])}
        if not accounts:
            raise ValueError(f"{path}: no accounts defined")
        return cls(accounts=accounts, tax_types={str(k): str(v) for k, v in (data.get("tax_types") or {}).items()})

    def valid(self, code: str | None) -> bool:
        return bool(code) and str(code).strip() in self.accounts

    def tax_type(self, rate: float | None, vat_charged: bool) -> str:
        if not vat_charged or rate is None:
            return self.tax_types.get("none", "")
        key = f"{rate:g}"
        return self.tax_types.get(key, "")

    def as_prompt(self) -> str:
        return "\n".join(f"- {c}: {a['name']} ({a['hint']})" for c, a in self.accounts.items())


@dataclass
class CodingResult:
    code: str | None
    source: str  # rule | llm | none
    reason: str
    valid: bool


def decide(rule: dict | None, suggestion: str | None, chart: Chart, reason: str = "") -> CodingResult:
    if rule and chart.valid(rule.get("account_code")):
        return CodingResult(rule["account_code"], "rule", "supplier rule from an approved document", True)
    if suggestion is None:
        return CodingResult(None, "none", reason or "no suggestion", False)
    s = str(suggestion).strip()
    if chart.valid(s):
        return CodingResult(s, "llm", reason, True)
    return CodingResult(None, "llm", f"model suggested {s[:20]!r}, which is not in the chart", False)
