"""Untrusted-data wrapping, the instruction heuristic, redaction, hashing and bank-detail masking."""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets as _secrets
from pathlib import Path
from typing import Any, Iterable

# Long opaque tokens that look like API keys or bearer tokens.
_KEYLIKE_RE = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|[A-Fa-f0-9]{32,}|[A-Za-z0-9_-]{40,})\b")
# UK sort code + account number pairs and IBANs, in case one ever reaches a log line.
_SORT_RE = re.compile(r"\b\d{2}[- ]\d{2}[- ]\d{2}\b")
_ACCOUNT_RE = re.compile(r"\b\d{8}\b")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?\b")

# Text in a document that tries to instruct whoever (or whatever) processes it. A document is data:
# a hit never changes anything except adding the flag "instructions found in document" (forces review).
_INSTRUCTION_RE = re.compile(
    r"(?i)("
    r"ignore (?:all |any |your |the )?(?:previous|prior|above|earlier) instructions|"
    r"disregard (?:all |any |your |the )?(?:previous|prior|above|earlier)|"
    r"(?:mark|set|flag) (?:(?:this|it|the)(?: invoice| document| bill)? )?(?:as )?(?:approved|paid|verified)\b|"
    r"auto[- ]?approve|approve (?:this|automatically|immediately)|"
    r"(?:use|set|code (?:this|it) to) account code|account code (?:to |=|:)?\s*0\b|"
    r"set (?:the )?(?:total|amount|balance)[a-z ]{0,20} to|"
    r"(?:update|change|amend) (?:our|the|your|supplier) bank details|"
    r"(?:new|updated) bank details (?:are|below|as follows)|"
    r"pay (?:it |this )?(?:today|immediately|now|urgently)|urgent(?:ly)? (?:payment|transfer)|"
    r"(?:note|message|instructions?) (?:to|for) (?:the )?(?:ai|assistant|agent|system|model|llm|automation)|"
    r"system prompt|you are (?:now )?(?:an? )?(?:ai|assistant|agent|language model)"
    r")")


def sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def wrap_untrusted(label: str, text: str, attrs: str = "") -> str:
    """Wrap document text so the model treats it as data. Neutralise any attempt to open or close a data tag."""
    safe = re.sub(r"(?i)</?\s*untrusted[^>]*>", "[tag removed]", text or "")
    a = f" {attrs}" if attrs else ""
    return f"<untrusted_{label}{a}>\n{safe}\n</untrusted_{label}>"


def find_instructions(text: str) -> list[str]:
    """Phrases in document text that read like instructions to the processor (heuristic, flag only)."""
    return sorted({re.sub(r"\s+", " ", m.group(0).lower()).strip() for m in _INSTRUCTION_RE.finditer(text or "")})


def redact(value: Any, secrets: Iterable[str] = ()) -> Any:
    """Recursively redact known secrets, key-like tokens and bank-account-like numbers from strings."""
    secrets = list(secrets)
    if isinstance(value, str):
        out = value
        for s in secrets:
            if s and len(s) >= 6:
                out = out.replace(s, "[REDACTED]")
        out = _KEYLIKE_RE.sub("[REDACTED]", out)
        out = _IBAN_RE.sub("[IBAN REDACTED]", out)
        out = _SORT_RE.sub("[SORT CODE REDACTED]", out)
        return _ACCOUNT_RE.sub("[ACCOUNT REDACTED]", out)
    if isinstance(value, dict):
        return {k: redact(v, secrets) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v, secrets) for v in value]
    return value


# ------------------------------------------------------------------ bank details

def normalise_bank(sort_code: str | None, account: str | None, iban: str | None) -> str:
    """Canonical string for bank details shown on a document ("" when none)."""
    if iban:
        return "IBAN:" + re.sub(r"[^A-Z0-9]", "", iban.upper())
    sc = re.sub(r"\D", "", sort_code or "")
    ac = re.sub(r"\D", "", account or "")
    return f"UK:{sc}:{ac}" if (sc or ac) else ""


def last4(canonical: str) -> str:
    digits = re.sub(r"[^0-9A-Z]", "", canonical.split(":", 1)[-1])
    return digits[-4:] if digits else ""


class BankHasher:
    """Keyed hash (HMAC-SHA256 with a random per-installation salt in data/) of bank details.

    Bank details have little entropy (14 digits for a UK sort code + account), so a plain hash could be
    reversed by brute force. Only the keyed hash and the last 4 digits are ever stored.
    """

    def __init__(self, salt_file: Path | None):
        if salt_file is None:
            self._key = b"test-salt"
            return
        salt_file.parent.mkdir(parents=True, exist_ok=True)
        if not salt_file.exists():
            salt_file.write_text(_secrets.token_hex(32), encoding="ascii")
        self._key = salt_file.read_text(encoding="ascii").strip().encode()

    def hash(self, canonical: str) -> str:
        if not canonical:
            return ""
        return hmac.new(self._key, canonical.encode(), hashlib.sha256).hexdigest()[:24]


def mask(canonical: str) -> str:
    """Display form for the review UI and reports: only the last 4 characters."""
    if not canonical:
        return ""
    kind = "IBAN" if canonical.startswith("IBAN:") else "account"
    return f"{kind} ending {last4(canonical)}"
