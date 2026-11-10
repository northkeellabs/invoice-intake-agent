"""Settings from environment variables. Secrets are read here and nowhere else."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Frozen auto-approve thresholds on the extraction confidence (defined in code, see checks.extraction_confidence).
# Chosen on evals/calibration/ only (never on evals/docs/) by `intake calibrate`, then copied here before eval run 1.
# Below the threshold a document goes to the review queue. See evals/calibration/RESULTS-<extractor>.md.
DEFAULT_MIN_CONFIDENCE = {"ocr": 0.0, "vision": 0.0}  # replaced by the calibration result before the first eval run

# Low-OCR-confidence flag: mean RapidOCR line score on a page below this value. Fixed design choice, not calibrated.
LOW_OCR_CONFIDENCE = 0.80

DEFAULT_CLIENT = "Harbour Lane Bakery (Demo)"


def _user_env(name: str) -> str:
    """Environment variable, falling back to the Windows user environment (registry)."""
    val = os.environ.get(name, "")
    if val or os.name != "nt":
        return val.strip()
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return str(winreg.QueryValueEx(k, name)[0]).strip()
    except OSError:
        return ""


def lacrm_api_key() -> str:
    return _user_env("LACRM_API_KEY")


def webhook_secret() -> str:
    return _user_env("INTAKE_WEBHOOK_SECRET")


def known_secrets() -> list[str]:
    return [s for s in (lacrm_api_key(), webhook_secret(), os.environ.get("ANTHROPIC_API_KEY", ""),
                        os.environ.get("OPENAI_API_KEY", "")) if s]


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _thresholds() -> dict[str, float]:
    """Frozen calibration result (config/thresholds.json) unless overridden by INTAKE_MIN_CONFIDENCE."""
    out = dict(DEFAULT_MIN_CONFIDENCE)
    f = ROOT / "config" / "thresholds.json"
    if f.exists():
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            out.update({k: float(v["threshold"]) for k, v in data.items() if k in out})
        except (ValueError, KeyError, TypeError):
            pass
    return out


@dataclass
class Settings:
    extractor: str = field(default_factory=lambda: os.environ.get("INTAKE_EXTRACTOR", "ocr").lower())
    model_provider: str = field(default_factory=lambda: os.environ.get("MODEL_PROVIDER", "ollama").lower())
    model_name: str = field(default_factory=lambda: os.environ.get("MODEL_NAME", "qwen3:8b"))
    vision_model: str = field(default_factory=lambda: os.environ.get("VISION_MODEL", "qwen3-vl:8b"))
    ollama_base_url: str = field(default_factory=lambda: os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("INTAKE_DATA_DIR", ROOT / "data")))
    out_dir: Path = field(default_factory=lambda: Path(os.environ.get("INTAKE_OUT_DIR", ROOT / "out")))
    inbox_dir: Path = field(default_factory=lambda: Path(os.environ.get("INTAKE_INBOX_DIR", ROOT / "inbox")))
    coa_path: Path = field(default_factory=lambda: Path(os.environ.get("INTAKE_COA", ROOT / "config" / "coa.yaml")))
    client_name: str = field(default_factory=lambda: os.environ.get("INTAKE_CLIENT", DEFAULT_CLIENT))
    min_confidence_override: float | None = field(
        default_factory=lambda: _float_env("INTAKE_MIN_CONFIDENCE", -1.0) if "INTAKE_MIN_CONFIDENCE" in os.environ else None)
    lacrm_mcp_dir: str = field(default_factory=lambda: _user_env("LACRM_MCP_DIR"))
    lacrm_mcp_env: str = field(default_factory=lambda: _user_env("LACRM_MCP_ENV"))

    def __post_init__(self) -> None:
        if self.extractor not in ("ocr", "vision"):
            raise ValueError(f"INTAKE_EXTRACTOR must be 'ocr' or 'vision', not {self.extractor!r}")

    @property
    def min_confidence(self) -> float:
        if self.min_confidence_override is not None:
            return self.min_confidence_override
        return _thresholds()[self.extractor]

    def ensure_dirs(self) -> None:
        (self.data_dir / "audit").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "files").mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_db(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    @property
    def store_db(self) -> Path:
        return self.data_dir / "documents.sqlite"
