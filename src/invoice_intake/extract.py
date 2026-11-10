"""Two extraction pipelines behind one interface (INTAKE_EXTRACTOR):

- ocr:    PDF text layer or RapidOCR text -> qwen3:8b (text model)
- vision: rendered page images -> qwen3-vl:8b (vision model reads the page directly)

Both modes always compute the text layer / OCR text too: the grounding check compares every extracted
value with that text, whichever model produced it.
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from . import prompts
from .coding import Chart
from .documents import PageText
from .llm import structured_call
from .schema import AccountSuggestion, Classification, Extraction, StatementExtraction

MAX_VISION_PAGES = 3


@dataclass
class LlmExtractor:
    name: str          # "ocr" | "vision"
    model: object      # chat model; never bound to tools
    provider: str = "ollama"

    @property
    def uses_images(self) -> bool:
        return self.name == "vision"

    def _inputs(self, pages: list[PageText], images: list[Image.Image]) -> tuple[str, list[Image.Image] | None]:
        if self.uses_images:
            return prompts.document_user("", with_image=True), images[:MAX_VISION_PAGES]
        text = "\n\n".join(f"--- page {p.page} ({p.source}) ---\n{p.text}" for p in pages)
        return prompts.document_user(text), None

    async def classify(self, pages, images) -> tuple[Classification, int]:
        user, imgs = self._inputs(pages, images)
        return await structured_call(self.model, prompts.classify_system(), user, Classification, self.provider, imgs)

    async def extract(self, pages, images) -> tuple[Extraction, int]:
        user, imgs = self._inputs(pages, images)
        return await structured_call(self.model, prompts.extract_system(), user, Extraction, self.provider, imgs)

    async def extract_statement(self, pages, images) -> tuple[StatementExtraction, int]:
        user, imgs = self._inputs(pages, images)
        return await structured_call(self.model, prompts.statement_system(), user, StatementExtraction,
                                     self.provider, imgs)

    async def suggest_code(self, supplier: str, lines: list[str], chart: Chart) -> tuple[AccountSuggestion, int]:
        # Text only, in both modes (the vision model handles text too, so only one model is ever loaded).
        return await structured_call(self.model, prompts.coding_system(chart.as_prompt()),
                                     prompts.coding_user(supplier, lines), AccountSuggestion, self.provider)
