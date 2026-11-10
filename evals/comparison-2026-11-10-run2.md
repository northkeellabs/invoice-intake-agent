# Extractor comparison, 2026-11-10

Same 36 synthetic documents, same checks, same frozen thresholds per extractor. ocr: qwen3:8b | vision: qwen3-vl:8b

| metric | ocr (qwen3:8b) | vision (qwen3-vl:8b) |
|---|---|---|
| Review recall (positive = review) | 100% (11/11) | 100% (11/11) |
| Review precision | 73% (11/15) | 61% (11/18) |
| Document type accuracy | 36/36 (100%) | 36/36 (100%) |
| Bills with every field correct | 32/33 (97%) | 26/33 (79%) |
|   text PDF | 20/21 (95%) | 17/21 (81%) |
|   scan | 7/7 (100%) | 4/7 (57%) |
|   receipt | 5/5 (100%) | 5/5 (100%) |
| Field: supplier_name | 33/33 (100%) | 33/33 (100%) |
| Field: supplier_vat_number | 33/33 (100%) | 33/33 (100%) |
| Field: document_number | 33/33 (100%) | 33/33 (100%) |
| Field: issue_date | 33/33 (100%) | 28/33 (85%) |
| Field: due_date | 32/33 (97%) | 32/33 (97%) |
| Field: currency | 33/33 (100%) | 33/33 (100%) |
| Field: net_total | 33/33 (100%) | 33/33 (100%) |
| Field: vat_total | 33/33 (100%) | 33/33 (100%) |
| Field: gross_total | 33/33 (100%) | 33/33 (100%) |
| Field: bank_details | 33/33 (100%) | 31/33 (94%) |
| Field: line_count | 33/33 (100%) | 33/33 (100%) |
| Account code accuracy | 32/33 (97%) | 32/33 (97%) |
| Duplicate detected | 1/1 | 1/1 |
| Bank-detail change detected | 2/2 | 2/2 |
| Missing documents found | 2/2 | 2/2 |
| Injection resisted | 3/3 | 3/3 |
| Grounding: wrong values caught / missed | 1 / 0 | 5 / 1 |
| Grounding: correct values flagged | 0 | 0 |
| Straight-through (no edit needed) | 31/33 (94%) | 25/33 (76%) |
| Latency mean / p95 per document | 5.3 s / 6.4 s | 7.4 s / 9.4 s |
| Peak GPU memory in use | 6995 MiB | 9748 MiB |
