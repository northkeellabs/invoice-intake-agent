# Auto-approve threshold calibration: ocr extractor, 2026-11-10

Model: qwen3:8b. 12 fictional calibration documents (evals/calibration/; the eval set was not used). Confidence = lowest page text confidence x share of extracted values found in the document text (defined in code; the model's self-rating is not used). Correct = every scored field matches the ground truth after normalisation.

- Correct extractions: 8/12
- Threshold: 0.9956 (groups overlap: cut with best balanced accuracy (75%), ties to the highest cut)

| document | medium | confidence | all fields correct | wrong fields |
|---|---|---|---|---|
| cal-01-invoice.pdf | text_pdf | 1.0000 | True | - |
| cal-02-invoice.pdf | text_pdf | 0.9583 | False | due_date |
| cal-03-invoice.pdf | text_pdf | 1.0000 | True | - |
| cal-04-invoice.pdf | text_pdf | 1.0000 | True | - |
| cal-05-invoice.pdf | text_pdf | 0.9583 | True | - |
| cal-06-invoice.pdf | text_pdf | 1.0000 | True | - |
| cal-07-scan.jpg | scan_jpg | 0.9904 | True | - |
| cal-08-scan.pdf | scan_pdf | 0.9103 | True | - |
| cal-09-scan.jpg | scan_jpg | 0.8383 | False | supplier_vat_number, document_number, issue_date |
| cal-10-scan.jpg | scan_jpg | 0.8690 | False | net_total, vat_total, gross_total |
| cal-11-receipt.jpg | receipt_image | 0.9912 | False | document_number, net_total, vat_total, gross_total |
| cal-12-receipt.png | receipt_image | 0.9844 | True | - |
