# Auto-approve threshold calibration: vision extractor, 2026-11-10

Model: qwen3-vl:8b. 12 fictional calibration documents (evals/calibration/; the eval set was not used). Confidence = lowest page text confidence x share of extracted values found in the document text (defined in code; the model's self-rating is not used). Correct = every scored field matches the ground truth after normalisation.

- Correct extractions: 6/12
- Threshold: 0.9714 (groups separate: midpoint of 0.9583 and 0.9844)

| document | medium | confidence | all fields correct | wrong fields |
|---|---|---|---|---|
| cal-01-invoice.pdf | text_pdf | 0.9583 | False | issue_date |
| cal-02-invoice.pdf | text_pdf | 1.0000 | True | - |
| cal-03-invoice.pdf | text_pdf | 1.0000 | True | - |
| cal-04-invoice.pdf | text_pdf | 1.0000 | True | - |
| cal-05-invoice.pdf | text_pdf | 0.8750 | False | issue_date, due_date |
| cal-06-invoice.pdf | text_pdf | 0.9167 | False | issue_date, due_date |
| cal-07-scan.jpg | scan_jpg | 0.9904 | True | - |
| cal-08-scan.pdf | scan_pdf | 0.9930 | True | - |
| cal-09-scan.jpg | scan_jpg | 0.5831 | False | document_number, issue_date, due_date |
| cal-10-scan.jpg | scan_jpg | 0.8690 | False | document_number, issue_date, due_date |
| cal-11-receipt.jpg | receipt_image | 0.9461 | False | document_number |
| cal-12-receipt.png | receipt_image | 0.9844 | True | - |
