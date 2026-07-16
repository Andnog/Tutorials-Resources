# Security Notes

OCR is executed locally with Tesseract. Its extracted text and model responses can contain fiscal
or personal data, so generated comparison CSV files remain under the git-ignored `data/outputs/`
directory. Review those files before sharing or moving them outside the project.

## API Keys

API keys must live in `.env` or in local environment variables. They must never be written into notebooks, committed to git, or shared in screenshots.

## Receipt Images

Tickets can include RFC, location, transaction amount, date, partial card data, or employee-related details. For that reason, `data/raw_receipts/` is ignored by git except for `.gitkeep`.

## Provider Choice

Sending receipts to a cloud API is a business decision. It can improve quality for blurry or complex images, but it also moves fiscal data outside the local machine. Local models reduce that exposure but may have lower extraction quality.

## Dependencies

This project uses direct provider calls with `requests` and minimal dependencies. In a production setting, every extra wrapper should be justified because it can expand the attack surface around credentials and fiscal data.

## Classroom Guidance

Before running the API sections, remind students:

- `.env` is local-only.
- The CSV contains expected answers and may reveal sensitive information.
- Generated outputs in `data/outputs/` should be reviewed before sharing.
