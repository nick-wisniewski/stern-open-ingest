# CLAUDE.md

Project rules for AI assistants (Claude Code, Cursor, etc.) working in this repo.

## What this repo is

This is **Stern Risk's fork** of Tensorlake's OpenIngest, run as a self-hosted
document-parsing service for insurance policy processing. It turns PDFs and
images into layout-aware markdown plus structured data.

It is a **standalone provider service**. Our Rails app (`stern-liability`)
submits a parse request and gets the result back via webhook (or polls after a
delay). Nothing in the Rails stack changes to accommodate this — treat the two
as separate systems that talk over HTTP.

## How we run it — and what we do NOT use

We run the pipeline **ourselves**. We do **not** use Tensorlake's hosted
orchestration.

- We invoke the workflow with the `--local` runner — each task runs in our own
  process/container.
- Production target: a Modal web endpoint that stores provider state/artifacts in
  S3, returns a provider job id immediately, and spawns Modal CPU/GPU functions
  that fire the Rails webhook on completion.
- Do **not** use `tl deploy`, `TENSORLAKE_API_KEY`, or `scripts/sync-secrets.sh`.
  Those ship functions to "Tensorlake's pool" and are upstream-only — they do
  not apply here.
- The upstream `README.md` still describes the Tensorlake-hosted runtime (S3-backed
  durable orchestration, the Tensorlake UI, `tl deploy` as the production path).
  That is upstream context, not how we deploy. Do not wire up a Tensorlake API
  key or assume a managed runtime exists.

## OCR backend direction

The shipped `dots-ocr` backend (DotsOCR 3B + Ovis2.5-9B, needs an ~80GB GPU) is
heavier than we want, and not a strong table performer for our document mix.

- We are moving to a **lighter OCR backend** (currently evaluating PaddleOCR-VL,
  ~0.9B, fits a 12GB+ GPU) added under `src/tensorlake_docai/ocr/` via the
  documented bring-your-own-OCR path (register the backend and call
  `route_after_ocr` so it inherits table merging, enrichment, and structured
  extraction).
- Ahead of OCR we route **born-digital pages** (PDFs that already carry an
  extractable text layer) through cheap CPU text extraction, and only send
  scanned/image pages — and structurally hard tables — to the GPU OCR worker.
- Do not assume `dots-ocr`/Ovis is the engine. The specific backend is still
  being validated against our own documents; check the live registrations in
  `src/tensorlake_docai/ocr/` rather than the upstream default.

## Dev commands

After editing any Python under `src/`, `tests/`, or `examples/`, run both
formatters and the tests before reporting the task complete:

```bash
black src/ tests/ examples/
ruff check src/ tests/ examples/
pytest tests/ -q
```

Fix `ruff` errors before handing back. Don't silence rules with `# noqa` unless
the violation is intentional and worth a brief comment.

## Package structure

All functions live under `src/tensorlake_docai/...` and use absolute imports
(`from tensorlake_docai.… import …`). Keep that package layout intact when
adding modules — moving files out of the package breaks the bundled imports.
