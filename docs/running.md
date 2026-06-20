# Running the pipeline

We run this pipeline **ourselves** via the `--local` runner — each task executes
in our own process/container, not on Tensorlake's hosted orchestration. See
[`CLAUDE.md`](../CLAUDE.md) for how this fork is deployed and what we don't use.

The `ParseRequest` object is the single API surface. `examples/parse_pdf.py`
builds one and runs it locally.

---

## 1. Install

```bash
git clone <your-fork-url> stern-open-ingest
cd stern-open-ingest
pip install -e .
```

The `-e` (editable) flag means edits under `src/tensorlake_docai/` take effect
immediately — no reinstall after each change.

Optional dev tools:

```bash
pip install -e ".[dev]"   # pytest, ruff, black
```

For the GPU OCR path you also need torch + transformers:

```bash
pip install -e ".[cpu]"   # CPU box (no GPU OCR)
# pip install -e ".[gpu]"  # Linux CUDA box
# pip install vllm          # additionally required for dots-ocr on a GPU box
```

Sanity check:

```bash
python -c "from tensorlake_docai.pipeline.file_converter import normalize_file_type_and_upload; print('ok')"
pytest tests/ -q
```

---

## 2. Configure keys

Only the features you actually use need keys. The `.env.example` file groups them
by feature.

```bash
cp .env.example .env
$EDITOR .env
set -a; source .env; set +a
```

### Keys by feature

| Feature | Required env vars |
|---|---|
| `ocr_model="dots-ocr"` | none — needs a CUDA GPU host |
| Key-value extraction | `GEMINI_API_KEY` (default provider) or another LLM key |
| Table merging | `GEMINI_API_KEY` (default provider) or another LLM key |

Missing keys silently disable the dependent feature — the rest of the pipeline
keeps running.

---

## 2b. GPU pipeline (`dots-ocr`)

`dots-ocr` runs both DotsOCR (layout + OCR) and Ovis2.5-9B (figure OCR) on a
CUDA GPU. Run a `--local` Python process on a CUDA-equipped host.

**Worker setup.** The two GPU tasks (`DotsOCRTask` and `OvisFigureOCRTask`)
declare `gpu=["H100", "A100-80GB"]` on their `@function(...)` decorators. To
target a different GPU type, edit the `GPU_MODELS` constant at the top of
`src/tensorlake_docai/ocr/dots_ocr.py` and `src/tensorlake_docai/ocr/figure_ocr.py`.
Both tasks raise a `RequestError` at startup if `torch.cuda.is_available()` is
false, so misconfigurations fail fast.

**Model weights.** Both [rednote-hilab/dots.mocr](https://huggingface.co/rednote-hilab/dots.mocr)
and [AIDC-AI/Ovis2.5-9B](https://huggingface.co/AIDC-AI/Ovis2.5-9B) are pulled
from Hugging Face Hub on first cold-start (~25 GB).

Skip this section if you are only iterating on non-OCR stages (output formatting,
docs, routing predicates, etc.).

---

## 3. Run locally

```bash
python examples/parse_pdf.py --file my.pdf --ocr-model dots-ocr --local
```

What `--local` does:

- Every `@function`/`@application` in `src/tensorlake_docai/` runs **in your
  current Python process**, sequentially.
- External HTTP calls (OpenAI, Anthropic, Gemini) still happen for real.
- `print(...)` and `breakpoint()` work normally; you see every routing decision
  in your terminal.
- `dots-ocr` runs in-process too; you need a CUDA GPU on the host.

Output: `./debug/document.json` plus `./debug/document.md`.

---

## 4. Exercising more of the DAG

Every enrichment / detection stage is gated by a field on `ParseRequest`
(defined in [`src/tensorlake_docai/pipeline/api.py`](../src/tensorlake_docai/pipeline/api.py))
and `examples/parse_pdf.py` exposes them as CLI flags. Run
`python examples/parse_pdf.py --help` for the full list.

### OCR + retained enrichment

```bash
python examples/parse_pdf.py --file my.pdf --local \
    --ocr-model dots-ocr \
    --table-merging \
    --key-value-extraction \
    --xpage-header-detection
```

| Flag | Maps to `ParseRequest` field | Notes |
|---|---|---|
| `--table-merging` | `table_merging` | Stitch tables across pages or split by intervening text |
| `--key-value-extraction` | `key_value_extraction` | Key-value region markdown extraction |
| `--xpage-header-detection` | `xpage_header_detection` | Remove repeating headers/footers |

### Page selection and table format

```bash
python examples/parse_pdf.py --file my.pdf --local \
    --ocr-model dots-ocr \
    --pages 1 2 5 \
    --table-output-mode html \
    --table-merging
```

| Flag | Maps to `ParseRequest` field |
|---|---|
| `--pages` | `pages_to_parse` (1-indexed) |
| `--table-output-mode` | `table_output_mode` — `markdown` \| `html` |
| `--ignore-sections` | `ignore_sections` — e.g. `--ignore-sections page_footer figure` |

### File inputs

`ParseRequest` accepts the file in one of two mutually-exclusive forms:

| Field | Use it when… | Value |
|---|---|---|
| `file_bytes` | the file lives on local disk | base64-encoded raw bytes (string) |
| `file_url` | Rails has generated a presigned S3 URL | `"https://my-bucket.s3.amazonaws.com/key.pdf?...signature..."` |

`file_name` and `mime_type` are required in both cases. Direct `s3://` URLs are
not accepted; Rails should send a presigned HTTPS URL when the document lives in
S3.

---

## What you get back

Both examples produce a small response with the rendered markdown and usage metrics:

| Field | Contents |
|---|---|
| `document.document_markdown` | Full rendered markdown. Cross-page merged tables are applied here when `table_merging` is enabled. |
| `usage` | Token counts per stage (OCR / VLM / header correction) |

`parse_pdf.py` writes `./debug/document.json` plus `./debug/document.md`.

---

## Tests as runnable docs

```bash
pytest tests/ -q                          # full suite
pytest tests/test_pipeline_routing.py -v  # one passing test per DAG branch
pytest tests/test_routing.py -v
```

`tests/test_pipeline_routing.py` is the most useful file to skim once — every
routing predicate has a test that shows the exact `ParseRequest` fields that send
a request down each branch.
