# stern-open-ingest

Stern Risk's self-hosted fork of Tensorlake's [OpenIngest](https://github.com/tensorlakeai/openingest)
— a standalone document-parsing service for insurance policy processing. It turns
PDFs and images into layout-aware markdown plus structured data.

It runs as a **standalone provider service**. Our Rails app (`stern-liability`)
submits a parse request and gets the result back over HTTP (webhook or poll).
The two systems are independent and talk only over HTTP.

> We deliberately don't use Tensorlake-hosted deployment.

---

## How it runs

We run the pipeline **ourselves** with the `--local` runner — each task executes
in our own process/container. We do **not** use Tensorlake's hosted
orchestration, `tl deploy`, or a `TENSORLAKE_API_KEY`.

The production target is scale-to-zero GPU workers pulling from our own queue,
with a small always-on HTTP receiver in front (accepts the Rails request, returns
200, enqueues, fires the webhook on completion). That layer is not built yet —
today the entry point is the example scripts and the `ParseRequest` API.

---

## What it does

Given a PDF or image (PNG, JPEG, HEIF, HEIC), the workflow:

1. **Validates** the input MIME type (content detection via `python-magic`, with filename
   extension as a fallback).
2. **Runs OCR / layout** with `dots-ocr` (the only backend in this fork; see
   [`docs/models.md`](docs/models.md)).
3. **Enriches** (optional): table/figure summarization, cross-page table merging,
   chart extraction, key-value extraction, page classification.
4. **Extracts structured data** against a JSON schema with citation tracking.
5. **Returns** a single `ParsedDocument` (pages, fragments, tables, structured
   outputs, usage).

### Accepted inputs

| MIME type | Notes |
|---|---|
| `application/pdf` | Multi-page |
| `image/png` | Single page |
| `image/jpeg`, `image/jpg` | Single page |
| `image/heif`, `image/heic` | Single page (requires `pillow-heif`) |

Anything else is rejected at ingest with a clear error.

See [`docs/pipeline.md`](docs/pipeline.md) for the full DAG.

---

## Setup

### 1. Install

```bash
git clone <your-fork-url> stern-open-ingest
cd stern-open-ingest
pip install -e ".[dev]"      # runtime + pytest/ruff/black
```

For the GPU OCR path also install torch + transformers (and vLLM on a GPU box):

```bash
pip install -e ".[cpu]"      # CPU box (no GPU OCR)
# pip install -e ".[gpu]"     # Linux CUDA box
# pip install vllm            # additionally required for dots-ocr on a GPU box
```

Sanity check:

```bash
python -c "from tensorlake_docai.pipeline.file_converter import normalize_file_type_and_upload; print('ok')"
pytest tests/ -q
```

### 2. Configure keys

```bash
cp .env.example .env
$EDITOR .env
set -a; source .env; set +a
```

Only the features you use need keys:

| Feature | Keys |
|---|---|
| `ocr_model="dots-ocr"` | none — needs a CUDA GPU host |
| VLM enrichment | one of `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` |
| `s3://` file inputs | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `S3_BUCKET_NAME` |

Missing keys silently disable the dependent feature; the rest of the pipeline
keeps running.

### 3. Run

```bash
python examples/parse_pdf.py --file my.pdf --ocr-model dots-ocr --local
```

Results land in `./debug/` (`document.json` plus one markdown file per chunk).

Full walkthrough — every `ParseRequest` flag, VLM enrichment, page
classification, and file input mode — is in [`docs/running.md`](docs/running.md).

---

## Dev workflow

After editing any Python under `src/`, `tests/`, or `examples/`:

```bash
black src/ tests/ examples/
ruff check src/ tests/ examples/
pytest tests/ -q
```

Fix `ruff` errors before handing work back.

All functions live under `src/tensorlake_docai/...` and use absolute imports
(`from tensorlake_docai.… import …`). Keep that package layout intact when adding
modules — moving files out of the package breaks the bundled imports.
`src/workflow.py` sits one level above the package and imports every task to
register it with the runner.

---

## Extending the pipeline

- **Bring your own OCR backend.** Drop a file in `src/tensorlake_docai/ocr/`,
  register it in `ocr/__init__.py`, widen the `ocr_model` enum in
  `pipeline/api.py`, and import it in `workflow.py`. Calling `route_after_ocr`
  at the end of your task wires it into table merging, VLM enrichment, and the
  unified output format. Step-by-step:
  [`src/tensorlake_docai/ocr/README.md`](src/tensorlake_docai/ocr/README.md).
  This is the path we'll use to add a lighter OCR backend (see `CLAUDE.md`).
- **Add a VLM enrichment pass.** Table/figure summarization, chart extraction,
  and key-value extraction live in `src/tensorlake_docai/vlm/cloud.py` as batched
  passes over the document — add another by following the same shape.

The `dots-ocr` backend doubles as the reference implementation for serving a GPU
model: vLLM engine setup, model caching, two-stage classification → extraction,
and masked-region retries live in `src/tensorlake_docai/ocr/dots_ocr.py` and
`figure_ocr.py`.

---

## Project layout

```
stern-open-ingest/
├── examples/                    # parse_pdf.py
├── src/
│   ├── workflow.py              # imports every task to register it
│   └── tensorlake_docai/
│       ├── pipeline/            # file_converter, routing, output_formatter, api
│       ├── ocr/                 # dots_ocr, figure_ocr (BYO-OCR registry)
│       ├── vlm/                 # VLM summarization, grounding, chart extraction
│       ├── extraction/          # output chunking and key-value extraction helpers
│       ├── tables/              # cross-page merging, cell grounding, correction
│       ├── postprocess/         # header correction, formatter, output cleaner
│       ├── models/              # ParseResult, PageLayout, etc.
│       ├── providers/           # LLM client wrappers
│       └── prompts/             # prompt templates
├── docs/                        # pipeline.md, models.md, running.md
├── pyproject.toml
└── .env.example
```

---

## Acknowledgments

Forked from Tensorlake's [OpenIngest](https://github.com/tensorlakeai/openingest)
(Apache-2.0). The OCR stack stands on:

- **[DotsOCR](https://github.com/rednote-hilab/dots.mocr)** (rednote-hilab) — the
  layout + OCR model behind `dots-ocr`.
- **[Ovis2.5-9B](https://huggingface.co/AIDC-AI/Ovis2.5-9B)** (AIDC-AI) — the VLM
  used for figure OCR.
- **[vLLM](https://github.com/vllm-project/vllm)** — the inference server in the
  GPU OCR container.
- **[jdeskew](https://github.com/phamquiluan/jdeskew)** — skew correction during
  page preprocessing.
