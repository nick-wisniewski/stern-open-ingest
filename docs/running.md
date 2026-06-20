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
| `ocr_model="paddle-ocr-vl"` | `ENABLE_GPU_OCR_TASKS=1`, CUDA GPU host, local PaddleOCR-VL recognition server |
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

## 2c. GPU pipeline (`paddle-ocr-vl`)

`paddle-ocr-vl` uses a local PaddleOCR client for document parsing and points the
multimodal recognition step at a local vLLM/SGLang-style server. Run both the
server and the `--local` Python process on a CUDA-equipped host.

**Task registration.** Enable GPU task imports before running the workflow:

```bash
export ENABLE_GPU_OCR_TASKS=1
```

**Start the recognition server.** Paddle's docs recommend the dedicated server
image or the `paddleocr genai_server` CLI. A vLLM server launch looks like:

```bash
paddleocr genai_server \
  --model_name PaddleOCR-VL-1.6-0.9B \
  --host 0.0.0.0 \
  --port 8118 \
  --backend vllm
```

Set the client env vars to match the server:

```bash
export PADDLE_OCR_VL_SERVER_URL=http://127.0.0.1:8118/v1
export PADDLE_OCR_VL_REC_BACKEND=vllm-server
```

Optional sizing knobs:

```bash
export PADDLE_OCR_VL_MEMORY_IN_GB=24
export PADDLE_OCR_VL_GPU_MODELS=L4,A10G
```

**One-page smoke test.**

```bash
python examples/parse_pdf.py \
  --file sample.pdf \
  --ocr-model paddle-ocr-vl \
  --pages 1 \
  --local
```

The CLI preflights CUDA and `PADDLE_OCR_VL_SERVER_URL/models` before rendering
pages. Output is written to `./debug/document.json` and `./debug/document.md`.

---

## 2d. Modal smoke test from a Mac

If you are on a Mac, use Modal to get the CUDA host for the Paddle smoke test.
This is still a smoke harness, not the production receiver/queue/webhook
deployment.

Install and authenticate Modal locally:

```bash
python -m pip install modal
modal setup
```

Run one page on a Modal L4 GPU:

```bash
modal run examples/modal_paddle_smoke.py \
  --file sample.pdf \
  --pages 1
```

What the Modal function does:

- Builds a GPU image with PaddleOCR-VL, vLLM server dependencies, and this repo.
- Starts `paddleocr genai_server` inside the Modal GPU container.
- Waits for `http://127.0.0.1:8118/v1/models`.
- Runs the existing Open Ingest local runner inside that same GPU container with
  `ocr_model="paddle-ocr-vl"`.
- Returns the parsed document to your Mac and writes
  `./debug/modal-paddle-smoke/document.json` plus `document.md`.

Optional arguments:

```bash
modal run examples/modal_paddle_smoke.py \
  --file sample.pdf \
  --pages 1,2 \
  --out debug/my-paddle-smoke \
  --table-output-mode markdown
```

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
- GPU OCR runs in-process too; you need a CUDA GPU on the host.

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
