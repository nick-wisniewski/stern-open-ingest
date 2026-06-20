# OCR model

This fork ships self-hosted OCR backends only. The cloud backends from upstream
(Azure Document Intelligence, AWS Textract, Google Gemini) have been removed.
Select the backend via `ocr_model` on the `ParseRequest` (it defaults to
`dots-ocr`).

> Gemini is still available as an LLM/VLM **provider** (e.g. for table merging
> and key-value extraction) — it is just no longer an OCR backend.

## `dots-ocr`

| Capability | Support |
|---|---|
| Native PDF | converts pages to images first |
| Forms / key-value | ✅ |
| Barcodes | no dedicated decoding pass |
| Custom figure-OCR prompt | no |
| Hardware | CUDA GPU |

`dots-ocr` runs [DotsOCR](https://github.com/rednote-hilab/dots.mocr) (layout +
OCR) plus [Ovis2.5-9B](https://huggingface.co/AIDC-AI/Ovis2.5-9B) (figure OCR)
on a CUDA-equipped host. The full serving setup lives in the repo: vLLM engine,
two-stage Ovis figure OCR, and masked-region retries. A request that asks for
`dots-ocr` without a GPU fails at task start with a `RequestError`.

> **Direction:** per [`CLAUDE.md`](../CLAUDE.md), we are evaluating a lighter OCR
> backend (e.g. PaddleOCR-VL) added under `src/tensorlake_docai/ocr/` via the
> bring-your-own-OCR path, plus a born-digital-first CPU text path ahead of the
> GPU OCR worker. Check the live registrations in `src/tensorlake_docai/ocr/`
> for what is actually wired up.

## Required env vars

See [`.env.example`](../.env.example). `dots-ocr` needs no API keys — just a
CUDA host. Weights are pulled from Hugging Face Hub on first cold-start.

## `paddle-ocr-vl`

| Capability | Support |
|---|---|
| Native PDF | converts selected pages to images first |
| Forms / key-value | layout regions feed the shared VLM path |
| Barcodes | no dedicated decoding pass |
| Custom figure-OCR prompt | no |
| Hardware | CUDA GPU plus local PaddleOCR-VL recognition server |

`paddle-ocr-vl` is a non-default backend for PaddleOCR-VL validation. The worker
uses the PaddleOCR client locally and points its multimodal recognition step at a
local vLLM/SGLang-style server. Configure that server with:

- `PADDLE_OCR_VL_SERVER_URL` (default `http://127.0.0.1:8118/v1`)
- `PADDLE_OCR_VL_REC_BACKEND` (default `vllm-server`)
- `PADDLE_OCR_VL_MEMORY_IN_GB` (default `24`)
- `PADDLE_OCR_VL_GPU_MODELS` (default `L4,A10G`)

A request that asks for `paddle-ocr-vl` without CUDA fails at task start with a
`RequestError`. The backend is import-safe in CPU environments, but it is not
intended as a CPU fallback.

## Figure OCR

DotsOCR outputs are post-processed by Ovis2.5-9B running on a separate GPU
container. The Ovis pass classifies each cropped figure (`BARCODE`, `CHART`,
`DIAGRAM`, `FORM`, `TABLE`, `OTHER`) and extracts content with a type-specific
prompt.
