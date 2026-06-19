# OCR model

This fork ships a single OCR backend: **`dots-ocr`**, the open-source engine we
run ourselves. The cloud backends from upstream (Azure Document Intelligence,
AWS Textract, Google Gemini) have been removed. Select the backend via
`ocr_model` on the `ParseRequest` (it defaults to `dots-ocr`).

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

## Figure OCR

DotsOCR outputs are post-processed by Ovis2.5-9B running on a separate GPU
container. The Ovis pass classifies each cropped figure (`BARCODE`, `CHART`,
`DIAGRAM`, `FORM`, `TABLE`, `OTHER`) and extracts content with a type-specific
prompt.
