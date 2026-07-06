# Model-output benchmark suite

This directory holds the **scaffolding** for a benchmark that measures how well
this service's parse output serves its *actual* downstream consumer — an
analysis agent (OpenAI `gpt-5-mini`), not a human reader.

> **Status: skeleton only.** The directory structure, conventions, templates,
> and docs are in place so a human can drop redacted golden documents into
> labeled slots. The runnable harness and scoring are **not built yet** — see
> [Deferred / not yet built](#deferred--not-yet-built).

## Purpose

The pipeline turns insurance-policy PDFs/images into layout-aware markdown plus
structured data. That output is consumed by a downstream analysis agent, not by
a person. So the question this benchmark answers is **not** "how faithful is the
markdown to the original?" — it is:

> **Given the markdown we produced, can the downstream agent recover the right
> answers about the policy?**

Exact markdown fidelity is only a directional tripwire. **Field accuracy is the
headline metric.**

## Metric design

For each golden document we freeze a **question set**. Every question has two
parts:

- **presence** — "does the policy have X *at all*?" (`present: true|false`)
- **value** — "if so, what is it?" (`value`, meaningful only when `present`)

Scoring (deferred — see below) will work like this: the produced markdown for a
document is fed to a **stand-in LLM** (`gpt-5-mini`, matching the real
downstream consumer) along with each question's `prompt`. The stand-in's answer
is compared against the frozen `present` / `value` labels after normalizing by
`type`.

- **Field accuracy (headline):** fraction of questions where the stand-in gets
  both presence and (when present) the normalized value correct.
- **Text fidelity (tripwire only):** a directional signal (e.g. similarity of
  produced markdown to a committed baseline). Used to catch large regressions,
  **not** as a primary score. A document can score perfectly on field accuracy
  with imperfect text fidelity — that is acceptable and expected.

### Why a stand-in LLM instead of exact string match?

The real consumer is an LLM that reasons over the markdown. A rigid string match
would penalize harmless formatting differences and reward brittle output. Using
`gpt-5-mini` as a stand-in judge mirrors the production reader, so the score
tracks what actually matters: can the agent get the answer out.

## Directory layout

```
benchmarks/
├── README.md                     # this file
├── run.py                        # PLACEHOLDER — planned harness entrypoint (not implemented)
├── golden/                       # the corpus: one folder per golden document
│   └── EXAMPLE-state-farm-dec-page/
│       ├── README.md             # note: this is a template; copy it per real doc
│       ├── meta.yaml             # document metadata (schema below)
│       ├── questions.json        # frozen question set (schema below)
│       └── source.PLACEHOLDER.txt# where the redacted source.pdf/.png goes
├── baselines/                    # committed scores / reference outputs (per doc)
│   └── .gitkeep
└── results/                      # gitignored run outputs (never committed)
    └── .gitkeep
```

## How to add a new golden document

1. **Make a folder** under `golden/` named for the document, e.g.
   `golden/acme-homeowners-dec-2025/`. Copy the `EXAMPLE-state-farm-dec-page/`
   folder as your starting point.
2. **Drop the redacted source file** into the folder as `source.pdf`,
   `source.png`, or `source.jpg`. It **must be redacted / synthetic** — see
   `provenance` in the meta schema. Delete the `source.PLACEHOLDER.txt` note
   once the real file is in place.
3. **Fill `meta.yaml`** — see the schema below. Set `source_file` to match the
   filename you added.
4. **Fill `questions.json`** — see the schema below. Start from the frozen
   starter question set and add document-specific questions as pure data edits.
5. That's it. Adding or changing a question is a **pure data edit** to
   `questions.json`; no code changes are needed.

### `meta.yaml` schema

```yaml
carrier: State Farm                # insurance carrier name
source_policy_id: 3800             # stern-liability policy id this doc was derived from (traceability)
provenance: synthetic              # synthetic | deidentified | consented-real
doc_type: declaration_page         # human-curated document type
source_file: source.pdf            # the redacted document filename (pdf|png|jpg)
ocr_model: paddle-ocr-vl           # OCR backend the doc is benchmarked against
pipeline_flags:
  key_value_extraction: true       # pipeline feature toggles used for this doc
  table_merging: true
notes: ""                          # free-text notes (edge cases, known quirks)
```

- **`provenance`** — how the document was sourced. `synthetic` = fabricated;
  `deidentified` = a real doc with PII removed; `consented-real` = a real doc we
  have permission to use. Never commit un-redacted real customer data.
- **`source_policy_id`** — links the golden doc back to a `stern-liability`
  policy for traceability, even when the content is synthetic.

### `questions.json` schema

```json
{
  "source_policy_id": 3800,
  "carrier": "State Farm",
  "questions": [
    { "key": "effective_date", "prompt": "Does the policy have an effective date? If so, what is it?", "present": true, "value": "2025-03-01", "type": "date" }
  ]
}
```

Each entry in `questions` is one presence + value question:

| field     | meaning |
|-----------|---------|
| `key`     | stable machine identifier for the question (dotted keys like `coverage_limit.personal_property` group related fields). |
| `prompt`  | the natural-language question fed to the stand-in LLM. Phrased as presence + value: "Does the policy have X? If so, what is it?" |
| `present` | is the fact in the document **at all**? `true` if the doc contains it, `false` if it is genuinely absent. Include at least one `false` case per doc so the "absent" path is exercised. |
| `value`   | the correct answer **when present**. `null` when `present` is `false`. |
| `type`    | how the value is normalized before comparison (see below). |

#### `type` normalization

The `type` tells the (deferred) scorer how to normalize both the stand-in's
answer and the golden `value` before comparing:

| `type`     | normalization |
|------------|---------------|
| `date`     | parse to ISO `YYYY-MM-DD`. |
| `currency` | strip symbols/commas, parse to a number (e.g. `"$1,000,000"` → `1000000`). |
| `string`   | casefold + trim whitespace. |

Add new types here (and in the scorer, once built) as documents demand them.

## Deferred / not yet built

The following are **intentionally out of scope** for this scaffolding change and
are follow-up work **pending model API keys not yet wired**:

- **`run.py` harness** — the runnable entrypoint that parses each golden doc
  through the pipeline and collects produced markdown. `run.py` in this dir is a
  placeholder docstring only.
- **Scoring logic** — feeding produced markdown + prompts to the `gpt-5-mini`
  stand-in, normalizing by `type`, and computing field accuracy + the
  text-fidelity tripwire.
- **CI integration** — running the benchmark on PRs and comparing against
  committed `baselines/`.
- **A separate Modal instance for CI** — an isolated deployment so benchmark
  runs don't contend with production traffic.

These depend on wiring up the OpenAI (`gpt-5-mini`) API key and are tracked
separately. Do not build them here.
