"""Benchmark harness entrypoint — PLACEHOLDER, NOT YET IMPLEMENTED.

This file marks the planned interface for the model-output benchmark. The
runnable harness and scoring logic are deferred follow-up work pending model
API keys (OpenAI `gpt-5-mini`) that are not yet wired. See ``benchmarks/README.md``.

Planned interface (subject to change when actually built):

    python -m benchmarks.run [--golden DIR] [--out benchmarks/results/RUN_ID]

For each document folder under ``benchmarks/golden/``:

    1. Parse the redacted ``source_file`` (from ``meta.yaml``) through the
       pipeline to produce layout-aware markdown.
    2. For each question in ``questions.json``, feed the produced markdown plus
       the question ``prompt`` to the ``gpt-5-mini`` stand-in LLM and capture
       its presence + value answer.
    3. Normalize the stand-in's answer and the golden ``value`` by ``type``
       (date -> ISO, currency -> number, string -> casefold/trim) and score:
         - field accuracy (headline): presence + value correctness
         - text fidelity (tripwire): produced markdown vs committed baseline
    4. Write per-document + aggregate scores to ``benchmarks/results/`` and
       compare against committed ``benchmarks/baselines/``.

None of the above is implemented. Do not import or invoke this module yet.
"""

raise NotImplementedError(
    "benchmarks/run.py is a placeholder. The harness and scoring are deferred "
    "follow-up work pending gpt-5-mini API keys. See benchmarks/README.md."
)
