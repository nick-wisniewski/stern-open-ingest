# Open Ingest Pipeline

This document is a visual reference for the current ingestion DAG. The pipeline
flows: **upload -> file validation -> OCR / layout -> optional retained enrichment ->
assembled output**. Each stage is a `@function`/`@cls` task; we run them
ourselves via the `--local` runner (see [`CLAUDE.md`](../CLAUDE.md)).

The service accepts PDF and image inputs only. Structured extraction is out of
scope for this repo; Rails owns that layer.

## Complete Flow

```mermaid
flowchart TD
    Start([User Upload]) --> FileConv[FILE_CONVERTOR<br/>normalize_file_type_and_upload]

    FileConv --> Validate{Supported MIME?<br/>PDF or image}
    Validate -->|reject| EndReject([Error: unsupported type])
    Validate -->|ok| ValidateQuota[Validate Quotas<br/>Count Pages]

    ValidateQuota --> Route{Routing<br/>Decision}
    Route -->|Need OCR| OCRSelect{OCR Model}

    OCRSelect -->|dots-ocr| DotsOCR[DotsOCRTask<br/>Layout + Markdown<br/>Figure OCR]
    DotsOCR --> HeaderOpt{Header<br/>Correction?}
    HeaderOpt -->|yes| HeaderCorr[Header Correction]
    HeaderOpt -->|no| PostOCR
    HeaderCorr --> PostOCR{Post-OCR Routing}

    PostOCR -->|Table merging| TableMerging[TableMerging]
    PostOCR -->|VLM tasks| VLMTask[VLMExtractionTask]
    PostOCR -->|No more work| OutOCR[OutputFormatter]

    TableMerging -->|VLM tasks| VLMTask
    TableMerging -->|Done| OutTable[OutputFormatter]

    VLMTask --> VLMProcess[VLM Batch Processing<br/>Key-Value Extraction]
    VLMProcess --> OutVLM[OutputFormatter]

    OutOCR --> Final[ParsedDocumentRef]
    OutTable --> Final
    OutVLM --> Final
    Final --> End([Return to Caller])

    classDef entry fill:#e1f5ff,stroke:#01579b,stroke-width:3px,color:#000
    classDef ocr fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    classDef vlm fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000
    classDef output fill:#ffebee,stroke:#b71c1c,stroke-width:3px,color:#000
    classDef decision fill:#fff9c4,stroke:#f57f17,stroke-width:2px,color:#000

    class FileConv entry
    class DotsOCR ocr
    class VLMTask,VLMProcess vlm
    class OutOCR,OutTable,OutVLM,Final output
    class Validate,Route,OCRSelect,HeaderOpt,PostOCR decision
```

## Retained VLM Enrichment

```mermaid
flowchart TD
    VLMStart[VLMExtractionTask Start] --> BatchCreate[Create Page Image Batches]
    BatchCreate --> Batch{For each batch}

    Batch --> KV[Key-Value Extraction]

    KV --> Update

    Update --> More{More batches?}
    More -->|yes| Batch
    More -->|no| Out[OutputFormatter]
```

## File Type Processing

```mermaid
flowchart TD
    Input[File Input] --> Validate{Supported MIME?}

    Validate -->|application/pdf| PDF[PDF Processing<br/>Multi-page OCR path]
    Validate -->|image/png<br/>image/jpeg / image/jpg<br/>image/heif / image/heic| IMG[Image Processing<br/>Single-page OCR path]
    Validate -->|anything else| Reject[Reject at ingest]

    classDef needsOCR fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    classDef reject fill:#ffebee,stroke:#b71c1c,stroke-width:2px,color:#000

    class PDF,IMG needsOCR
    class Reject reject
```
