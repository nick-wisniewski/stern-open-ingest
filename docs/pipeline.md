# Open Ingest Pipeline - Mermaid Flowchart

This document is a visual reference for the end-to-end ingestion DAG. The pipeline
flows: **upload → file conversion → routing → OCR / layout → optional VLM enrichment
→ optional structured extraction → assembled output**. Each stage is a
`@function`/`@cls` task; we run them ourselves via the `--local` runner (see
[`CLAUDE.md`](../CLAUDE.md)).

The diagrams below show the full graph and per-branch detail; use them to find the
function that owns a given behavior before diving into `src/tensorlake_docai/`.

A few terms used in the diagrams:

- **`skip_ocr`** — a per-request flag on `StructuredExtractionRequest`. When
  `True`, the pipeline skips OCR entirely and feeds page images directly to a
  vision LLM (Gemini) for structured extraction. Useful for visually dense
  documents where OCR misses layout cues.
- **Chunking strategies** — `none` (whole doc), `page` (one chunk per page),
  `section` / `fragment` (layout-driven), or `patterns` (regex boundaries). Set
  per-extraction via `StructuredExtractionRequest.chunking_strategy`.

## How to View/Edit This Diagram

1. **GitHub/GitLab**: Paste this code in a .md file - it will render automatically
2. **VS Code**: Install "Markdown Preview Mermaid Support" extension
3. **Online Editor**: https://mermaid.live/ - paste and edit in real-time
4. **Notion**: Use `/code` block and select "Mermaid"
5. **Draw.io**: Import Mermaid code directly
6. **Obsidian**: Native Mermaid support

---

## Complete Pipeline Flowchart

```mermaid
flowchart TD
    Start([User Upload]) --> FileConv[FILE_CONVERTOR<br/>normalize_file_type_and_upload]
    
    FileConv --> Validate{Supported MIME?<br/>PDF or image}
    Validate -->|reject| EndReject([Error: unsupported type])
    Validate -->|ok| ValidateQuota[Validate Quotas<br/>Count Pages]
    
    ValidateQuota --> Route{Routing<br/>Decision}
    
    Route -->|Skip OCR=True| VLMDirect[VLMExtractionTask<br/>Direct VLM Processing]
    
    Route -->|Need OCR| OCRSelect{OCR Model}
    OCRSelect -->|dots-ocr<br/>CUDA GPU| DotsOCR[DotsOCRTask<br/>Layout + Markdown<br/>Figure OCR + Barcodes]
    
    DotsOCR --> HeaderOpt{Header<br/>Correction?}
    HeaderOpt -->|Yes| HeaderCorr[Header Correction<br/>OpenAI GPT]
    HeaderOpt -->|No| PostOCR
    HeaderCorr --> PostOCR{Post-OCR<br/>Routing}
    
    PostOCR -->|No Further<br/>Processing| OutOCR[OutputFormatter]
    PostOCR -->|VLM Tasks<br/>Needed| VLMTask[VLMExtractionTask]
    PostOCR -->|Structured<br/>Extraction Only| SETask[StructuredExtraction]
    
    VLMTask --> VLMProcess[VLM Batch Processing:<br/>Table/Figure Summarization<br/>Page Classification<br/>Structured Extraction when skip_ocr]
    VLMDirect --> VLMProcess
    
    VLMProcess --> VLMRoute{More<br/>Processing?}
    VLMRoute -->|Structured<br/>Extraction| SEFromVLM[StructuredExtraction]
    VLMRoute -->|Done| OutVLM[OutputFormatter]
    
    SETask --> SEProcess[Structured Extraction:<br/>OpenAI / Claude / Gemini<br/>Chunking + citations]
    SEFromVLM --> SEProcess
    
    SEProcess --> OutSE[OutputFormatter]
    
    OutOCR --> Final[Final Output:<br/>ParsedDocumentRef]
    OutVLM --> Final
    OutSE --> Final
    
    Final --> End([Return to User])
    
    classDef entryPoint fill:#e1f5ff,stroke:#01579b,stroke-width:3px,color:#000
    classDef ocrModel fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    classDef vlmTask fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000
    classDef structTask fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#000
    classDef output fill:#ffebee,stroke:#b71c1c,stroke-width:3px,color:#000
    classDef decision fill:#fff9c4,stroke:#f57f17,stroke-width:2px,color:#000
    
    class FileConv entryPoint
    class DotsOCR ocrModel
    class VLMTask,VLMDirect,VLMProcess vlmTask
    class SETask,SEFromVLM,SEProcess structTask
    class OutOCR,OutVLM,OutSE,Final output
    class Validate,Route,OCRSelect,HeaderOpt,PostOCR,VLMRoute decision
```

---

## Simplified High-Level Flow

```mermaid
flowchart LR
    A[File Upload] --> B[File Convertor]
    B --> C{File Type?}
    
    C -->|PDF| E{OCR Model}
    C -->|Image| E
    
    E -->|dots-ocr| F4[DotsOCR on CUDA GPU worker]
    
    F4 --> G{Processing<br/>Needed?}
    
    G -->|VLM Tasks| H[VLM Processing<br/>• Table/Figure: OpenAI<br/>• Page Class: OpenAI<br/>• Structured Extraction skip_ocr: Gemini]
    G -->|LLM SE Only| J
    G -->|None| K
    
    H --> I{LLM SE<br/>Needed?}
    
    I -->|Yes| J[LLM Extraction<br/>OpenAI/Claude/Gemini<br/>• Chunking strategies<br/>• Citation tracking]
    I -->|No| K[Output Formatter]
    
    J --> K
    K --> L[API Response]
    
    classDef entry fill:#e1f5ff,stroke:#01579b,stroke-width:2px,color:#000
    classDef ocr fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    classDef vlm fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000
    classDef llm fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#000
    classDef out fill:#ffebee,stroke:#b71c1c,stroke-width:2px,color:#000
    classDef convert fill:#ffe0b2,stroke:#e65100,stroke-width:2px,color:#000
    
    class A,B entry
    class F4 ocr
    class H vlm
    class J llm
    class K,L out
```

---

## VLM Extraction Task Detail

```mermaid
flowchart TD
    VLMStart[VLMExtractionTask Start] --> BatchCreate[Create Image Batches<br/>Memory-efficient processing]
    
    BatchCreate --> Batch1{Batch Processing}
    
    Batch1 -->|For each batch| TableSum[Table Summarization<br/>OpenAI VLM<br/>Crop + Describe]
    Batch1 -->|For each batch| FigSum[Figure Summarization<br/>OpenAI VLM<br/>Crop + Describe]
    Batch1 -->|For each batch| PageClass[Page Classification<br/>OpenAI VLM<br/>Multi-label support]
    Batch1 -->|For each batch| SkipOCRSE[Structured Extraction<br/>skip_ocr=True<br/>Gemini VLM]
    
    TableSum --> UpdateElements[Update PageLayout<br/>Elements In-Place]
    FigSum --> UpdateElements
    PageClass --> UpdateElements
    SkipOCRSE --> StoreResults[Store in<br/>structured_outputs_by_page]
    
    UpdateElements --> MoreBatches{More<br/>Batches?}
    StoreResults --> MoreBatches
    
    MoreBatches -->|Yes| Batch1
    MoreBatches -->|No| Deferred{Deferred SE<br/>None-chunked?}
    
    Deferred -->|Yes| DeferredSE[Process Deferred<br/>Structured Extraction<br/>Document-level across pages]
    Deferred -->|No| TokenAgg[Aggregate Token Usage<br/>- Summarization tokens<br/>- Extraction tokens]
    
    DeferredSE --> TokenAgg
    TokenAgg --> VLMEnd[Route to Next Stage]
    
    classDef vlm fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000
    classDef process fill:#e3f2fd,stroke:#0277bd,stroke-width:2px,color:#000
    
    class VLMStart,VLMEnd vlm
    class TableSum,FigSum,PageClass,SkipOCRSE,UpdateElements,StoreResults,DeferredSE,TokenAgg process
```

---

## Structured Extraction Task Detail

```mermaid
flowchart TD
    SEStart[StructuredExtraction Start] --> CheckVLM{VLM outputs<br/>exist?}
    
    CheckVLM -->|Yes| SkipLLM[Skip LLM Extraction<br/>Use VLM results]
    CheckVLM -->|No| GetRequests[Get structured_extraction_requests]
    
    GetRequests --> FilterPages{Filter by<br/>page_classes?}
    
    FilterPages -->|Yes| FilterLogic[Filter pages matching<br/>specified classes]
    FilterPages -->|No| AllPages[Use all pages]
    
    FilterLogic --> ChunkStrategy{Chunking<br/>Strategy?}
    AllPages --> ChunkStrategy
    
    ChunkStrategy -->|None| WholeDoc[Process whole document]
    ChunkStrategy -->|Page| PerPage[Process per page]
    ChunkStrategy -->|Section| PerSection[Process per section]
    ChunkStrategy -->|Fragment| PerFragment[Process per fragment]
    ChunkStrategy -->|Pattern| ByPattern[Process by regex patterns]
    
    WholeDoc --> PrepText[Prepare Text Content]
    PerPage --> PrepText
    PerSection --> PrepText
    PerFragment --> PrepText
    ByPattern --> PrepText
    
    PrepText --> Citations{Citations<br/>enabled?}
    
    Citations -->|Yes| AddRefs[Add Citation References<br/>ref_id tracking]
    Citations -->|No| PageMarkers{Add page<br/>markers?}
    
    AddRefs --> CheckDense
    PageMarkers -->|Yes| AddMarkers[Add page boundary markers]
    PageMarkers -->|No| CheckDense
    AddMarkers --> CheckDense{Dense table<br/>content?}
    
    CheckDense -->|Yes CSV/Excel| SplitTable[Split Dense Table<br/>Row-based chunking<br/>Parallel processing]
    CheckDense -->|No| ModelCall[LLM API Call<br/>OpenAI/Claude/Gemini]
    
    SplitTable --> MergeResults[Merge Chunked Results]
    ModelCall --> ParseJSON[Parse JSON Response]
    MergeResults --> ParseJSON
    
    ParseJSON --> ResolveCite{Resolve<br/>citations?}
    
    ResolveCite -->|Yes| ValidateCite[Validate page references<br/>Clean citation data]
    ResolveCite -->|No| StoreResult[Store Structured Data<br/>by page/chunk key]
    
    ValidateCite --> StoreResult
    StoreResult --> MoreReq{More<br/>requests?}
    
    MoreReq -->|Yes| GetRequests
    MoreReq -->|No| AggTokens[Aggregate Token Usage<br/>Input + Output tokens]
    
    AggTokens --> SEEnd[Route to OutputFormatter]
    SkipLLM --> SEEnd
    
    classDef struct fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#000
    classDef process fill:#e3f2fd,stroke:#0277bd,stroke-width:2px,color:#000
    
    class SEStart,SEEnd struct
    class PrepText,AddRefs,AddMarkers,SplitTable,ModelCall,MergeResults,ParseJSON,ValidateCite,StoreResult,AggTokens process
```

---

## OCR Model Comparison Table

| `ocr_model` | Provider | Speed | Layout | Tables | Figures | Forms | Special Features |
|-------------|----------|-------|--------|--------|---------|-------|------------------|
| `dots-ocr` | DotsOCR on CUDA GPU worker | Fast | ✓ | ✓ | ✓ | ✓ | Custom prompts, Barcodes, two-stage Ovis figure OCR |

---

## File Type Processing Matrix

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
