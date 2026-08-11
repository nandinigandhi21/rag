# Amaya — Air-Gapped RAG & Cognitive Research System

Amaya is an enterprise-grade, privacy-preserving Retrieval-Augmented Generation (RAG) platform designed for **100% offline, air-gapped operations**. It combines layout-aware PDF ingestion, hybrid semantic/keyword search with cross-encoder reranking, local LLM generation via Ollama, and automated executive PowerPoint (.pptx) presentation deck generation.

---

## 🌟 Key Features

* **🔒 100% Air-Gapped & Offline Security**: Forces strict offline behavior (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `NO_PROXY=*`). Zero telemetry or external API calls.
* **📄 Layout-Aware Document Ingestion**: Powered by **Docling**, preserving complex multi-page tables, headers, breadcrumbs, formulas, and OCR for scanned PDFs.
* **🔍 Two-Stage Hybrid Retrieval**:
  * **Stage 1 (Fusion)**: Combines dense vector similarity (ChromaDB / Cosine) and sparse keyword search (BM25) via Reciprocal Rank Fusion (RRF).
  * **Stage 2 (Reranking)**: Uses a local Cross-Encoder (`ms-marco-MiniLM-L6-v2`) for high-precision context relevance.
* **🧠 Local LLM Generation**: Integrates with local **Ollama** instances (`qwen2:7b`, `qwen2.5:3b`, etc.) for real-time streaming answers with exact source citations.
* **📊 Executive Presentation Deck Generator**: Automatically distills research chat sessions into custom-themed PowerPoint (.pptx) decks.

---

## 🏗️ System Architecture

```
[ PDF / Documents ] ──► [ Docling Ingestion Engine ] ──► [ Layout & Table Chunking ]
                                                                 │
                                                                 ▼
[ Interactive UI (Streamlit) ] ◄── [ RAG Orchestrator ] ◄── [ Dual Vector & BM25 Store ]
            │                              │                     │
            ▼                              ▼                     ▼
[ Presentation Generator ]     [ Local Ollama LLM ]     [ Cross-Encoder Reranker ]
   (.pptx Export Engine)         (Qwen2 / Nomic-Embed)      (ms-marco-MiniLM-L6-v2)
```

---

## 📁 Repository Structure

```
amaya/
├── config.py                 # Centralized AppConfig (Pydantic Settings & Offline Flags)
├── gui.py                    # Streamlit Dashboard (Knowledge Studio, Research Lab, Decks)
├── ingestion_engine.py       # PDF Parsing & Chunking Engine (Docling + RapidOCR)
├── vector_engine.py          # Vector Engine (ChromaDB + BM25 + Cross-Encoder Reranker)
├── generation_engine.py      # LLM Generation Engine & Streaming RAG Orchestrator
├── presentation_engine.py    # Automated PowerPoint (.pptx) Deck Builder
├── presentation_themes.py    # Executive Presentation Color & Layout Themes
├── schema.py                 # Core Pydantic Schemas (Chunks, Results, Responses)
├── requirements.txt          # Python Dependencies
├── .gitignore                # Git Ignore Rules
├── models_cache/             # Offline Pre-Downloaded Models (HuggingFace/Docling/CrossEncoder)
├── results/                  # Staging for Ingested Chunks, Extracted Tables & Session Reports
└── app.log                   # System Log File
```

---

## 📦 Local Model Cache Directory (`models_cache/`)

The `models_cache/` directory contains pre-downloaded model weights for offline execution (OCR, layout parsing, formula recognition, figure classification, and cross-encoder reranking).

### 📁 Directory Structure (Model Folder Names Only)

```text
models_cache/
├── RapidOcr/
├── docling-project--CodeFormulaV2/
├── docling-project--TableFormerV2/
├── docling-project--docling-layout-heron/
├── docling-project--docling-models/
├── ds4sd--DocumentFigureClassifier/
├── ibm-granite--granite-docling-258M/
└── ms-marco-MiniLM-L6-v2/
```

#### Model Descriptions:
* **`RapidOcr/`**: Local OCR model weights used by Docling for extracting text from scanned PDF documents and images.
* **`docling-project--CodeFormulaV2/`**: Docling model for detecting and converting mathematical formulas into LaTeX/Markdown format.
* **`docling-project--TableFormerV2/`**: Table structure model for extracting multi-page, complex data tables.
* **`docling-project--docling-layout-heron/`**: Layout segmentation model for identifying reading order, headers, and document hierarchy.
* **`docling-project--docling-models/`**: Core Docling parser assets and metadata.
* **`ds4sd--DocumentFigureClassifier/`**: Document element classification model for identifying figures, charts, and diagrams.
* **`ibm-granite--granite-docling-258M/`**: IBM Granite lightweight VLM model for layout & visual parsing.
* **`ms-marco-MiniLM-L6-v2/`**: Cross-Encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) used for two-stage RAG query reranking.

---

### 📥 Commands to Download `models_cache` Models

Run these commands on an **online machine** prior to transferring the files to your air-gapped laptop:

#### Option 1: Using `huggingface-cli` (Command Line)

```cmd
# Create the local cache directory
mkdir models_cache

# Download Reranker Model
huggingface-cli download cross-encoder/ms-marco-MiniLM-L-6-v2 --local-dir models_cache/ms-marco-MiniLM-L6-v2

# Download Docling & Layout Models
huggingface-cli download docling-project/TableFormerV2 --local-dir models_cache/docling-project--TableFormerV2
huggingface-cli download docling-project/CodeFormulaV2 --local-dir models_cache/docling-project--CodeFormulaV2
huggingface-cli download docling-project/docling-layout-heron --local-dir models_cache/docling-project--docling-layout-heron
huggingface-cli download ds4sd/DocumentFigureClassifier --local-dir models_cache/ds4sd--DocumentFigureClassifier
huggingface-cli download ibm-granite/granite-docling-258M --local-dir models_cache/ibm-granite--granite-docling-258M
```

#### Option 2: Python Script (`download_models.py`)

Save and run the following script on an internet-connected machine:

```python
from pathlib import Path
from huggingface_hub import snapshot_download

cache_dir = Path("models_cache")

models = {
    "ms-marco-MiniLM-L6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "docling-project--TableFormerV2": "docling-project/TableFormerV2",
    "docling-project--CodeFormulaV2": "docling-project/CodeFormulaV2",
    "docling-project--docling-layout-heron": "docling-project/docling-layout-heron",
    "ds4sd--DocumentFigureClassifier": "ds4sd/DocumentFigureClassifier",
    "ibm-granite--granite-docling-258M": "ibm-granite/granite-docling-258M"
}

for local_name, repo_id in models.items():
    print(f"Downloading {repo_id} -> {local_name}...")
    snapshot_download(repo_id=repo_id, local_dir=cache_dir / local_name)

print("All models successfully cached in models_cache/")
```

---

## ⚙️ Ollama Setup & Model Downloads

### 1. Start Ollama Service

```cmd
ollama serve
```

### 2. Download Recommended Models (`ollama pull`)

Run these `ollama pull` commands on an online machine (or directly on your machine if connected):

```cmd
# Download Default LLM Model (Qwen 2.5 3B)
ollama pull qwen2.5:3b

# Download Default Embedding Model (Nomic Embed Text)
ollama pull nomic-embed-text:latest
```

### 3. Verify Installed Ollama Models

```cmd
ollama list
```

*Expected output includes:*
* `qwen2.5:3b`
* `nomic-embed-text:latest`

### 4. Transferring Models to Air-Gapped Laptop

If preparing an offline machine, copy downloaded Ollama models from the source machine's Ollama directory:
* **Windows**: `%USERPROFILE%\.ollama\models`
* **Linux**: `~/.ollama/models`
* **macOS**: `~/.ollama/models`

---

## 💡 Supported Model Examples (LLM & Embeddings)

You can substitute models in `config.py` (`OLLAMA_LLM_MODEL` and `OLLAMA_EMBED_MODEL`) or environment variables.

### 🧠 1. LLM Models for Generation (`OLLAMA_LLM_MODEL`)

| Model Name | Ollama Pull Command | VRAM / RAM Size | Recommended Use Case |
| :--- | :--- | :--- | :--- |
| **`qwen2.5:3b`** *(Default)* | `ollama pull qwen2.5:3b` | ~2.0 GB | **Recommended default for laptops**. Fast inference with high accuracy. |
| **`qwen2.5:7b`** | `ollama pull qwen2.5:7b` | ~4.7 GB | High-precision reasoning, complex document synthesis, and accurate coding. |
| **`qwen2:7b`** | `ollama pull qwen2:7b` | ~4.4 GB | Enterprise RAG performance with long-context understanding. |
| **`llama3.2:3b`** | `ollama pull llama3.2:3b` | ~2.0 GB | Ultra-fast lightweight model from Meta for quick mobile/laptop QA. |
| **`llama3.1:8b`** | `ollama pull llama3.1:8b` | ~4.7 GB | Strong general-purpose instruction following and high accuracy. |
| **`mistral:7b`** | `ollama pull mistral:7b` | ~4.1 GB | Balanced performance for general text generation and summarization. |
| **`phi3.5:latest`** | `ollama pull phi3.5:latest` | ~2.2 GB | Microsoft high-efficiency reasoning model with low memory footprint. |
| **`gemma2:2b`** | `ollama pull gemma2:2b` | ~1.6 GB | Low-memory laptops with minimal RAM/VRAM. |
| **`gemma2:9b`** | `ollama pull gemma2:9b` | ~5.4 GB | High quality Google Gemma 2 model for complex analytical queries. |
| **`deepseek-r1:1.5b`** | `ollama pull deepseek-r1:1.5b` | ~1.1 GB | Lightweight reasoning model with explicit step-by-step thinking. |
| **`deepseek-r1:7b`** | `ollama pull deepseek-r1:7b` | ~4.7 GB | Advanced Chain-of-Thought (CoT) reasoning for deep document analysis. |
| **`deepseek-r1:8b`** | `ollama pull deepseek-r1:8b` | ~4.9 GB | Llama-3.1-based DeepSeek R1 reasoning variant. |

---

### 🔍 2. Embedding Models (`OLLAMA_EMBED_MODEL`)

| Model Name | Ollama Pull Command | Embedding Dimension | Recommended Use Case |
| :--- | :--- | :--- | :--- |
| **`nomic-embed-text:latest`** *(Default)* | `ollama pull nomic-embed-text:latest` | 768 | **Recommended default**. 8192-token context window for long RAG chunks. |
| **`mxbai-embed-large:latest`** | `ollama pull mxbai-embed-large:latest` | 1024 | State-of-the-art retrieval accuracy and vector search precision. |
| **`all-minilm:latest`** | `ollama pull all-minilm:latest` | 384 | Ultra-fast lightweight embeddings for low-power hardware. |
| **`bge-m3:latest`** | `ollama pull bge-m3:latest` | 1024 | Multilingual support, multi-function dense embedding. |
| **`snowflake-arctic-embed:latest`** | `ollama pull snowflake-arctic-embed:latest` | 1024 | Enterprise-grade retrieval performance. |
| **`bge-large:latest`** | `ollama pull bge-large:latest` | 1024 | Deep semantic matching for scientific and technical documentation. |

---

### ⚙️ How to Switch Models in Configuration

To switch to a different model (e.g., `deepseek-r1:7b` and `mxbai-embed-large:latest`), update `config.py`:

```python
# In config.py:
OLLAMA_LLM_MODEL: str = "deepseek-r1:7b"
OLLAMA_EMBED_MODEL: str = "mxbai-embed-large:latest"
```

Or set environment variables before running the application:

```cmd
set DOCLING_PRO_OLLAMA_LLM_MODEL=deepseek-r1:7b
set DOCLING_PRO_OLLAMA_EMBED_MODEL=mxbai-embed-large:latest
```


---

## ⚙️ Configuration Management (`config.py`)

Amaya automatically configures offline environment flags upon startup. Settings can be inspected or overridden in `config.py` or via `DOCLING_PRO_` prefixed environment variables.

### Key Settings:

| Setting | Default Value | Description |
| :--- | :--- | :--- |
| `OFFLINE_MODE` | `True` | Disables all outbound internet calls (`HF_HUB_OFFLINE=1`). |
| `USE_OLLAMA` | `True` | Routes LLM queries and embeddings to the local Ollama API. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Endpoint for the Ollama server. |
| `OLLAMA_LLM_MODEL` | `qwen2.5:3b` | Active LLM model name in Ollama. |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text:latest` | Active embedding model name in Ollama. |
| `MODELS_CACHE` | `BASE_DIR / "models_cache"` | Local directory for cached models. |
| `RERANKER_MODEL_PATH` | `MODELS_CACHE / "ms-marco-MiniLM-L6-v2"` | Path to the Cross-Encoder model. |

---

## 💻 Running the Application

1. **Activate Environment & Run Streamlit App**:
   ```cmd
   amaya_env\Scripts\activate
   python gui.py
   ```
   *Or using Streamlit directly:*
   ```cmd
   streamlit run gui.py
   ```

2. Open your web browser at `http://localhost:8501`.

---

## 🖥️ User Workflow Guide

### 1. Knowledge Engineering Studio (Document Ingestion)
- **Upload Document**: Drag and drop PDF files.
- **Configure Parameters**: Toggle OCR, Formula Enrichment, Table Extraction Mode (`accurate` / `fast`), and Chunking Strategy (`hybrid` / `hierarchical`).
- **Process**: Click **Process Document**. The system extracts markdown, saves structured tables as `.csv`, and builds indexed chunks.

### 2. Cognitive Research Lab (Interactive RAG Chat)
- **Select Strategy**: Choose between ChromaDB Persistent Vector Search or Direct Batch Search.
- **Select Persona**: Standard, Strict Auditor, Synthesizer, or Custom System Instructions.
- **Ask Questions**: Receive streaming responses grounded in your uploaded documents, complete with expandable citation cards and exact page numbers.

### 3. Executive Presentation Deck (PowerPoint Export)
- Select a visual theme (e.g., *Modern Corporate*, *Midnight Tech*, *Nordic Light*).
- Click **Generate Presentation Deck** to export research summaries into a polished `.pptx` presentation.

---

## 🛠️ Troubleshooting & Support

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| `ModuleNotFoundError` | Virtual environment not activated. | Run `amaya_env\Scripts\activate` before launching. |
| `HTTP 404 / Model Not Found` | Ollama model name typo in `config.py`. | Verify model names with `ollama list` and update `config.py`. |
| `ConnectionError` | Ollama server is not running. | Run `ollama serve` in a separate command window. |
| `HFValidationError / OSError` | Model folder name mismatch in `models_cache`. | Ensure folder name in `models_cache` matches `config.RERANKER_MODEL_PATH`. |

---

## 🛡️ Privacy & Compliance Statement

Amaya processes **all text, vector embeddings, and LLM inference strictly within local hardware memory**. No data is transmitted to external cloud endpoints.
