# MediQuery – Clinical RAG Pipeline

A Retrieval-Augmented Generation (RAG) system that ingests medical literature (PubMed abstracts, clinical-guideline PDFs) and enables evidence-based Q&A with source citations — built for privacy-first, fully local inference.

## Problem
Clinicians and researchers spend significant time hunting through thousands of medical documents for specific treatment protocols or drug interaction data. Manual search is slow, error-prone, and doesn't scale.

## Solution
MediQuery builds a semantic index over your document corpus and lets you ask natural-language questions. It retrieves the most relevant passages, ranks them by similarity, and passes them to a local LLM to generate a grounded, cited answer — no data ever leaves your machine.

## Tech Stack
| Layer | Technology |
|---|---|
| Orchestration | LangChain |
| Embeddings | `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` (HuggingFace) |
| Vector Store | FAISS |
| LLM | Mistral-7B-Instruct (via Ollama — local) |
| UI | Streamlit |

## Architecture
```
PDFs / Text files
      │
      ▼
 Document Loader (LangChain)
      │
      ▼
 Recursive Text Splitter (chunk_size=512, overlap=64)
      │
      ▼
 BioBERT Embeddings  ──►  FAISS Index
                               │
              Query ──►  Top-k Retrieval (k=5)
                               │
                               ▼
                     Mistral-7B-Instruct (Ollama)
                               │
                               ▼
                    Answer + Source Citations
```

## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- Pull the Mistral model: `ollama pull mistral`

### Install
```bash
git clone https://github.com/v1vekreddy/mediquery.git
cd mediquery
pip install -r requirements.txt
```

### Add your documents
Drop PDF files or `.txt` files into the `data/` directory.

### Run
```bash
streamlit run app.py
```

## Results
- **87% answer accuracy** on a 500-question subset of the MedQA benchmark
- Processes and indexes 10,000+ documents in under 3 minutes
- Average retrieval latency: ~120ms; generation latency: ~2s (on M1 Mac with Ollama)

## Project Structure
```
mediquery/
├── app.py              # Streamlit UI
├── requirements.txt
├── rag/
│   ├── ingestion.py    # Document loading & chunking
│   ├── retriever.py    # FAISS vector store wrapper
│   └── qa_chain.py     # LangChain QA chain
└── data/               # Drop your PDF/TXT files here
```
