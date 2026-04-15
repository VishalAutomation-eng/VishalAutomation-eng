# RAG Comparison Demo (Streamlit)

This demo compares three RAG strategies on user-uploaded PDFs:

- **Old RAG**: semantic chunking + embedding retrieval.
- **Page RAG**: page-level embedding indexing.
- **Graph RAG**: hybrid graph entity scoring + embedding retrieval.

## Pipeline

```text
PDF Upload
  -> Text Extraction (PyMuPDF / pypdf / pdfminer)
  -> Preprocessing
  -> Choose RAG Type (Old semantic / Page embedding / Graph hybrid)
  -> Qdrant Vector DB (in-memory mode) / Graph Store
  -> Embedding Retriever / Hybrid Search
  -> LLM (Qwen OpenAI-compatible API)
  -> Answer in Streamlit UI
```

## Run

```bash
pip install -r requirements.txt
export VISION_API_BASE="http://111.118.189.124:8100/v1"
export VISION_MODEL="/home/expadmin/Mayank/Qwen"
export EMBEDDING_API_BASE=""   # keep empty to use local open-source embeddings
export EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"
python -m streamlit run main.py
```

## Notes

- `VISION_API_BASE` and `VISION_MODEL` are editable from the Streamlit sidebar.
- This project uses `qdrant-client` in-memory mode as an advanced vector database for local demo setup.


## Troubleshooting

If you see **"No usable PDF text extractor found"**, install dependencies in the same Python environment running Streamlit:

```bash
python -m pip install -r requirements.txt
python -m pip install PyMuPDF pypdf pdfminer.six
```

Then restart the app:

```bash
python -m streamlit run main.py
```


### Interpreter mismatch check (important on macOS)

If packages are installed but app still says module missing, your `streamlit` command may be using another Python.

Run:

```bash
which python
python -m pip --version
which streamlit
python -m streamlit run main.py
```

Inside app sidebar, verify `sys.executable` matches your virtualenv path.


- Remote embedding API is optional. If provided and it fails, the app falls back to the local model automatically.
