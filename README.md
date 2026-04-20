# Semantic Search Engine (SciFact, BEIR)

This project implements a full semantic retrieval workflow on the SciFact dataset from BEIR, with multiple retrieval pipelines and both Python and C++ components:

- BERT bi-encoder retrieval
- BERT retrieval + cross-encoder reranking
- Word2Vec retrieval
- Custom HNSW ANN index implemented in C++ and exposed to Python with pybind11
- Streamlit app for interactive search

The repository includes both FAISS-based experiments and an HNSW-based replacement path for ANN search.

## What This Project Does

Given a natural-language query, the system retrieves relevant documents from SciFact using vector similarity. You can compare retrieval quality and speed across multiple methods:

- Pipeline A: BERT embeddings + ANN search
- Pipeline B: BERT first-stage retrieval + cross-encoder reranking
- Pipeline C: Word2Vec embeddings + ANN search

Evaluation is done with:

- Precision@k
- Recall@k

## Repository Structure

Top-level files:

- `app.py`: Streamlit UI for interactive semantic search
- `train.ipynb`: FAISS-based training, retrieval, and evaluation notebook
- `hnsw_retrieval.ipynb`: HNSW-based retrieval/evaluation notebook (replaces FAISS retrieval calls)
- `pyproject.toml`: Python dependencies and project metadata
- `main.py`: Placeholder entry script
- `README.md`: Project documentation

Data and artifacts:

- `datasets/scifact/`: SciFact corpus, queries, and qrels
- `bert_embeddings.npy`: Cached BERT document embeddings
- `w2v_embeddings.npy`: Cached Word2Vec document embeddings
- `w2v_model.bin`: Trained Word2Vec model
- `doc_ids.txt`: Document ID mapping for embeddings
- `corpus_data.pkl`: Serialized corpus IDs + document text for fast app loading

Custom ANN implementation:

- `my_algos/hnsw.cpp`: Standalone C++ HNSW implementation + demo main()
- `my_algos/hnsw_bind.cpp`: pybind11 module exposing HNSW to Python as `hnsw_index`
- `my_algos/setup.py`: Build script for native extension
- `my_algos/hnsw_retriever.py`: Python wrapper class around `hnsw_index`

## Retrieval Pipelines

### 1) BERT Bi-Encoder

- Model: `all-MiniLM-L6-v2`
- Embedding dimension: 384
- Fast first-stage retrieval from vector index

### 2) BERT + Cross-Encoder Reranker

- First stage: ANN retrieval for candidate generation
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Higher precision, slower inference than pure bi-encoder retrieval

### 3) Word2Vec

- Trained on SciFact text
- Embedding dimension: 300
- Lightweight classical baseline

### 4) Custom HNSW ANN (C++ via pybind11)

- Insertion and k-NN search implemented in C++
- Python API supports single and batch insert/search
- Hyperparameters used in notebooks include `M`, `ef_construction`, and `ef_search`

## Setup

### Prerequisites

- Python 3.10 to 3.12 (project requires `>=3.10,<3.13`)
- Windows, Linux, or macOS
- C++ toolchain for building extension:
	- Windows: MSVC Build Tools
	- Linux/macOS: GCC/Clang

### Option A: Using uv (recommended)

```powershell
uv sync
```

### Option B: Using venv + pip

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .
pip install sentence-transformers pandas tqdm
```

Note: Some notebook code imports `sentence_transformers`, `pandas`, and `tqdm`. Install them if they are not already present in your environment.

## Build the Custom HNSW Extension

From the project root:

```powershell
cd my_algos
python setup.py build_ext --inplace
```

Expected output is a compiled module in `my_algos/` (on Windows, a `.pyd` file such as `hnsw_index.cp310-win_amd64.pyd`).

## How To Run

### 1) Notebook Workflow (recommended first run)

Run notebooks in this order:

1. `train.ipynb`
	 - Downloads/loads SciFact
	 - Generates BERT and Word2Vec embeddings
	 - Builds FAISS indices
	 - Evaluates Precision@k and Recall@k
	 - Saves reusable artifacts (`bert_embeddings.npy`, `w2v_embeddings.npy`, `doc_ids.txt`, `corpus_data.pkl`)
2. `hnsw_retrieval.ipynb`
	 - Loads/builds custom HNSW indices
	 - Evaluates HNSW pipelines against SciFact qrels
	 - Compares FAISS vs HNSW and explores `ef_search` trade-off

### 2) Streamlit App

After artifact generation, start the app:

```powershell
streamlit run app.py
```

App features:

- Query input and top-k slider
- Selectable mode:
	- Bi-Encoder (fast)
	- Cross-Encoder reranked (higher precision)
- Expandable ranked result display with scores

## Key Implementation Notes

- `app.py` builds an in-memory FAISS `IndexFlatIP` from cached BERT embeddings.
- `corpus_data.pkl` is used by the app for fast corpus loading instead of reparsing BEIR each run.
- The C++ extension in `my_algos/hnsw_bind.cpp` returns squared Euclidean distances and integer indices.
- `my_algos/hnsw_retriever.py` wraps the extension with a document store so results are returned as `(document, distance)` tuples.

## Typical Workflow

1. Set up Python environment and dependencies.
2. Build the native HNSW extension.
3. Run `train.ipynb` to produce embeddings and corpus cache.
4. Run `hnsw_retrieval.ipynb` for HNSW experiments and evaluation.
5. Launch `app.py` with Streamlit for interactive querying.

## Troubleshooting

### Extension import error (`No module named hnsw_index`)

- Ensure build succeeded in `my_algos/`.
- Confirm the compiled module file exists in `my_algos/`.
- Run code from a working directory that can import from `my_algos`, or add that path to `sys.path`.

### CUDA or Torch issues

- If CUDA is unavailable, code falls back to CPU in both notebooks and app.
- If GPU wheels fail, install CPU-compatible torch packages for your environment.

### Missing NLTK resources

- The notebooks download `punkt` and `stopwords` automatically.
- If blocked by network policy, pre-download NLTK corpora in your environment.

## Future Improvements

- Add an end-to-end CLI entrypoint for indexing + retrieval + evaluation.
- Add automated tests for HNSW bindings and retrieval metric correctness.
- Persist HNSW graph to disk and reload instead of rebuilding in-memory.
- Unify FAISS and HNSW code paths behind a shared retrieval interface.

## License

No license file is currently present in this repository. Add a license if you plan to distribute or open-source the project.
