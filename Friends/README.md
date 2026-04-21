# Semantic Search Engine with Dense Retrieval, Hybrid Ranking, and Analysis (MS MARCO)

This project builds a complete search system that compares:

- `BM25` (lexical search)
- `BERT dense retrieval` (semantic search)
- `Hybrid retrieval` (combining both)

It also includes:

- Performance evaluation (`Precision@k`, `Recall@k`)
- ANN vs Exact FAISS comparison
- Query-type analysis
- PCA embedding visualization
- Interactive Streamlit app

The core dataset is a subset of **MS MARCO v1.1**.

---

## 1) What Problem Are We Solving?

Traditional keyword search (like BM25) is great when query words exactly match document words.  
But many queries are semantic: users ask the same meaning using different words.

Example:

- Query: `how to stop a running nose`
- Relevant doc may contain: `ways to treat rhinorrhea`

Keyword overlap is weak, but semantic meaning is close.  
Dense retrieval helps by searching in embedding space (vector similarity).

---

## 2) Project Goal

Build an end-to-end system that:

1. Loads MS MARCO subset
2. Generates dense embeddings with `all-MiniLM-L6-v2`
3. Builds FAISS indexes (`ANN` + `Exact`)
4. Implements BM25, BERT, and Hybrid search
5. Evaluates with Precision/Recall on labeled relevance
6. Provides visual and interactive analysis via Streamlit
7. Avoids recomputation by caching all heavy artifacts locally

---

## 3) High-Level Pipeline

1. Load MS MARCO subset (`train[:N]`)
2. Extract passages as documents and deduplicate
3. Build query records with relevant doc IDs (`is_selected == 1`)
4. Compute document embeddings (GPU CUDA if available)
5. Save embeddings and metadata to local cache
6. Build FAISS indexes:
   - `IndexHNSWFlat` (ANN, fast)
   - `IndexFlatIP` (Exact, brute-force baseline)
7. Build BM25 index over tokenized docs
8. Serve 3 retrieval methods (BM25/BERT/Hybrid)
9. Run experiments and metrics
10. Visualize in Streamlit

---

## 4) Repository Files

- `main.py`  
  CLI runner for building cache, running evaluations, and saving report JSON.

- `retrieval_engine.py`  
  Main engine: data prep, caching, embeddings, FAISS, BM25, hybrid scoring, evaluation, PCA.

- `streamlit_app.py`  
  UI app for query search + side-by-side model comparison + dashboard experiments.

- `requirements.txt`  
  Python dependencies.

---

## 5) Core Concepts (Beginner-Friendly)

### A) BM25 (Lexical Search)

BM25 scores documents based on query term overlap and term frequency.  
Good for exact keyword matching, weak on synonyms/paraphrases.

### B) Dense Retrieval (Semantic Search)

We convert text into vectors (embeddings) using SentenceTransformer (`all-MiniLM-L6-v2`).  
Semantically similar texts have similar vectors.  
Search = nearest vectors in embedding space.

### C) ANN vs Exact

- `Exact` compares query with every document vector (slow, accurate).
- `ANN` uses approximate structures (HNSW graph) to search faster with tiny quality loss.

### D) Hybrid Retrieval

Combine semantic and lexical strengths:

`final_score = alpha * bert_score + (1 - alpha) * bm25_score`

Scores are normalized first, then fused.

### E) Metrics

- `Precision@k` = relevant results in top-k / k  
  High precision means top results are clean/relevant.

- `Recall@k` = relevant results in top-k / total relevant  
  High recall means system finds more of the relevant set.

Increasing `k` usually decreases precision and increases recall.

### F) PCA

Embeddings are high-dimensional (384D).  
PCA projects them to 2D for visualization to inspect rough clustering/semantic grouping.

---

## 6) Caching Strategy (Very Important)

First run is heavy. Later runs are fast because cached files are reused.

Cached artifacts in `cache/`:

- `documents.pkl`
- `query_records.pkl`
- `doc_embeddings.npy`
- `faiss_hnsw.index`
- `faiss_exact.index`
- `tokenized_docs.pkl`
- metadata JSON files for cache consistency checks

Current repo defaults are set for easy hosting/demo:

- subset size = `3,900` (small profile)
- embedding model = `all-MiniLM-L6-v2`
- cache directory = `cache/`

You can always run the full heavy version:

- set `FIXED_SUBSET_SIZE = 50_000` in `streamlit_app.py`
- set `FIXED_SUBSET_SIZE = 50_000` in `main.py`
- delete old `cache/` and rebuild

Only missing/inconsistent artifacts are rebuilt automatically.

---

## 7) GPU Usage

The engine auto-selects CUDA if available:

- `device = "cuda"` when `torch.cuda.is_available() == True`
- else CPU fallback

Embedding generation benefits the most from GPU.

Check quickly:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

During embedding run, verify with:

```bash
nvidia-smi
```

---

## 8) Setup Instructions

## Prerequisites

- Python 3.9+ recommended
- (Optional but recommended) NVIDIA GPU + CUDA-compatible PyTorch

Install dependencies:

```bash
pip install -r requirements.txt
```

If you want GPU PyTorch, install CUDA wheel (example):

```bash
pip uninstall -y torch torchvision torchaudio
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## 9) Run from CLI (`main.py`)

Build cache + run evaluations:

```bash
python main.py --eval-queries 200 --k-values 5,10,20 --top-k 10 --alpha 0.5
```

Useful options:

- `--skip-eval`  
  Only build/load pipeline quickly (no heavy evaluation loops).

- `--query "your query" --model hybrid --top-k 5`  
  Quick one-query CLI demo.

Output report:

- `outputs/evaluation_report.json`

---

## 10) Run Streamlit App

```bash
streamlit run streamlit_app.py
```

Note: this repo ships with a smaller demo profile for simpler hosting.  
For the full heavy run, switch `FIXED_SUBSET_SIZE` back to `50_000` in both `streamlit_app.py` and `main.py`, then rebuild cache.

### App Tabs

1. `Search`  
   Enter one query, choose one model, inspect top-k docs.

2. `Comparison View`  
   Same query across BERT vs BM25 vs Hybrid side by side.

3. `Analysis Dashboard`  
   Run experiments on `N` evaluation queries and view:
   - Precision vs k
   - Recall vs k
   - ANN vs Exact timing table
   - Query-type performance

4. `PCA Visualization`  
   2D scatter of sampled embeddings.

---

## 11) Understanding Typical Results

Expected behavior:

- Precision decreases as `k` increases.
- Recall increases as `k` increases.
- BERT often beats BM25 on semantic tasks.
- ANN is much faster than Exact with minimal quality drop.
- Hybrid may improve over BM25 and sometimes approach/exceed BERT after alpha tuning.

If Hybrid underperforms BERT, tune:

- `alpha` (try `0.2` to `0.8`)
- candidate pool size
- evaluation query count

---

## 12) Common Questions

### Q1: Do I need to run `main.py` every time?

No.  
After first successful cache build, you can usually run only Streamlit.  
Run `main.py` again when you want new experiments/reports.

### Q2: Why is it slow after embeddings finish?

Evaluation loops (BM25/BERT/Hybrid across many queries and multiple `k`) can still be heavy.

### Q3: Why do I see CPU usage even when device is CUDA?

Normal.  
Some pipeline stages are CPU (tokenization, BM25, FAISS CPU index, orchestration).  
Embedding inference uses GPU.

---

## 13) GitHub and Deployment Notes

For this repo's small demo profile, you can keep the generated `cache/` in GitHub if file sizes stay within GitHub limits.

Important:

- This is mainly for demo/easy hosting.
- You can always use the full heavy dataset locally by changing `FIXED_SUBSET_SIZE` back to `50_000` and rebuilding.
- For full heavy hosting, do **not** push cache artifacts to GitHub; use external storage or persistent disk instead.

---

## 14) One-Line Project Summary

We built and analyzed a semantic search system using dense retrieval, compared it with lexical BM25, and introduced a hybrid ranker with evaluation across retrieval depth, query types, and ANN-vs-Exact performance, delivered through an interactive Streamlit app.
URL For the hosted application : https://sementicsearchengine.streamlit.app/
