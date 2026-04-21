import json
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
import torch
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


@dataclass
class SearchConfig:
    subset_size: int = 50_000
    eval_queries: int = 200
    model_name: str = "all-MiniLM-L6-v2"
    cache_dir: str = "cache"
    hnsw_m: int = 32
    hnsw_ef_search: int = 64
    hnsw_ef_construction: int = 200
    embedding_batch_size: int = 256
    hybrid_alpha: float = 0.5
    hybrid_pool_factor: int = 5
    pca_sample_size: int = 1000
    random_seed: int = 42
    force_rebuild: bool = False
    device_preference: str = "cuda"

    def to_dataset_metadata(self) -> Dict[str, object]:
        return {"subset_size": self.subset_size}


class SemanticSearchEngine:
    def __init__(self, config: Optional[SearchConfig] = None) -> None:
        self.config = config or SearchConfig()
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.device = self._resolve_device()
        self.model: Optional[SentenceTransformer] = None

        self.documents: List[str] = []
        self.query_records: List[Dict[str, object]] = []
        self.doc_embeddings: Optional[np.ndarray] = None
        self.ann_index: Optional[faiss.Index] = None
        self.exact_index: Optional[faiss.Index] = None
        self.tokenized_docs: Optional[List[List[str]]] = None
        self.bm25: Optional[BM25Okapi] = None

    def _resolve_device(self) -> str:
        if self.config.device_preference == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _cache_path(self, filename: str) -> Path:
        return self.cache_dir / filename

    def load_or_build(self) -> "SemanticSearchEngine":
        self._load_or_prepare_dataset()
        self._load_model()
        self._load_or_create_embeddings()
        self._load_or_create_indices()
        self._load_or_create_bm25()
        return self

    def _load_or_prepare_dataset(self) -> None:
        docs_path = self._cache_path("documents.pkl")
        queries_path = self._cache_path("query_records.pkl")
        metadata_path = self._cache_path("metadata.json")
        expected = self.config.to_dataset_metadata()

        if (
            not self.config.force_rebuild
            and docs_path.exists()
            and queries_path.exists()
            and metadata_path.exists()
        ):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("config") == expected:
                with docs_path.open("rb") as f:
                    self.documents = pickle.load(f)
                with queries_path.open("rb") as f:
                    self.query_records = pickle.load(f)
                return

        dataset = load_dataset("ms_marco", "v1.1", split=f"train[:{self.config.subset_size}]")
        doc_to_id: Dict[str, int] = {}
        documents: List[str] = []
        query_records: List[Dict[str, object]] = []

        for item in dataset:
            query_text = (item.get("query") or "").strip()
            passages = item["passages"]["passage_text"]
            labels = item["passages"]["is_selected"]

            relevant_ids = set()
            for passage, label in zip(passages, labels):
                text = (passage or "").strip()
                if not text:
                    continue
                doc_id = doc_to_id.get(text)
                if doc_id is None:
                    doc_id = len(documents)
                    doc_to_id[text] = doc_id
                    documents.append(text)
                if int(label) == 1:
                    relevant_ids.add(doc_id)

            query_records.append(
                {
                    "query": query_text,
                    "relevant_ids": sorted(relevant_ids),
                }
            )

        self.documents = documents
        self.query_records = query_records

        with docs_path.open("wb") as f:
            pickle.dump(self.documents, f)
        with queries_path.open("wb") as f:
            pickle.dump(self.query_records, f)
        metadata_path.write_text(
            json.dumps(
                {
                    "config": expected,
                    "n_documents": len(self.documents),
                    "n_queries": len(self.query_records),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load_model(self) -> None:
        if self.model is None:
            self.model = SentenceTransformer(self.config.model_name, device=self.device)

    def _load_or_create_embeddings(self) -> None:
        embeddings_path = self._cache_path("doc_embeddings.npy")
        embeddings_meta_path = self._cache_path("embeddings_meta.json")
        expected_meta = {
            "model_name": self.config.model_name,
            "num_documents": len(self.documents),
        }

        if embeddings_path.exists() and embeddings_meta_path.exists() and not self.config.force_rebuild:
            loaded_meta = json.loads(embeddings_meta_path.read_text(encoding="utf-8"))
            loaded = np.load(embeddings_path)
            if (
                loaded_meta == expected_meta
                and loaded.ndim == 2
                and loaded.shape[0] == len(self.documents)
            ):
                self.doc_embeddings = loaded.astype(np.float32, copy=False)
                return

        if not self.documents:
            raise RuntimeError("Documents are empty. Cannot create embeddings.")

        self.doc_embeddings = self.model.encode(
            self.documents,
            batch_size=self.config.embedding_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device,
        ).astype(np.float32)
        np.save(embeddings_path, self.doc_embeddings)
        embeddings_meta_path.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")

    def _build_ann_index(self) -> faiss.Index:
        dim = self.doc_embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dim, self.config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = self.config.hnsw_ef_search
        index.hnsw.efConstruction = self.config.hnsw_ef_construction
        index.add(self.doc_embeddings)
        return index

    def _build_exact_index(self) -> faiss.Index:
        dim = self.doc_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(self.doc_embeddings)
        return index

    def _index_matches(self, index: faiss.Index) -> bool:
        return index.ntotal == len(self.documents) and index.d == self.doc_embeddings.shape[1]

    def _load_or_create_indices(self) -> None:
        ann_path = self._cache_path("faiss_hnsw.index")
        exact_path = self._cache_path("faiss_exact.index")
        index_meta_path = self._cache_path("index_meta.json")
        expected_meta = {
            "num_documents": len(self.documents),
            "embedding_dim": int(self.doc_embeddings.shape[1]),
            "hnsw_m": self.config.hnsw_m,
            "metric": "inner_product",
        }
        loaded_meta = {}
        if index_meta_path.exists():
            loaded_meta = json.loads(index_meta_path.read_text(encoding="utf-8"))

        if ann_path.exists() and not self.config.force_rebuild and loaded_meta == expected_meta:
            loaded_ann = faiss.read_index(str(ann_path))
            if self._index_matches(loaded_ann):
                loaded_ann.hnsw.efSearch = self.config.hnsw_ef_search
                self.ann_index = loaded_ann
        if self.ann_index is None:
            self.ann_index = self._build_ann_index()
            faiss.write_index(self.ann_index, str(ann_path))

        if exact_path.exists() and not self.config.force_rebuild and loaded_meta == expected_meta:
            loaded_exact = faiss.read_index(str(exact_path))
            if self._index_matches(loaded_exact):
                self.exact_index = loaded_exact
        if self.exact_index is None:
            self.exact_index = self._build_exact_index()
            faiss.write_index(self.exact_index, str(exact_path))

        index_meta_path.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _load_or_create_bm25(self) -> None:
        tokenized_path = self._cache_path("tokenized_docs.pkl")
        should_rebuild = True
        if tokenized_path.exists() and not self.config.force_rebuild:
            try:
                with tokenized_path.open("rb") as f:
                    loaded_tokens = pickle.load(f)
                if isinstance(loaded_tokens, list) and len(loaded_tokens) == len(self.documents):
                    self.tokenized_docs = loaded_tokens
                    should_rebuild = False
            except Exception:
                should_rebuild = True

        if should_rebuild:
            self.tokenized_docs = [self._tokenize(doc) for doc in self.documents]
            with tokenized_path.open("wb") as f:
                pickle.dump(self.tokenized_docs, f)

        self.bm25 = BM25Okapi(self.tokenized_docs)

    def _encode_query(self, query: str) -> np.ndarray:
        vec = self.model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device,
        ).astype(np.float32)
        if vec.ndim == 1:
            vec = np.expand_dims(vec, axis=0)
        return vec

    def _format_hits(self, doc_ids: Sequence[int], scores: Sequence[float]) -> List[Dict[str, object]]:
        results = []
        for rank, (doc_id, score) in enumerate(zip(doc_ids, scores), start=1):
            if doc_id < 0:
                continue
            results.append(
                {
                    "rank": rank,
                    "doc_id": int(doc_id),
                    "score": float(score),
                    "text": self.documents[int(doc_id)],
                }
            )
        return results

    def semantic_search(self, query: str, k: int = 5, use_exact: bool = False) -> List[Dict[str, object]]:
        if k <= 0:
            return []
        index = self.exact_index if use_exact else self.ann_index
        query_vec = self._encode_query(query)
        scores, indices = index.search(query_vec, min(k, len(self.documents)))
        return self._format_hits(indices[0].tolist(), scores[0].tolist())

    def bm25_search(self, query: str, k: int = 5) -> List[Dict[str, object]]:
        if k <= 0:
            return []
        tokenized_query = self._tokenize(query)
        scores = np.asarray(self.bm25.get_scores(tokenized_query), dtype=np.float32)
        k = min(k, len(scores))
        if k == len(scores):
            top_indices = np.arange(len(scores))
        else:
            top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        top_scores = scores[top_indices]
        return self._format_hits(top_indices.tolist(), top_scores.tolist())

    @staticmethod
    def _minmax(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        low = float(values.min())
        high = float(values.max())
        if high - low < 1e-8:
            return np.zeros_like(values, dtype=np.float32)
        return (values - low) / (high - low)

    def hybrid_search(self, query: str, k: int = 5, alpha: Optional[float] = None) -> List[Dict[str, object]]:
        if k <= 0:
            return []
        alpha = self.config.hybrid_alpha if alpha is None else float(alpha)
        pool_k = min(len(self.documents), max(k, k * self.config.hybrid_pool_factor))

        dense_hits = self.semantic_search(query, k=pool_k, use_exact=False)
        dense_score_map = {hit["doc_id"]: hit["score"] for hit in dense_hits}

        tokenized_query = self._tokenize(query)
        bm25_scores = np.asarray(self.bm25.get_scores(tokenized_query), dtype=np.float32)
        if pool_k == len(bm25_scores):
            bm25_top_indices = np.arange(len(bm25_scores))
        else:
            bm25_top_indices = np.argpartition(bm25_scores, -pool_k)[-pool_k:]

        candidate_ids = list(set(dense_score_map.keys()).union(set(bm25_top_indices.tolist())))
        dense_values = np.array([dense_score_map.get(doc_id, 0.0) for doc_id in candidate_ids], dtype=np.float32)
        bm25_values = np.array([bm25_scores[doc_id] for doc_id in candidate_ids], dtype=np.float32)

        dense_norm = self._minmax(dense_values)
        bm25_norm = self._minmax(bm25_values)
        fused_scores = alpha * dense_norm + (1.0 - alpha) * bm25_norm

        order = np.argsort(fused_scores)[::-1][:k]
        top_doc_ids = [candidate_ids[i] for i in order]
        top_scores = [float(fused_scores[i]) for i in order]
        return self._format_hits(top_doc_ids, top_scores)

    def retrieve_doc_ids(
        self,
        query: str,
        method: str,
        k: int,
        index_type: str = "ann",
        alpha: Optional[float] = None,
    ) -> List[int]:
        method = method.lower()
        if method == "bm25":
            hits = self.bm25_search(query, k=k)
        elif method == "bert":
            hits = self.semantic_search(query, k=k, use_exact=(index_type.lower() == "exact"))
        elif method == "hybrid":
            hits = self.hybrid_search(query, k=k, alpha=alpha)
        else:
            raise ValueError(f"Unsupported method: {method}")
        return [hit["doc_id"] for hit in hits]

    def evaluate_single(
        self,
        query: str,
        relevant_ids: Sequence[int],
        method: str,
        k: int,
        index_type: str = "ann",
        alpha: Optional[float] = None,
    ) -> Tuple[float, float]:
        relevant_set = set(relevant_ids)
        if not relevant_set:
            return 0.0, 0.0
        retrieved = self.retrieve_doc_ids(query, method=method, k=k, index_type=index_type, alpha=alpha)
        hit_count = len(relevant_set.intersection(retrieved))
        precision = hit_count / float(k) if k > 0 else 0.0
        recall = hit_count / float(len(relevant_set))
        return precision, recall

    def evaluate_k(
        self,
        k_values: Sequence[int] = (5, 10, 20),
        methods: Sequence[str] = ("bm25", "bert", "hybrid"),
        n_queries: Optional[int] = None,
        index_type: str = "ann",
        alpha: Optional[float] = None,
    ) -> Dict[str, Dict[int, Dict[str, float]]]:
        n = min(n_queries or self.config.eval_queries, len(self.query_records))
        eval_set = [rec for rec in self.query_records[:n] if rec["relevant_ids"]]
        results: Dict[str, Dict[int, Dict[str, float]]] = {}

        for method in methods:
            method_results: Dict[int, Dict[str, float]] = {}
            for k in k_values:
                precisions: List[float] = []
                recalls: List[float] = []
                for rec in eval_set:
                    p, r = self.evaluate_single(
                        rec["query"],
                        rec["relevant_ids"],
                        method=method,
                        k=int(k),
                        index_type=index_type,
                        alpha=alpha,
                    )
                    precisions.append(p)
                    recalls.append(r)
                method_results[int(k)] = {
                    "precision": float(np.mean(precisions)) if precisions else 0.0,
                    "recall": float(np.mean(recalls)) if recalls else 0.0,
                    "num_queries": float(len(precisions)),
                }
            results[method] = method_results
        return results

    @staticmethod
    def classify_query(query: str) -> str:
        stripped = query.strip().lower()
        if any(stripped.startswith(prefix) for prefix in ("who", "what", "when", "where", "which", "whom")):
            return "factual"
        if len(stripped.split()) <= 2:
            return "short"
        return "complex"

    def query_type_performance(
        self,
        methods: Sequence[str] = ("bm25", "bert", "hybrid"),
        k: int = 10,
        n_queries: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        n = min(n_queries or self.config.eval_queries, len(self.query_records))
        eval_set = [rec for rec in self.query_records[:n] if rec["relevant_ids"]]

        raw: Dict[str, Dict[str, List[float]]] = {
            method: {"factual": [], "short": [], "complex": []} for method in methods
        }

        for rec in eval_set:
            q_type = self.classify_query(rec["query"])
            for method in methods:
                p, _ = self.evaluate_single(
                    rec["query"],
                    rec["relevant_ids"],
                    method=method,
                    k=k,
                    index_type="ann",
                    alpha=alpha,
                )
                raw[method][q_type].append(p)

        out: Dict[str, Dict[str, Dict[str, float]]] = {}
        for method, buckets in raw.items():
            out[method] = {}
            for q_type, values in buckets.items():
                out[method][q_type] = {
                    "precision": float(np.mean(values)) if values else 0.0,
                    "num_queries": float(len(values)),
                }
        return out

    def compare_ann_exact(self, k: int = 10, n_queries: Optional[int] = None) -> Dict[str, float]:
        n = min(n_queries or self.config.eval_queries, len(self.query_records))
        eval_set = [rec for rec in self.query_records[:n] if rec["relevant_ids"]]
        k = min(k, len(self.documents))

        ann_times: List[float] = []
        exact_times: List[float] = []
        ann_recalls: List[float] = []
        exact_recalls: List[float] = []
        overlaps: List[float] = []

        for rec in eval_set:
            query_vec = self._encode_query(rec["query"])
            relevant = set(rec["relevant_ids"])

            start = time.perf_counter()
            ann_scores, ann_indices = self.ann_index.search(query_vec, k)
            ann_times.append((time.perf_counter() - start) * 1000.0)

            start = time.perf_counter()
            exact_scores, exact_indices = self.exact_index.search(query_vec, k)
            exact_times.append((time.perf_counter() - start) * 1000.0)

            ann_ids = [int(i) for i in ann_indices[0] if i >= 0]
            exact_ids = [int(i) for i in exact_indices[0] if i >= 0]

            ann_hit = len(relevant.intersection(ann_ids))
            exact_hit = len(relevant.intersection(exact_ids))
            ann_recalls.append(ann_hit / float(len(relevant)))
            exact_recalls.append(exact_hit / float(len(relevant)))

            overlap = len(set(ann_ids).intersection(exact_ids)) / float(k)
            overlaps.append(overlap)

        return {
            "num_queries": float(len(eval_set)),
            "k": float(k),
            "ann_avg_time_ms": float(np.mean(ann_times)) if ann_times else 0.0,
            "exact_avg_time_ms": float(np.mean(exact_times)) if exact_times else 0.0,
            "ann_avg_recall": float(np.mean(ann_recalls)) if ann_recalls else 0.0,
            "exact_avg_recall": float(np.mean(exact_recalls)) if exact_recalls else 0.0,
            "result_overlap_at_k": float(np.mean(overlaps)) if overlaps else 0.0,
        }

    def comparison_for_query(self, query: str, k: int = 5, alpha: Optional[float] = None) -> Dict[str, List[Dict[str, object]]]:
        return {
            "bert": self.semantic_search(query, k=k, use_exact=False),
            "bm25": self.bm25_search(query, k=k),
            "hybrid": self.hybrid_search(query, k=k, alpha=alpha),
        }

    def pca_projection(self, sample_size: Optional[int] = None) -> List[Dict[str, object]]:
        if self.doc_embeddings is None or len(self.doc_embeddings) < 2:
            return []
        size = min(sample_size or self.config.pca_sample_size, len(self.doc_embeddings))
        if size < 2:
            return []

        rng = np.random.default_rng(self.config.random_seed)
        indices = rng.choice(len(self.doc_embeddings), size=size, replace=False)
        sample_vectors = self.doc_embeddings[indices]

        pca = PCA(n_components=2, random_state=self.config.random_seed)
        coords = pca.fit_transform(sample_vectors)

        points: List[Dict[str, object]] = []
        for pos, doc_id in enumerate(indices):
            points.append(
                {
                    "doc_id": int(doc_id),
                    "x": float(coords[pos, 0]),
                    "y": float(coords[pos, 1]),
                    "snippet": self.documents[int(doc_id)][:200],
                }
            )
        return points

    def system_summary(self) -> Dict[str, object]:
        return {
            "device": self.device,
            "subset_size": self.config.subset_size,
            "model_name": self.config.model_name,
            "documents": len(self.documents),
            "queries": len(self.query_records),
            "embedding_dim": int(self.doc_embeddings.shape[1]) if self.doc_embeddings is not None else 0,
            "cache_dir": str(self.cache_dir.resolve()),
        }
