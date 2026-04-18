"""
hnsw_retriever.py
=================
Drop-in document retriever backed by your custom HNSW index.
Replaces a FAISS IndexFlatL2 / IndexHNSWFlat workflow.

Quick-start
-----------
    from hnsw_retriever import HNSWRetriever
    import numpy as np

    retriever = HNSWRetriever(dim=768)
    retriever.add_documents(["doc 1 text", "doc 2 text", ...], embeddings)
    results = retriever.query(query_embedding, k=5)
"""

import numpy as np
from typing import List, Tuple, Any

# The compiled C++ extension (hnsw_index.so built via setup.py)
import hnsw_index


# ---------------------------------------------------------------------------
# HNSWRetriever  — high-level Python wrapper
# ---------------------------------------------------------------------------
class HNSWRetriever:
    """
    Wraps the HNSW C++ extension with a document-store so you can retrieve
    text (or any metadata) by vector similarity.

    Parameters
    ----------
    dim : int
        Dimensionality of the embedding vectors.
    M : int
        HNSW M parameter (edges per node per layer). Default 16.
    ef_construction : int
        Candidate list size during index build. Default 200.
    ef_search : int
        Candidate list size during search. Higher → better recall. Default 50.
    seed : int
        RNG seed for reproducible level assignment. Default 42.
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        seed: int = 42,
    ):
        self.dim = dim
        self.ef_search = ef_search
        self._index = hnsw_index.HNSWIndex(M, ef_construction, seed)
        self._docs: List[Any] = []   # parallel list: doc at position i ↔ node id i

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: List[Any],
        embeddings: np.ndarray,
    ) -> None:
        """
        Index a batch of documents.

        Parameters
        ----------
        documents : list
            Any Python objects (strings, dicts, …) — stored as metadata.
        embeddings : np.ndarray, shape (N, dim), dtype float32
            One embedding per document, in the same order.
        """
        embeddings = self._coerce(embeddings)
        if len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"documents ({len(documents)}) and embeddings "
                f"({embeddings.shape[0]}) must have the same length"
            )
        self._docs.extend(documents)
        self._index.insert_batch(embeddings)

    def add_one(self, document: Any, embedding: np.ndarray) -> None:
        """Add a single document and its embedding."""
        embedding = self._coerce(embedding.reshape(1, -1))[0]
        self._docs.append(document)
        self._index.insert(embedding)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        ef: int = None,
    ) -> List[Tuple[Any, float]]:
        """
        Return the k most similar documents for a single query.

        Parameters
        ----------
        query_embedding : np.ndarray, shape (dim,) or (1, dim)
        k : int
        ef : int, optional
            Override ef_search for this call.

        Returns
        -------
        list of (document, squared_distance) tuples, sorted nearest-first.
        """
        q = self._coerce(query_embedding.reshape(1, -1))[0]
        ef_val = ef if ef is not None else max(self.ef_search, k)
        distances, indices = self._index.search(q, k, ef_val)
        return [(self._docs[i], float(d)) for d, i in zip(distances, indices)]

    def query_batch(
        self,
        query_embeddings: np.ndarray,
        k: int = 5,
        ef: int = None,
    ) -> List[List[Tuple[Any, float]]]:
        """
        Query multiple vectors at once.

        Parameters
        ----------
        query_embeddings : np.ndarray, shape (N, dim)

        Returns
        -------
        list of N result lists, each containing (document, distance) tuples.
        """
        qs = self._coerce(query_embeddings)
        ef_val = ef if ef is not None else max(self.ef_search, k)
        all_distances, all_indices = self._index.search_batch(qs, k, ef_val)
        results = []
        for row_d, row_i in zip(all_distances, all_indices):
            results.append([
                (self._docs[i], float(d))
                for d, i in zip(row_d, row_i)
                if i >= 0   # -1 means no result (index smaller than k)
            ])
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _coerce(self, arr: np.ndarray) -> np.ndarray:
        """Ensure C-contiguous float32 array."""
        arr = np.asarray(arr, dtype=np.float32)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return arr

    def __len__(self) -> int:
        return len(self._docs)

    def __repr__(self) -> str:
        return (
            f"HNSWRetriever(docs={len(self._docs)}, "
            f"dim={self.dim}, ef_search={self.ef_search})"
        )


# ---------------------------------------------------------------------------
# Example usage  (run:  python hnsw_retriever.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    # ---- Simulate a document corpus with embeddings ----------------------
    # In practice, replace this with your actual embedder, e.g.:
    #   from sentence_transformers import SentenceTransformer
    #   model = SentenceTransformer("all-MiniLM-L6-v2")
    #   embeddings = model.encode(documents, convert_to_numpy=True)

    N_DOCS  = 5_000
    DIM     = 384        # e.g. all-MiniLM-L6-v2 output dimension
    K       = 5

    rng = np.random.default_rng(0)
    documents  = [f"Document #{i}: lorem ipsum..." for i in range(N_DOCS)]
    embeddings = rng.standard_normal((N_DOCS, DIM)).astype(np.float32)

    # ---- Build index -----------------------------------------------------
    print(f"Building HNSW index over {N_DOCS} documents (dim={DIM})...")
    t0 = time.perf_counter()

    retriever = HNSWRetriever(dim=DIM, M=16, ef_construction=200, ef_search=50)
    retriever.add_documents(documents, embeddings)

    build_time = time.perf_counter() - t0
    print(f"  Done in {build_time:.2f}s  →  {retriever}")

    # ---- Single query ----------------------------------------------------
    query_vec = rng.standard_normal(DIM).astype(np.float32)

    t0 = time.perf_counter()
    hits = retriever.query(query_vec, k=K)
    query_time = (time.perf_counter() - t0) * 1000

    print(f"\nTop-{K} results (query time: {query_time:.2f} ms):")
    for rank, (doc, dist) in enumerate(hits, 1):
        print(f"  {rank}. dist²={dist:.4f}  |  {doc[:60]}")

    # ---- Batch query -----------------------------------------------------
    N_QUERIES = 100
    query_batch = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)

    t0 = time.perf_counter()
    batch_hits = retriever.query_batch(query_batch, k=K)
    batch_time = (time.perf_counter() - t0) * 1000

    print(f"\nBatch query: {N_QUERIES} queries in {batch_time:.2f} ms "
          f"({batch_time/N_QUERIES:.2f} ms/query)")
    print(f"First result of query[0]: {batch_hits[0][0]}")