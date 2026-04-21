import os
import pickle
import time

import faiss
import numpy as np
import pandas as pd
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder


st.set_page_config(page_title="Semantic Search Engine", page_icon="🔎", layout="wide")

APP_BERT_EMB_PATH = "app_bert_embeddings.npy"
LEGACY_BERT_EMB_PATH = "bert_embeddings.npy"

EXAMPLE_QUERIES = [
    "Does aspirin reduce risk of cardiovascular events?",
    "Is vitamin D supplementation associated with bone health outcomes?",
    "Do statins lower LDL cholesterol in high-risk patients?",
    "What evidence links smoking to lung cancer risk?",
    "Are antibiotics effective for viral respiratory infections?",
    "Does exercise improve insulin sensitivity in adults?",
    "Can probiotics reduce symptoms of irritable bowel syndrome?",
    "Is high blood pressure associated with stroke risk?",
    "Do omega-3 supplements improve heart health outcomes?",
    "Is obesity linked to increased risk of type 2 diabetes?",
]

LOW_SIMILARITY_WARNING_THRESHOLD = 0.35


@st.cache_data(show_spinner=False)
def load_corpus_data():
    if not os.path.exists("corpus_data.pkl"):
        raise FileNotFoundError("Missing corpus_data.pkl. Run train.ipynb to generate it.")
    with open("corpus_data.pkl", "rb") as f:
        data = pickle.load(f)
    return data["ids"], data["text"]


@st.cache_data(show_spinner=False)
def load_eval_results():
    if not os.path.exists("final_eval_results.csv"):
        return None
    return pd.read_csv("final_eval_results.csv")


@st.cache_resource(show_spinner=False)
def get_bi_encoder(device):
    return SentenceTransformer("all-MiniLM-L6-v2", device=device)


@st.cache_resource(show_spinner=False)
def get_cross_encoder(device):
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)


@st.cache_resource(show_spinner=False)
def get_faiss_index(doc_count, device):
    _, docs = load_corpus_data()
    if len(docs) != doc_count:
        raise ValueError("Corpus size changed during runtime. Please refresh the app.")

    embeddings = None
    embedding_source = None

    if os.path.exists(APP_BERT_EMB_PATH):
        cached = np.load(APP_BERT_EMB_PATH).astype("float32")
        if cached.ndim == 2 and cached.shape[0] == doc_count:
            embeddings = cached
            embedding_source = APP_BERT_EMB_PATH

    if embeddings is None and os.path.exists(LEGACY_BERT_EMB_PATH):
        cached = np.load(LEGACY_BERT_EMB_PATH).astype("float32")
        if cached.ndim == 2 and cached.shape[0] == doc_count:
            embeddings = cached
            embedding_source = LEGACY_BERT_EMB_PATH

    if embeddings is None:
        # Self-heal when final_eval notebook overwrote bert_embeddings.npy for a different dataset.
        bi_encoder = get_bi_encoder(device)
        embeddings = bi_encoder.encode(
            docs,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        np.save(APP_BERT_EMB_PATH, embeddings)
        embedding_source = APP_BERT_EMB_PATH

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embedding_source


def search_bi_encoder(query, k, bi_encoder, index, document_ids, documents_text):
    query_vector = bi_encoder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vector)
    scores, indices = index.search(query_vector, k)

    results = []
    for rank, idx in enumerate(indices[0], start=1):
        results.append(
            {
                "rank": rank,
                "doc_id": document_ids[idx],
                "score": float(scores[0][rank - 1]),
                "text": documents_text[idx],
            }
        )
    return results


def search_cross_encoder(query, k, initial_k, bi_encoder, cross_encoder, index, document_ids, documents_text):
    candidates = search_bi_encoder(query, initial_k, bi_encoder, index, document_ids, documents_text)
    if not candidates:
        return []

    pairs = [[query, c["text"]] for c in candidates]
    scores = cross_encoder.predict(pairs)
    for i, score in enumerate(scores):
        candidates[i]["score"] = float(score)

    reranked = sorted(candidates, key=lambda x: x["score"], reverse=True)[:k]
    for rank, hit in enumerate(reranked, start=1):
        hit["rank"] = rank
    return reranked


def render_hits(title, hits, score_label):
    st.subheader(title)
    if not hits:
        st.info("No results returned.")
        return

    for hit in hits:
        with st.expander(
            f"#{hit['rank']} | {score_label}: {hit['score']:.4f} | Doc ID: {hit['doc_id']}",
            expanded=(hit["rank"] == 1),
        ):
            st.write(hit["text"])


device = "cuda" if torch.cuda.is_available() else "cpu"
document_ids, documents_text = load_corpus_data()


def ensure_index_ready():
    with st.spinner("Preparing retrieval index (first run may take a bit)..."):
        return get_faiss_index(len(documents_text), device)

st.title("Semantic Search Engine")
st.caption("Fast startup, rich UX: cached corpus/index + lazy model loading")

c1, c2, c3 = st.columns(3)
c1.metric("Documents", f"{len(documents_text):,}")
c2.metric("Device", device.upper())
c3.metric("Index", "FAISS IndexFlatIP")

tabs = st.tabs(["Search", "Comparison View", "Evaluation Snapshot", "System Status"])

with tabs[0]:
    st.subheader("Single Model Search")
    input_mode = st.radio(
        "Query source",
        ["Prewritten", "Custom"],
        horizontal=True,
        key="single_input_mode",
    )
    if input_mode == "Prewritten":
        query = st.selectbox("Choose a prewritten query", EXAMPLE_QUERIES, key="single_query_dropdown")
    else:
        query = st.text_input(
            "Enter your query",
            placeholder="e.g., How do complex biomaterials show inductive properties?",
            key="single_query_custom",
        )

    mode = st.selectbox(
        "Pipeline",
        ["Bi-Encoder (Fast)", "Cross-Encoder (High Precision)"],
        key="single_mode",
    )
    top_k = int(st.slider("Top-k", min_value=1, max_value=20, value=5, key="single_k"))
    initial_k = int(st.slider("Reranker candidate pool", min_value=20, max_value=200, value=80, step=10, key="single_initial"))

    if st.button("Run Search", use_container_width=True):
        if not query.strip():
            st.warning("Please type a query.")
        else:
            t0 = time.perf_counter()
            bi_encoder = get_bi_encoder(device)
            index, _ = ensure_index_ready()

            if mode == "Bi-Encoder (Fast)":
                hits = search_bi_encoder(query, top_k, bi_encoder, index, document_ids, documents_text)
                score_label = "Cosine Similarity"
            else:
                cross_encoder = get_cross_encoder(device)
                hits = search_cross_encoder(
                    query,
                    top_k,
                    max(initial_k, top_k),
                    bi_encoder,
                    cross_encoder,
                    index,
                    document_ids,
                    documents_text,
                )
                score_label = "Cross-Encoder Score"

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            st.success(f"Retrieved {len(hits)} results in {elapsed_ms:.1f} ms")
            if mode == "Bi-Encoder (Fast)" and hits and hits[0]["score"] < LOW_SIMILARITY_WARNING_THRESHOLD:
                st.warning(
                    "Top similarity is low for this corpus. The query may be out-of-domain for the indexed dataset. "
                    "Try a biomedical/scientific claim-style query from the dropdown."
                )
            render_hits(mode, hits, score_label)

with tabs[1]:
    st.subheader("Same Query: Fast vs High-Precision")
    cmp_input_mode = st.radio(
        "Comparison query source",
        ["Prewritten", "Custom"],
        horizontal=True,
        key="cmp_input_mode",
    )
    if cmp_input_mode == "Prewritten":
        cmp_query = st.selectbox("Choose a comparison query", EXAMPLE_QUERIES, key="cmp_query_dropdown")
    else:
        cmp_query = st.text_input("Comparison query", value="", key="cmp_query_custom")
    cmp_k = int(st.slider("Comparison Top-k", min_value=1, max_value=20, value=5, key="cmp_k"))
    cmp_initial = int(st.slider("Comparison candidate pool", min_value=20, max_value=200, value=100, step=10, key="cmp_initial"))

    if st.button("Run Side-by-Side", use_container_width=True):
        if not cmp_query.strip():
            st.warning("Please type a query.")
        else:
            bi_encoder = get_bi_encoder(device)
            index, _ = ensure_index_ready()
            t1 = time.perf_counter()
            fast_hits = search_bi_encoder(cmp_query, cmp_k, bi_encoder, index, document_ids, documents_text)
            fast_ms = (time.perf_counter() - t1) * 1000.0

            t2 = time.perf_counter()
            cross_encoder = get_cross_encoder(device)
            rerank_hits = search_cross_encoder(
                cmp_query,
                cmp_k,
                max(cmp_initial, cmp_k),
                bi_encoder,
                cross_encoder,
                index,
                document_ids,
                documents_text,
            )
            rerank_ms = (time.perf_counter() - t2) * 1000.0

            left, right = st.columns(2)
            with left:
                st.caption(f"Bi-Encoder latency: {fast_ms:.1f} ms")
                render_hits("Bi-Encoder (Fast)", fast_hits, "Cosine Similarity")
            with right:
                st.caption(f"Cross-Encoder latency: {rerank_ms:.1f} ms")
                render_hits("Cross-Encoder (Reranked)", rerank_hits, "Cross-Encoder Score")

with tabs[2]:
    st.subheader("Evaluation Snapshot")
    eval_df = load_eval_results()
    if eval_df is None:
        st.info("No final_eval_results.csv found yet. Run final_eval.ipynb to generate evaluation output.")
    else:
        st.dataframe(eval_df, use_container_width=True)
        st.bar_chart(eval_df.set_index("Method")[["Precision@10", "Recall@10"]])

with tabs[3]:
    st.subheader("System Status")
    st.write("Artifacts detected:")
    ready_source = "Not loaded yet"
    if st.button("Check active index source", key="check_index_source"):
        _, src = ensure_index_ready()
        ready_source = src
        st.session_state["active_index_source"] = src
    elif "active_index_source" in st.session_state:
        ready_source = st.session_state["active_index_source"]

    status_rows = [
        {"Artifact": "corpus_data.pkl", "Exists": os.path.exists("corpus_data.pkl")},
        {"Artifact": "bert_embeddings.npy", "Exists": os.path.exists(LEGACY_BERT_EMB_PATH)},
        {"Artifact": "app_bert_embeddings.npy", "Exists": os.path.exists(APP_BERT_EMB_PATH)},
        {"Artifact": "final_eval_results.csv", "Exists": os.path.exists("final_eval_results.csv")},
        {"Artifact": "w2v_embeddings.npy", "Exists": os.path.exists("w2v_embeddings.npy")},
        {"Artifact": "w2v_model.bin", "Exists": os.path.exists("w2v_model.bin")},
    ]
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True)
    st.caption(f"Active index embeddings source: {ready_source}")

    st.info(
        "Performance note: Bi-Encoder loads lazily on first search. Cross-Encoder loads only if reranking is used, "
        "which keeps initial app startup fast."
    )