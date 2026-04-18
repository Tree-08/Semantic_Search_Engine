import streamlit as st
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from beir.datasets.data_loader import GenericDataLoader
import os
import pickle

# --- Page Configuration ---
st.set_page_config(page_title="ANN Search Engine", page_icon="🔍", layout="wide")
st.title("🔍 Semantic Search Engine")
st.markdown("A vector-based search engine utilizing Approximate Nearest Neighbor (ANN) search, Bi-Encoders, and Cross-Encoders.")

# --- Load Data and Models (Cached) ---
@st.cache_resource(show_spinner="Loading models and data... This will take a moment.")
@st.cache_resource(show_spinner="Loading models and data... This will take a moment.")
def load_backend():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    bi_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
    
    # LOAD THE FAST PICKLE FILE INSTEAD OF PARSING BEIR
    with open("corpus_data.pkl", "rb") as f:
        data = pickle.load(f)
        document_ids = data["ids"]
        documents_text = data["text"]
    
    embeddings = np.load("bert_embeddings.npy").astype('float32')
        
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    return bi_encoder, cross_encoder, document_ids, documents_text, index

# Initialize backend
bi_encoder, cross_encoder, document_ids, documents_text, index = load_backend()

# --- Search Functions ---
def search_bi_encoder(query, k):
    query_vector = bi_encoder.encode([query]).astype('float32')
    faiss.normalize_L2(query_vector)
    distances, indices = index.search(query_vector, k)
    
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        results.append({
            "doc_id": document_ids[idx],
            "score": float(distances[0][i]),
            "text": documents_text[idx]
        })
    return results

def search_cross_encoder(query, k, initial_k=100):
    initial_hits = search_bi_encoder(query, k=initial_k)
    if not initial_hits:
        return []
        
    cross_inp = [[query, hit["text"]] for hit in initial_hits]
    cross_scores = cross_encoder.predict(cross_inp)
    
    for idx in range(len(cross_scores)):
        initial_hits[idx]["score"] = float(cross_scores[idx]) # Replace score with Cross-Encoder logit
        
    reranked_hits = sorted(initial_hits, key=lambda x: x["score"], reverse=True)
    return reranked_hits[:k]

# --- UI Layout ---
st.sidebar.header("Search Parameters")
search_mode = st.sidebar.radio("Pipeline Architecture", ["Bi-Encoder (Fast)", "Cross-Encoder (High Precision)"])
k_results = st.sidebar.slider("Number of Results (K)", min_value=1, max_value=20, value=5)

st.markdown("---")
query = st.text_input("Enter your search query:", placeholder="e.g., How do complex biomaterials show inductive properties?")

if st.button("Search") and query:
    with st.spinner("Searching..."):
        if search_mode == "Bi-Encoder (Fast)":
            results = search_bi_encoder(query, k=k_results)
            score_label = "Cosine Similarity"
        else:
            results = search_cross_encoder(query, k=k_results)
            score_label = "Cross-Encoder Logit"
            
    st.success(f"Retrieved Top {k_results} Documents")
    
    # Display Results
    for rank, hit in enumerate(results, 1):
        with st.expander(f"Rank {rank} | Score: {hit['score']:.4f} | Doc ID: {hit['doc_id']}", expanded=(rank==1)):
            st.write(hit['text'])