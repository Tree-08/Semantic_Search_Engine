import pandas as pd
import streamlit as st

from retrieval_engine import SearchConfig, SemanticSearchEngine


st.set_page_config(page_title="Semantic Search Engine", layout="wide")

FIXED_SUBSET_SIZE = 3_900
FIXED_CACHE_DIR = "cache"
FIXED_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

EXAMPLE_QUERIES = [
    "what is semantic search",
    "who invented the telephone",
    "when was the internet invented",
    "how does photosynthesis work",
    "best way to treat a fever",
    "difference between cpu and gpu",
    "what causes climate change",
    "how to improve memory retention",
    "symptoms of vitamin d deficiency",
    "why is the sky blue",
]


@st.cache_resource(show_spinner=False)
def get_engine() -> SemanticSearchEngine:
    config = SearchConfig(
        subset_size=FIXED_SUBSET_SIZE,
        cache_dir=FIXED_CACHE_DIR,
        model_name=FIXED_EMBEDDING_MODEL,
        force_rebuild=False,
    )
    return SemanticSearchEngine(config).load_or_build()


def run_selected_search(
    engine: SemanticSearchEngine,
    model_name: str,
    query: str,
    top_k: int,
    alpha: float,
):
    if model_name == "BERT":
        return engine.semantic_search(query, k=top_k, use_exact=False)
    if model_name == "BM25":
        return engine.bm25_search(query, k=top_k)
    return engine.hybrid_search(query, k=top_k, alpha=alpha)


def show_hits(title: str, hits):
    st.subheader(title)
    if not hits:
        st.info("No results.")
        return
    for hit in hits:
        st.markdown(
            f"**#{hit['rank']} | score={hit['score']:.4f} | doc_id={hit['doc_id']}**\n\n{hit['text']}"
        )


st.title("Semantic Search with Dense, Lexical, and Hybrid Retrieval")
st.caption("MS MARCO subset + cached local embeddings/documents + ANN/Exact analysis")
st.caption("Demo config for easy hosting: subset=3,900, embedding model=all-MiniLM-L6-v2, cache_dir=cache")
st.caption("For full heavy run, change FIXED_SUBSET_SIZE to 50,000 in this file and in main.py, then rebuild cache.")

with st.spinner("Loading cached artifacts (or building once if missing)..."):
    engine = get_engine()

summary = engine.system_summary()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Documents", f"{summary['documents']:,}")
c2.metric("Queries", f"{summary['queries']:,}")
c3.metric("Embedding Dim", summary["embedding_dim"])
c4.metric("Device", summary["device"])
st.caption(f"Cache path: `{summary['cache_dir']}`")

tabs = st.tabs(["Search", "Comparison View", "Analysis Dashboard", "PCA Visualization", "How To Read Results"])

with tabs[0]:
    st.subheader("Single Model Search")
    search_example = st.selectbox("Example queries", options=EXAMPLE_QUERIES, index=0, key="search_example")
    search_custom = st.text_input("Or type your own query (optional)", value="", key="search_custom")
    query = search_custom.strip() if search_custom.strip() else search_example
    model_name = st.selectbox("Select retrieval model", options=["BERT", "BM25", "Hybrid"], key="search_model")
    search_top_k = int(st.slider("Top-k", min_value=1, max_value=50, value=10, key="search_top_k"))
    if model_name == "Hybrid":
        search_alpha = float(
            st.slider("Hybrid alpha", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="search_alpha")
        )
    else:
        search_alpha = 0.5

    if st.button("Run Search", use_container_width=True):
        hits = run_selected_search(engine, model_name, query, search_top_k, search_alpha)
        show_hits(f"{model_name} Results", hits)

with tabs[1]:
    st.subheader("Same Query: BERT vs BM25 vs Hybrid")
    compare_example = st.selectbox("Example comparison queries", options=EXAMPLE_QUERIES, index=1, key="compare_example")
    compare_custom = st.text_input("Or type your own comparison query (optional)", value="", key="compare_custom")
    cmp_query = compare_custom.strip() if compare_custom.strip() else compare_example
    cmp_top_k = int(st.slider("Top-k", min_value=1, max_value=50, value=10, key="compare_top_k"))
    cmp_alpha = float(st.slider("Hybrid alpha", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="compare_alpha"))

    if st.button("Run Comparison", use_container_width=True):
        comparison = engine.comparison_for_query(cmp_query, k=cmp_top_k, alpha=cmp_alpha)
        col1, col2, col3 = st.columns(3)
        with col1:
            show_hits("BERT", comparison["bert"])
        with col2:
            show_hits("BM25", comparison["bm25"])
        with col3:
            show_hits("Hybrid", comparison["hybrid"])

with tabs[2]:
    st.subheader("Evaluation and Experiments")
    analysis_eval_queries = int(
        st.number_input("Evaluation queries", min_value=20, max_value=2000, value=200, step=20, key="analysis_eval_queries")
    )
    analysis_alpha = float(
        st.slider("Hybrid alpha", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="analysis_alpha")
    )
    analysis_k_values = st.multiselect(
        "k values for evaluation",
        options=[5, 10, 15, 20, 25, 30],
        default=[5, 10, 20],
        key="analysis_k_values",
    )
    analysis_top_k = max(analysis_k_values) if analysis_k_values else 10
    st.caption(f"ANN vs Exact and query-type view will use k={analysis_top_k}.")

    if not analysis_k_values:
        st.warning("Pick at least one k value for evaluation.")
    else:
        if st.button("Run / Refresh Analysis", use_container_width=True):
            with st.spinner("Running evaluation experiments..."):
                eval_results = engine.evaluate_k(
                    k_values=tuple(sorted(analysis_k_values)),
                    methods=("bm25", "bert", "hybrid"),
                    n_queries=analysis_eval_queries,
                    alpha=analysis_alpha,
                )
                ann_exact = engine.compare_ann_exact(k=analysis_top_k, n_queries=analysis_eval_queries)
                query_type = engine.query_type_performance(
                    methods=("bm25", "bert", "hybrid"),
                    k=analysis_top_k,
                    n_queries=analysis_eval_queries,
                    alpha=analysis_alpha,
                )
                st.session_state["analysis"] = {
                    "eval_results": eval_results,
                    "ann_exact": ann_exact,
                    "query_type": query_type,
                }

    analysis = st.session_state.get("analysis")
    if analysis:
        rows = []
        for method, by_k in analysis["eval_results"].items():
            for k, vals in by_k.items():
                rows.append(
                    {
                        "method": method.upper(),
                        "k": int(k),
                        "precision": vals["precision"],
                        "recall": vals["recall"],
                    }
                )
        eval_df = pd.DataFrame(rows).sort_values(["k", "method"])
        st.markdown("**Precision vs k**")
        st.line_chart(eval_df.pivot(index="k", columns="method", values="precision"))
        st.markdown("**Recall vs k**")
        st.line_chart(eval_df.pivot(index="k", columns="method", values="recall"))
        st.dataframe(eval_df, use_container_width=True)

        ann_table = pd.DataFrame(
            [
                {
                    "k": int(analysis["ann_exact"]["k"]),
                    "queries": int(analysis["ann_exact"]["num_queries"]),
                    "ANN avg time (ms)": analysis["ann_exact"]["ann_avg_time_ms"],
                    "Exact avg time (ms)": analysis["ann_exact"]["exact_avg_time_ms"],
                    "ANN avg recall": analysis["ann_exact"]["ann_avg_recall"],
                    "Exact avg recall": analysis["ann_exact"]["exact_avg_recall"],
                    "ANN/Exact overlap@k": analysis["ann_exact"]["result_overlap_at_k"],
                }
            ]
        )
        st.markdown("**ANN vs Exact Timing and Quality**")
        st.dataframe(ann_table, use_container_width=True)

        qt_rows = []
        for method, buckets in analysis["query_type"].items():
            for q_type, vals in buckets.items():
                qt_rows.append(
                    {
                        "method": method.upper(),
                        "query_type": q_type,
                        "precision": vals["precision"],
                        "num_queries": int(vals["num_queries"]),
                    }
                )
        qt_df = pd.DataFrame(qt_rows)
        st.markdown("**Query Type Performance (Precision@k)**")
        st.bar_chart(qt_df.pivot(index="query_type", columns="method", values="precision"))
        st.dataframe(qt_df, use_container_width=True)

with tabs[3]:
    st.subheader("PCA Embedding Visualization")
    sample_size = int(
        st.slider("PCA sample size", min_value=100, max_value=min(5000, len(engine.documents)), value=1000, step=100)
    )
    if st.button("Generate PCA", use_container_width=True):
        with st.spinner("Computing PCA projection..."):
            points = engine.pca_projection(sample_size=sample_size)
            pca_df = pd.DataFrame(points)
            st.session_state["pca_df"] = pca_df

    pca_df = st.session_state.get("pca_df")
    if pca_df is not None and not pca_df.empty:
        st.scatter_chart(pca_df, x="x", y="y")
        st.dataframe(pca_df[["doc_id", "snippet"]].head(20), use_container_width=True)

with tabs[4]:
    st.subheader("Understanding Alpha, Scores, and Comparison")

    st.markdown(
        """
### 1) What is `alpha` in Hybrid Search?

Hybrid combines semantic (BERT) and lexical (BM25) signals:

`final_score = alpha * bert_norm + (1 - alpha) * bm25_norm`

- `alpha = 1.0` -> only BERT matters
- `alpha = 0.0` -> only BM25 matters
- `alpha = 0.5` -> equal balance

If your query needs meaning/synonyms, increase alpha.  
If your query needs exact keywords, decrease alpha.
"""
    )

    st.markdown(
        """
### 2) Why are score numbers different across models?

Raw scores come from different scoring systems:

- **BERT** score: vector similarity (roughly in a small range, often near 0 to 1 here due to normalized embeddings)
- **BM25** score: term-matching score (can be much larger, e.g. 5, 20, 40...)
- **Hybrid** score: normalized fusion value (typically near 0 to 1)

So **do not compare raw score values across different models**.
"""
    )

    st.markdown(
        """
### 3) How should a common user compare results?

Use these rules:

1. Compare **rank order** inside each model (Rank #1 is best for that model).
2. Compare **document quality** across models (which top results are more relevant to your query).
3. Use dashboard metrics (`Precision@k`, `Recall@k`) for model-level performance.
4. Treat raw scores as **confidence within the same model**, not between models.
"""
    )

    st.markdown(
        """
### 4) Quick interpretation of your example

- BERT score `0.8`, BM25 score `24`, Hybrid score `1.0`
- This does **not** mean Hybrid is "better by number magnitude".
- It only means each model used its own scale.
- Correct comparison: check which model returns more relevant top-k documents.
"""
    )

    st.info(
        "Tip: In this app, use Comparison View for one-query quality checks and Analysis Dashboard for overall model evaluation."
    )

    st.markdown(
        """
### 5) What is an embedding in this project?

An **embedding** is a numeric vector representation of text.

- We convert each document into a 384-dimensional vector using `all-MiniLM-L6-v2`.
- Query text is also converted into a vector.
- Search then finds documents with vectors close to the query vector.

Why this helps:
- It can match meaning, not just exact words.
- Useful when users phrase questions differently from document wording.
"""
    )

    st.markdown(
        """
### 6) What is PCA and why is it here?

PCA (Principal Component Analysis) reduces high-dimensional vectors to 2D for visualization.

- In this app, PCA does **not** perform retrieval.
- It helps you visually inspect whether embeddings form meaningful clusters.

Simple use:
- Open **PCA Visualization** tab
- Generate projection
- Observe whether similar topics appear near each other
"""
    )

    st.markdown(
        """
### 7) What each app tab means for this project

- **Search**: Get top-k results for one query from one model.
- **Comparison View**: Same query across BERT, BM25, Hybrid to see differences.
- **Analysis Dashboard**: Dataset-level evaluation over many queries (Precision/Recall, ANN vs Exact, query types).
- **PCA Visualization**: Understand embedding space structure visually.
- **How To Read Results**: Learn how to interpret all outputs correctly.
"""
    )

    st.markdown(
        """
### 8) How a user can use this to make life easier

1. If you just need quick answers: use **Search**.
2. If results feel weak: use **Comparison View** and pick model behavior that fits your query style.
3. If you are building a system: use **Analysis Dashboard** to choose model and `k` based on metrics.
4. If you want trust/intuition in semantic behavior: check **PCA Visualization**.

Real-world examples:
- Student: find concept explanations even with paraphrased questions.
- Researcher: compare lexical vs semantic retrieval quality.
- Product builder: tune `alpha`, `k`, and index type using measurable evidence.
"""
    )
