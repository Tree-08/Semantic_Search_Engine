import argparse
import json
from pathlib import Path
from typing import Dict, List

from retrieval_engine import SearchConfig, SemanticSearchEngine

FIXED_SUBSET_SIZE = 3_900
FIXED_CACHE_DIR = "cache"
FIXED_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def parse_k_values(raw: str) -> List[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("k-values cannot be empty.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic Search + BM25 + Hybrid on MS MARCO")
    parser.add_argument("--eval-queries", type=int, default=200)
    parser.add_argument("--k-values", type=str, default="5,10,20")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--model", choices=["bert", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--report-path", type=str, default="outputs/evaluation_report.json")
    return parser


def print_summary(summary: Dict[str, object]) -> None:
    print("=== System Summary ===")
    print(f"Device: {summary['device']}")
    print(f"Model: {summary['model_name']}")
    print(f"Documents: {summary['documents']}")
    print(f"Queries: {summary['queries']}")
    print(f"Embedding Dim: {summary['embedding_dim']}")
    print(f"Cache Dir: {summary['cache_dir']}")


def print_query_results(model_name: str, query: str, hits: List[Dict[str, object]]) -> None:
    print("\n=== Query Demo ===")
    print(f"Model: {model_name}")
    print(f"Query: {query}")
    for hit in hits:
        preview = hit["text"].replace("\n", " ").strip()
        if len(preview) > 220:
            preview = preview[:220] + "..."
        print(f"[{hit['rank']}] score={hit['score']:.4f} doc_id={hit['doc_id']} | {preview}")


def main() -> None:
    args = build_parser().parse_args()
    k_values = parse_k_values(args.k_values)

    config = SearchConfig(
        subset_size=FIXED_SUBSET_SIZE,
        eval_queries=args.eval_queries,
        cache_dir=FIXED_CACHE_DIR,
        model_name=FIXED_EMBEDDING_MODEL,
        hybrid_alpha=args.alpha,
        force_rebuild=False,
    )
    print(f"Starting pipeline. Preferred device: {config.device_preference}")
    engine = SemanticSearchEngine(config).load_or_build()
    print(f"Resolved runtime device: {engine.device}")
    print_summary(engine.system_summary())

    if args.query:
        if args.model == "bert":
            hits = engine.semantic_search(args.query, k=args.top_k)
        elif args.model == "bm25":
            hits = engine.bm25_search(args.query, k=args.top_k)
        else:
            hits = engine.hybrid_search(args.query, k=args.top_k, alpha=args.alpha)
        print_query_results(args.model, args.query, hits)

    if args.skip_eval:
        return

    eval_results = engine.evaluate_k(
        k_values=k_values,
        methods=("bm25", "bert", "hybrid"),
        n_queries=args.eval_queries,
        alpha=args.alpha,
    )
    ann_vs_exact = engine.compare_ann_exact(k=args.top_k, n_queries=args.eval_queries)
    query_type = engine.query_type_performance(
        methods=("bm25", "bert", "hybrid"),
        k=args.top_k,
        n_queries=args.eval_queries,
        alpha=args.alpha,
    )

    print("\n=== Precision/Recall by k ===")
    for method, by_k in eval_results.items():
        print(f"\n{method.upper()}:")
        for k in sorted(by_k.keys()):
            row = by_k[k]
            print(f"  k={k:<2} precision={row['precision']:.4f} recall={row['recall']:.4f}")

    print("\n=== ANN vs Exact ===")
    print(f"ANN avg time (ms): {ann_vs_exact['ann_avg_time_ms']:.4f}")
    print(f"Exact avg time (ms): {ann_vs_exact['exact_avg_time_ms']:.4f}")
    print(f"ANN avg recall: {ann_vs_exact['ann_avg_recall']:.4f}")
    print(f"Exact avg recall: {ann_vs_exact['exact_avg_recall']:.4f}")
    print(f"Overlap@k: {ann_vs_exact['result_overlap_at_k']:.4f}")

    print("\n=== Query Type Precision@k ===")
    for method, buckets in query_type.items():
        print(f"\n{method.upper()}:")
        for q_type, row in buckets.items():
            print(f"  {q_type:<8} precision={row['precision']:.4f} n={int(row['num_queries'])}")

    report = {
        "summary": engine.system_summary(),
        "k_values": k_values,
        "evaluation": eval_results,
        "ann_vs_exact": ann_vs_exact,
        "query_type_performance": query_type,
    }
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
