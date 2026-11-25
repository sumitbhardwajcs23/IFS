"""Phase 7: Streamlit UI for Hinglish Government Gazette Search & Ablation Comparison."""

from __future__ import annotations

import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st

from src.config import (
    CATEGORY_TARGETS,
    DOCUMENTS_JSONL_PATH,
    EVAL_CATEGORY_CSV_PATH,
    EVAL_COMPARISON_CSV_PATH,
    EVAL_RESULTS_PATH,
    MANIFEST_PATH,
)
from src.phase1_dataset import dataset_target_progress
from src.phase4_bm25 import BM25Retriever
from src.phase5_semantic import SemanticRetriever
from src.phase6_hybrid import HybridRetriever
from src.storage import read_jsonl

API_URL = os.environ.get("GAZETTE_API_URL", "http://127.0.0.1:8000")

SAMPLE_QUERIES = [
    "Sarkari karmchari ki padonnati aur vetan matrix ke niyam",
    "Jameen adhigrahan aur bhumi rajaswa notification",
    "Mahila kalyan vibhag bharti aur chayan aadesh",
    "Judge sahab ki adalat posting transfer order",
    "Chikitsalaya ayurved doctor bharti vigyapti",
    "Nagar nigam gram panchayat safai aur jal niyam",
]


@st.cache_resource(show_spinner="Loading BM25 + Multilingual-E5 + FAISS indices...")
def load_local_retriever() -> HybridRetriever:
    bm25 = BM25Retriever.load()
    semantic = SemanticRetriever.load()
    return HybridRetriever(bm25_retriever=bm25, semantic_retriever=semantic)


def _api_is_available() -> bool:
    try:
        resp = httpx.get(f"{API_URL}/health", timeout=1.2)
        return resp.status_code == 200
    except Exception:
        return False


def _run_search(
    query: str,
    top_k: int,
    mode: str,
    use_normalization: bool,
    category: str | None,
    prefer_api: bool,
) -> tuple[dict[str, Any], str]:
    if prefer_api:
        try:
            resp = httpx.post(
                f"{API_URL}/search",
                json={
                    "query": query,
                    "top_k": top_k,
                    "mode": mode,
                    "use_normalization": use_normalization,
                    "category": category,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json(), "FastAPI Backend"
        except Exception:
            pass

    retriever = load_local_retriever()
    return (
        retriever.search(
            query=query,
            top_k=top_k,
            mode=mode,
            use_normalization=use_normalization,
            category=category,
        ),
        "Local Python Engine",
    )


def _run_compare(
    query: str,
    top_k: int,
    category: str | None,
    prefer_api: bool,
) -> dict[str, Any]:
    if prefer_api:
        try:
            resp = httpx.post(
                f"{API_URL}/compare",
                json={"query": query, "top_k": top_k, "category": category},
                timeout=45.0,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            pass

    retriever = load_local_retriever()
    return retriever.compare(query=query, top_k=top_k, category=category)


def _render_result_card(item: dict[str, Any], show_scores: bool = True) -> None:
    rank = item.get("rank", "-")
    heading = item.get("heading") or item.get("title") or "Gazette Clause"
    title = item.get("title") or item.get("doc_id")
    category = item.get("category", "Unknown")
    ministry = item.get("ministry", "")
    notif_no = item.get("notification_no", "")
    pages = f"p. {item.get('page_start')}–{item.get('page_end')}"

    with st.container(border=True):
        st.markdown(f"**#{rank} — {heading[:140]}**")
        meta_parts = [f"**Source:** {title}", f"**Category:** {category}", f"**Pages:** {pages}"]
        if ministry:
            meta_parts.append(f"**Dept:** {ministry}")
        if notif_no:
            meta_parts.append(f"**Ref:** `{notif_no}`")
        st.caption(" | ".join(meta_parts))

        if show_scores:
            cols = st.columns(3)
            cols[0].metric("RRF Score", f"{item['rrf_score']:.4f}" if item.get("rrf_score") is not None else "—")
            bm25_label = f"#{item['bm25_rank']} ({item['bm25_score']:.2f})" if item.get("bm25_rank") else "—"
            sem_label = f"#{item['semantic_rank']} ({item['semantic_score']:.3f})" if item.get("semantic_rank") else "—"
            cols[1].metric("BM25", bm25_label)
            cols[2].metric("Semantic (E5)", sem_label)

        st.write(item.get("text", ""))


def main() -> None:
    st.set_page_config(
        page_title="Hinglish Gazette Search (IR)",
        page_icon="📜",
        layout="wide",
    )

    st.title("Hinglish-Gazette-IR")
    st.caption(
        "Phonetic & Dialect-Aware Hybrid Information Retrieval (BM25 + Multilingual-E5 + RRF) "
        "over Formal Government Gazettes"
    )

    api_up = _api_is_available()

    with st.sidebar:
        st.header("Retrieval Settings")
        mode = st.selectbox(
            "Retrieval Mode",
            options=["hybrid", "bm25", "semantic"],
            format_func=lambda m: {
                "hybrid": "Hybrid RRF (BM25 + Multilingual-E5)",
                "bm25": "BM25 Keyword Only",
                "semantic": "Semantic (Multilingual-E5) Only",
            }[m],
        )
        use_norm = st.checkbox("Enable Phonetic & Dialect Normalization", value=True)
        categories = ["All"] + list(CATEGORY_TARGETS.keys())
        selected_cat = st.selectbox("Category Filter", options=categories)
        category_filter = None if selected_cat == "All" else selected_cat
        top_k = st.slider("Top-K Results", min_value=3, max_value=20, value=8)

        st.divider()
        st.markdown(f"**Backend:** {'FastAPI (`' + API_URL + '`)' if api_up else 'Direct Local Index'}")

    tab_search, tab_compare, tab_eval, tab_corpus = st.tabs(
        [
            "Search Gazettes",
            "Compare BM25 vs Semantic vs Hybrid",
            "Evaluation Benchmark (200 Queries)",
            "Dataset & Corpus",
        ]
    )

    with tab_search:
        st.write("**Try an example Hinglish / dialect query:**")
        ex_cols = st.columns(3)
        if "search_query" not in st.session_state:
            st.session_state["search_query"] = SAMPLE_QUERIES[0]
        for idx, sample in enumerate(SAMPLE_QUERIES):
            if ex_cols[idx % 3].button(sample, key=f"sample_{idx}", use_container_width=True):
                st.session_state["search_query"] = sample

        query = st.text_input(
            "Enter your query (Hindi, Hinglish, spelling variants, or English):",
            key="search_query",
        )

        if st.button("Search Gazette Clauses", type="primary") or query:
            if query.strip():
                payload, source_label = _run_search(
                    query=query.strip(),
                    top_k=top_k,
                    mode=mode,
                    use_normalization=use_norm,
                    category=category_filter,
                    prefer_api=api_up,
                )

                norm = payload.get("normalization", {})
                with st.expander("Query Normalization & Explainability Trace", expanded=True):
                    c1, c2 = st.columns(2)
                    c1.markdown(f"**BM25 Expanded Tokens:** `{' '.join(norm.get('bm25_tokens', [])[:24])}`")
                    c2.markdown(f"**Semantic E5 Query:** `{norm.get('semantic_query', query)}`")
                    if norm.get("explanation"):
                        for step in norm["explanation"]:
                            st.write(f"- {step}")
                    else:
                        st.write("- No phonetic/dialect rewrites triggered; using raw tokens.")

                results = payload.get("results", [])
                st.subheader(
                    f"Top {len(results)} Relevant Gazette Clauses "
                    f"({payload.get('elapsed_ms', 0)} ms via {source_label})"
                )
                if not results:
                    st.info("No matching Gazette clauses found. Try removing the category filter or broadening your query.")
                for item in results:
                    _render_result_card(item)

    with tab_compare:
        cmp_query = st.text_input(
            "Query for Side-by-Side Model Comparison:",
            value=st.session_state.get("search_query", SAMPLE_QUERIES[0]),
            key="cmp_query",
        )
        cmp_k = st.slider("Results per method", min_value=3, max_value=10, value=5, key="cmp_k")
        if st.button("Run Side-by-Side Comparison", type="primary"):
            if cmp_query.strip():
                cmp_data = _run_compare(
                    query=cmp_query.strip(),
                    top_k=cmp_k,
                    category=category_filter,
                    prefer_api=api_up,
                )
                cols = st.columns(4)
                panels = [
                    ("1. Raw BM25", cmp_data["bm25_raw"]),
                    ("2. Normalized BM25", cmp_data["bm25_normalized"]),
                    ("3. Multilingual-E5", cmp_data["semantic"]),
                    ("4. Proposed Hybrid RRF", cmp_data["hybrid_rrf"]),
                ]
                for col, (label, hits) in zip(cols, panels):
                    with col:
                        st.markdown(f"### {label}")
                        if not hits:
                            st.info("No hits")
                        for h in hits:
                            with st.container(border=True):
                                st.markdown(f"**#{h.get('rank', '-')}** ({h['doc_id']}, p.{h['page_start']})")
                                st.caption((h.get("heading") or "")[:90])
                                st.write((h.get("text") or "")[:240] + "...")

    with tab_eval:
        if EVAL_COMPARISON_CSV_PATH.exists() and EVAL_CATEGORY_CSV_PATH.exists():
            import json

            st.subheader("Phase 8 Benchmark — 200 Hinglish Queries (Chunk-Level @10)")
            comp_df = pd.read_csv(EVAL_COMPARISON_CSV_PATH)
            st.dataframe(comp_df.drop(columns=["system_key"]), use_container_width=True)

            chart_df = comp_df.set_index("system")[["precision@10", "recall@10", "mrr@10", "ndcg@10"]]
            st.bar_chart(chart_df)

            if EVAL_RESULTS_PATH.exists():
                eval_json = json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
                gains = eval_json.get("improvements_pct", {}).get("vs_bm25_raw", {})
                if gains:
                    st.markdown("**Hybrid RRF Relative Gain vs. Raw BM25 Baseline:**")
                    gcols = st.columns(4)
                    gcols[0].metric("Precision@10", f"+{gains.get('precision@10', 0):.1f}%")
                    gcols[1].metric("Recall@10", f"+{gains.get('recall@10', 0):.1f}%")
                    gcols[2].metric("MRR@10", f"+{gains.get('mrr@10', 0):.1f}%")
                    gcols[3].metric("nDCG@10", f"+{gains.get('ndcg@10', 0):.1f}%")

            st.subheader("Breakdown by Hinglish Query Style (50 Queries Each)")
            cat_df = pd.read_csv(EVAL_CATEGORY_CSV_PATH)
            st.dataframe(cat_df.drop(columns=["system_key"]), use_container_width=True)
        else:
            st.info("Run `python scripts/run_phase8_eval.py` to generate the 200-query benchmark report.")

    with tab_corpus:
        if MANIFEST_PATH.exists():
            manifest_df = pd.read_csv(MANIFEST_PATH, dtype=str).fillna("")
            progress = dataset_target_progress(manifest_df)
            m1, m2, m3 = st.columns(3)
            m1.metric("Collected PDFs", f"{progress['total_current']} / {progress['total_target']}")
            if DOCUMENTS_JSONL_PATH.exists():
                docs = read_jsonl(DOCUMENTS_JSONL_PATH)
                m2.metric("Total Pages", sum(d["page_count"] for d in docs))
                m3.metric("Total Characters", f"{sum(d['char_count'] for d in docs):,}")
            st.subheader("Manifest Catalog")
            st.dataframe(manifest_df, use_container_width=True)
        else:
            st.warning("Run `python scripts/run_phase1.py` first to populate the dataset manifest.")


if __name__ == "__main__":
    main()
