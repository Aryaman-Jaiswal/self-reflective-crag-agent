"""
Streamlit Web Application for Self-Reflective Corrective RAG (CRAG) Agent.
Features real-time execution tracing, interactive query answering, document ingestion, and RAGAS benchmarking.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
import pandas as pd
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="CRAG Agent | Self-Reflective Corrective RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics, glassmorphism, and responsive layout
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Glassmorphism */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(18, 24, 38, 1) 0%, rgba(10, 14, 23, 1) 90%);
        color: #E2E8F0;
    }

    /* Header Banner */
    .header-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    .title-gradient {
        background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
    }

    /* Custom Cards */
    .crag-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .crag-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
        transform: translateY(-2px);
    }

    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-relevant {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-irrelevant {
        background-color: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .badge-grounded {
        background-color: rgba(14, 165, 233, 0.2);
        color: #38BDF8;
        border: 1px solid rgba(14, 165, 233, 0.4);
    }
    .badge-rewrite {
        background-color: rgba(245, 158, 11, 0.2);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .badge-score {
        background-color: rgba(99, 102, 241, 0.2);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.4);
    }

    /* Citation Tag */
    .citation-tag {
        color: #38BDF8;
        background: rgba(56, 189, 248, 0.15);
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85em;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }

    /* Metric Box */
    .metric-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38BDF8;
        font-family: 'Outfit', sans-serif;
    }
    .metric-lbl {
        font-size: 0.82rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# Import project modules
from src.config import settings
from src.ingestion import DocumentIngestor, EmbeddingWrapper, HybridRetriever
from src.reranker import CrossEncoderReranker, DocumentChunk
from src.agent_graph import CRAGPipeline
from src.evaluate import BenchmarkEvaluator, SYNTHETIC_BENCHMARK_DATASET
from src.llm_factory import get_llm


# ------------------ STATE INITIALIZATION ------------------

@st.cache_resource(show_spinner="Initializing CRAG Knowledge Base & Models...")
def initialize_knowledge_base():
    """Initializes embeddings, sample document chunks, and in-memory Qdrant index."""
    ingestor = DocumentIngestor(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    emb_wrapper = EmbeddingWrapper(model_name=settings.embedding_model_name)
    
    # Ingest default sample docs
    sample_files = list(settings.sample_docs_dir.glob("*.*"))
    chunks = ingestor.process_and_chunk(sample_files) if sample_files else []
    
    retriever = HybridRetriever(
        chunks=chunks,
        embedding_wrapper=emb_wrapper,
        dense_weight=settings.dense_weight,
        sparse_weight=settings.sparse_weight,
        collection_name=settings.qdrant_collection_name,
        qdrant_location=settings.qdrant_location
    )
    
    reranker = CrossEncoderReranker(model_name=settings.reranker_model_name)
    return ingestor, emb_wrapper, retriever, reranker, chunks


ingestor, emb_wrapper, retriever, reranker, current_chunks = initialize_knowledge_base()

if "uploaded_chunks" not in st.session_state:
    st.session_state.uploaded_chunks = current_chunks
if "pipeline_history" not in st.session_state:
    st.session_state.pipeline_history = []
if "eval_report" not in st.session_state:
    st.session_state.eval_report = None


# ------------------ SIDEBAR CONTROLS ------------------

with st.sidebar:
    st.markdown("### ⚙️ Pipeline Configuration")
    
    provider = st.selectbox(
        "LLM Provider",
        options=["OpenAI", "Google Gemini", "Groq", "Smart Rule-Based (Offline)"],
        index=0 if settings.openai_api_key else (1 if settings.google_api_key else 3)
    )
    
    provider_map = {
        "OpenAI": "openai",
        "Google Gemini": "gemini",
        "Groq": "groq",
        "Smart Rule-Based (Offline)": "local"
    }
    selected_provider = provider_map[provider]
    
    model_name = "gpt-4o-mini"
    if selected_provider == "openai":
        model_name = st.selectbox("OpenAI Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
        api_key_input = st.text_input("OpenAI API Key", value=settings.openai_api_key or "", type="password")
        if api_key_input:
            settings.openai_api_key = api_key_input
    elif selected_provider == "gemini":
        gemini_options = [
            "gemini-flash-lite-latest",
            "gemini-3.5-flash-lite",
            "gemini-3.6-flash",
            "gemini-3.1-flash-lite",
            "gemini-3-flash-preview",
            "gemini-3.7-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ]
        model_name = st.selectbox("Gemini Model", gemini_options, index=0)
        api_key_input = st.text_input("Google API Key", value=settings.google_api_key or "", type="password")
        if api_key_input:
            settings.google_api_key = api_key_input

    st.markdown("---")
    st.markdown("### 🎯 Hybrid Retrieval Weights")
    dense_w = st.slider("Dense (Qdrant) Weight", min_value=0.0, max_value=1.0, value=settings.dense_weight, step=0.05)
    sparse_w = round(1.0 - dense_w, 2)
    st.caption(f"Sparse (BM25) Weight: **{sparse_w}**")
    
    st.markdown("---")
    st.markdown("### 🔍 Search & Reranking Limits")
    top_k_ret = st.slider("Top-K Candidates", min_value=3, max_value=20, value=settings.top_k_retrieval)
    top_k_rer = st.slider("Top-K Reranked (to LLM)", min_value=1, max_value=5, value=settings.top_k_rerank)
    max_retries = st.slider("Max Query Rewrites", min_value=1, max_value=4, value=settings.max_retries)

    st.markdown("---")
    st.markdown("### 📂 Upload New Documents")
    uploaded_files = st.file_uploader("Upload PDF, Markdown, or TXT", accept_multiple_files=True, type=["pdf", "md", "txt"])
    
    if uploaded_files:
        if st.button("🚀 Ingest & Re-index", use_container_width=True):
            with st.spinner("Processing documents & building hybrid indexes..."):
                temp_paths = []
                for f in uploaded_files:
                    temp_path = settings.data_dir / "uploads" / f.name
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(temp_path, "wb") as out_f:
                        out_f.write(f.getbuffer())
                    temp_paths.append(temp_path)

                new_chunks = ingestor.process_and_chunk(temp_paths)
                all_chunks = st.session_state.uploaded_chunks + new_chunks
                st.session_state.uploaded_chunks = all_chunks

                # Rebuild retriever
                retriever = HybridRetriever(
                    chunks=all_chunks,
                    embedding_wrapper=emb_wrapper,
                    dense_weight=dense_w,
                    sparse_weight=sparse_w,
                    collection_name=settings.qdrant_collection_name,
                    qdrant_location=settings.qdrant_location
                )
                st.success(f"Indexed {len(new_chunks)} new chunks! Total knowledge chunks: {len(all_chunks)}")


# Instantiate Agent with current configuration
active_llm = get_llm(provider=selected_provider, model_name=model_name)
agent = CRAGPipeline(retriever=retriever, reranker=reranker, llm=active_llm)


# ------------------ MAIN UI CONTENT ------------------

st.markdown("""
<div class="header-banner">
    <div class="title-gradient">Self-Reflective Corrective RAG (CRAG) Agent</div>
    <div style="color: #94A3B8; margin-top: 6px; font-size: 0.95rem;">
        Production-grade agentic RAG powered by LangGraph, Qdrant Hybrid Search, Cross-Encoder Reranking, Reflection Loops & RAGAS.
    </div>
</div>
""", unsafe_allow_html=True)

tab_chat, tab_eval, tab_corpus = st.tabs([
    "💬 Interactive CRAG & Execution Trace",
    "📊 RAGAS Benchmark & Quality Dashboard",
    "🗄️ Knowledge Base & Chunk Explorer"
])


# ==========================================
# TAB 1: INTERACTIVE CRAG & LIVE TRACE
# ==========================================
with tab_chat:
    col_input, col_chips = st.columns([3, 2])
    with col_input:
        user_query = st.text_input(
            "Ask a question to the CRAG agent:",
            placeholder="e.g. What is the mathematical scaling factor in scaled dot-product attention?",
            key="main_query_input"
        )
    with col_chips:
        st.markdown("**Sample Prompt Chips:**")
        sample_prompts = [
            "What is the mathematical scaling factor in scaled dot-product attention?",
            "What are the discrete outcomes produced by the CRAG retrieval evaluator?",
            "What is the max token retention period under Enterprise Security Policy?",
            "What happens if all documents are irrelevant?"
        ]
        chip_col1, chip_col2 = st.columns(2)
        for i, p in enumerate(sample_prompts):
            target_col = chip_col1 if i % 2 == 0 else chip_col2
            if target_col.button(f"📌 {p[:32]}...", key=f"chip_{i}", use_container_width=True):
                user_query = p

    if st.button("⚡ Run CRAG Agent", type="primary", use_container_width=True) or user_query:
        if user_query:
            with st.spinner("Executing LangGraph State Machine..."):
                t_start = time.time()
                pipeline_state = agent.run(user_query)
                latency = round(time.time() - t_start, 3)

            # 1. Final Answer Display
            st.markdown("### 📝 Synthesized Grounded Response")
            generation_text = pipeline_state.get("generation", "No response generated.")
            
            with st.container(border=True):
                st.markdown(generation_text)
                st.markdown(
                    f"<div style='margin-top: 14px; font-size: 0.82rem; color: #94A3B8; border-top: 1px solid rgba(51, 65, 85, 0.7); padding-top: 10px;'>"
                    f"⏱️ Execution Latency: <b>{latency}s</b> &nbsp;|&nbsp; Retries: <b>{pipeline_state.get('retry_count', 0)}</b> &nbsp;|&nbsp; "
                    f"Status: <span style='background: rgba(16, 185, 129, 0.2); color: #34D399; padding: 2px 10px; border-radius: 9999px; font-weight: 600;'>{pipeline_state.get('hallucination_grade', 'GROUNDED').upper()}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            # 2. Detailed Execution Trace Visualizer
            st.markdown("### 🔬 Agent Execution Trace & Reflection Loops")
            trace_steps = pipeline_state.get("trace_steps", [])

            for step_idx, step in enumerate(trace_steps, start=1):
                step_name = step.get("step")

                # Step: Hybrid Retrieval & Reranking
                if step_name == "retrieve_and_rerank":
                    with st.expander(f"📍 Step {step_idx}: Hybrid Search & Cross-Encoder Reranking", expanded=True):
                        st.markdown(f"**Search Query Used:** `{step.get('active_query')}`")
                        st.caption(f"Retrieved {step.get('candidates_retrieved')} candidates → Cross-encoder scored top {step.get('top_reranked_count')}")
                        
                        chunks_data = step.get("chunks", [])
                        for c in chunks_data:
                            with st.container(border=True):
                                col_hdr1, col_hdr2 = st.columns([1, 2])
                                with col_hdr1:
                                    st.markdown(f"**#{c.get('rank')} | Source: `{c.get('source')}`**")
                                with col_hdr2:
                                    st.markdown(
                                        f"<div style='text-align: right;'>"
                                        f"<span class='badge badge-score'>Dense: {c.get('dense_score')}</span> "
                                        f"<span class='badge badge-score'>Sparse: {c.get('sparse_score')}</span> "
                                        f"<span class='badge badge-score'>Hybrid: {c.get('hybrid_score')}</span> "
                                        f"<span class='badge badge-grounded'>Cross-Encoder: {c.get('rerank_score')}</span>"
                                        f"</div>",
                                        unsafe_allow_html=True
                                    )
                                st.markdown(c.get('content_snippet', ''))

                # Step: Document Relevance Grading
                elif step_name == "grade_documents":
                    with st.expander(f"📍 Step {step_idx}: Document Relevance Grading", expanded=True):
                        st.caption(f"Evaluated {step.get('total_evaluated')} chunks → {step.get('vetted_count')} marked relevant for answer synthesis.")
                        for d in step.get("details", []):
                            is_rel = d.get("grade") == "relevant"
                            badge_class = "badge-relevant" if is_rel else "badge-irrelevant"
                            with st.container(border=True):
                                col_g1, col_g2 = st.columns([3, 1])
                                with col_g1:
                                    st.markdown(f"**Chunk ID: `{d.get('chunk_id')}`** ({d.get('source')})")
                                with col_g2:
                                    st.markdown(f"<div style='text-align: right;'><span class='badge {badge_class}'>{d.get('grade').upper()}</span></div>", unsafe_allow_html=True)
                                st.markdown(f"**Rationale:** {d.get('rationale')}")
                                st.markdown(f"**Chunk Context:**\n\n{d.get('content_preview')}")

                # Step: Query Rewriting
                elif step_name == "rewrite_query":
                    with st.expander(f"📍 Step {step_idx}: Query Transformation & Semantic Expansion", expanded=True):
                        with st.container(border=True):
                            st.markdown(f"<span class='badge badge-rewrite'>Retry Attempt #{step.get('retry_count')}</span>", unsafe_allow_html=True)
                            st.markdown(f"**Original Query:** `{step.get('original_query')}`")
                            st.markdown(f"**Optimized Rewritten Query:** `{step.get('rewritten_query')}`")
                            st.markdown(f"**Reasoning:** {step.get('reasoning')}")

                # Step: Hallucination Verification
                elif step_name == "hallucination_check":
                    with st.expander(f"📍 Step {step_idx}: Self-Reflective Hallucination Audit", expanded=True):
                        is_grounded = step.get("grade") == "grounded"
                        badge_class = "badge-grounded" if is_grounded else "badge-irrelevant"
                        with st.container(border=True):
                            col_h1, col_h2 = st.columns([3, 1])
                            with col_h1:
                                st.markdown("**Faithfulness Audit Assessment**")
                            with col_h2:
                                st.markdown(f"<div style='text-align: right;'><span class='badge {badge_class}'>{step.get('grade').upper()}</span></div>", unsafe_allow_html=True)
                            st.markdown(f"**Assessment:** {step.get('rationale')}")

                # Step: Fallback Generation
                elif step_name == "generate_fallback":
                    with st.expander(f"📍 Step {step_idx}: Fallback Handling", expanded=True):
                        with st.container(border=True):
                            st.markdown("<span class='badge badge-irrelevant'>Graceful Fallback</span>", unsafe_allow_html=True)
                            st.markdown(f"{step.get('reason')} (Exhausted {step.get('retry_count')} rewrite attempts).")


# ==========================================
# TAB 2: RAGAS BENCHMARK & EVALUATION
# ==========================================
with tab_eval:
    st.markdown("### 📊 Automated RAGAS Benchmark Suite")
    st.markdown("Run automated evaluation over a curated synthetic test dataset measuring **Faithfulness**, **Answer Relevancy**, **Context Precision**, and **Context Recall**.")

    col_scope, col_btn = st.columns([2, 1])
    with col_scope:
        bench_count = st.radio(
            "Benchmark Scope:",
            [3, 5, 10],
            index=0,
            horizontal=True,
            format_func=lambda x: f"⚡ Quick ({x} Queries)" if x == 3 else (f"🔍 Standard ({x} Queries)" if x == 5 else f"🚀 Full Suite ({x} Queries)")
        )
    with col_btn:
        st.write("")
        st.write("")
        run_eval_btn = st.button(f"🚀 Run Benchmark ({bench_count})", type="primary", use_container_width=True)

    if run_eval_btn:
        progress_bar = st.progress(0.0)
        status_box = st.empty()

        def update_progress(current_idx: int, total: int, q_text: str):
            pct = float(current_idx - 1) / float(total)
            progress_bar.progress(pct)
            status_box.info(f"⏳ **Evaluating query [{current_idx}/{total}]**: *{q_text[:70]}...*")

        evaluator = BenchmarkEvaluator(agent=agent)
        from src.evaluate import SYNTHETIC_BENCHMARK_DATASET
        subset = SYNTHETIC_BENCHMARK_DATASET[:bench_count]
        
        report = evaluator.evaluate_pipeline(test_dataset=subset, progress_callback=update_progress, save_report=True)
        progress_bar.progress(1.0)
        status_box.empty()
        progress_bar.empty()
        st.session_state.eval_report = report
        st.success(f"Benchmark evaluation across {len(subset)} queries completed successfully!")

    if st.session_state.eval_report is not None:
        report = st.session_state.eval_report
        summary = report["summary"]
        details = report["detailed_results"]
        df_details = pd.DataFrame(details)

        # Top KPI Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{summary['overall_crag_score']}</div>
            <div class="metric-lbl">Overall CRAG Score</div>
        </div>
        """, unsafe_allow_html=True)

        m2.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{summary['mean_faithfulness']}</div>
            <div class="metric-lbl">Faithfulness</div>
        </div>
        """, unsafe_allow_html=True)

        m3.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{summary['mean_answer_relevancy']}</div>
            <div class="metric-lbl">Answer Relevancy</div>
        </div>
        """, unsafe_allow_html=True)

        m4.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{summary['mean_context_precision']}</div>
            <div class="metric-lbl">Context Precision</div>
        </div>
        """, unsafe_allow_html=True)

        m5.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{summary['mean_context_recall']}</div>
            <div class="metric-lbl">Context Recall</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Category Breakdown Charts
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("#### Metric Scores by Test Case")
            metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
            st.bar_chart(df_details.set_index("question")[metric_cols])

        with chart_col2:
            st.markdown("#### Latency & Retries by Test Case")
            st.line_chart(df_details.set_index("id")[["latency_sec", "retries"]])

        # Detailed Table
        st.markdown("#### Detailed Benchmark Test Results")
        display_df = df_details.drop(columns=["retrieved_contexts"])
        st.dataframe(display_df, use_container_width=True)

        # Export Buttons
        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            st.download_button(
                "📥 Download JSON Benchmark Report",
                data=json.dumps(report, indent=2),
                file_name="crag_ragas_report.json",
                mime="application/json",
                use_container_width=True
            )
        with c_dl2:
            st.download_button(
                "📥 Download CSV Benchmark Report",
                data=display_df.to_csv(index=False),
                file_name="crag_ragas_report.csv",
                mime="text/csv",
                use_container_width=True
            )


# ==========================================
# TAB 3: KNOWLEDGE BASE & CHUNK INSPECTOR
# ==========================================
with tab_corpus:
    st.markdown("### 🗄️ Knowledge Base & Chunk Inspector")
    total_indexed = len(st.session_state.uploaded_chunks)
    st.info(f"Total Chunks Currently Indexed in Qdrant & BM25: **{total_indexed}**")

    search_filter = st.text_input("Filter chunks by keyword:", placeholder="e.g. attention, token, encryption")

    filtered_chunks = st.session_state.uploaded_chunks
    if search_filter:
        filtered_chunks = [c for c in filtered_chunks if search_filter.lower() in c.page_content.lower()]

    for idx, c in enumerate(filtered_chunks[:20], start=1):
        with st.expander(f"Chunk #{idx} | Source: {c.metadata.get('source', 'unknown')} | Type: {c.metadata.get('type', 'text')}"):
            st.markdown(f"**Metadata:** `{c.metadata}`")
            st.markdown(f"**Content ({len(c.page_content)} characters):**")
            st.code(c.page_content, language="markdown")
