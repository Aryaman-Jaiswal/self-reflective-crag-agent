"""
Automated unit and integration test suite for the CRAG pipeline.
"""

import os
import json
import pytest
from pathlib import Path
from langchain_core.documents import Document

from src.config import settings
from src.ingestion import DocumentIngestor, EmbeddingWrapper, HybridRetriever
from src.reranker import CrossEncoderReranker, DocumentChunk
from src.agent_graph import CRAGPipeline, GraphState
from src.evaluate import BenchmarkEvaluator, SYNTHETIC_BENCHMARK_DATASET
from src.llm_factory import SmartRuleBasedLLM, get_llm


@pytest.fixture
def sample_documents(tmp_path):
    """Create sample documents for testing."""
    doc1 = tmp_path / "attention_test.md"
    doc1.write_text(
        "# Scaled Dot-Product Attention\n\n"
        "The attention mechanism computes Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V.\n"
        "The scaling factor 1/sqrt(d_k) counteracts large dot products pushing softmax to saturated regions.\n"
        "Multi-head attention projects queries, keys, and values into multiple lower-dimensional subspaces.",
        encoding="utf-8"
    )

    doc2 = tmp_path / "security_test.txt"
    doc2.write_text(
        "ENTERPRISE POLICY: Temporary access tokens must expire within 8 hours of issuance.\n"
        "All data at rest must use AES-256 encryption. Data in transit must use TLS 1.3.",
        encoding="utf-8"
    )

    return [doc1, doc2]


def test_document_ingestion_and_chunking(sample_documents):
    """Test document parsing, loading, and chunk splitting."""
    ingestor = DocumentIngestor(chunk_size=300, chunk_overlap=40)
    chunks = ingestor.process_and_chunk(sample_documents)

    assert len(chunks) >= 2, "Expected at least 2 chunks from sample docs"
    for chunk in chunks:
        assert isinstance(chunk, Document)
        assert len(chunk.page_content) > 20
        assert "source" in chunk.metadata
        assert "chunk_id" in chunk.metadata


def test_hybrid_retrieval(sample_documents):
    """Test Qdrant + BM25 ensemble hybrid search."""
    ingestor = DocumentIngestor(chunk_size=300, chunk_overlap=40)
    chunks = ingestor.process_and_chunk(sample_documents)

    emb_wrapper = EmbeddingWrapper()
    retriever = HybridRetriever(
        chunks=chunks,
        embedding_wrapper=emb_wrapper,
        dense_weight=0.6,
        sparse_weight=0.4,
        collection_name="test_collection",
        qdrant_location=":memory:"
    )

    results = retriever.retrieve("scaled dot product attention scaling factor", top_k=2)

    assert len(results) > 0, "Retriever should return candidate chunks"
    top_chunk = results[0]
    assert isinstance(top_chunk, DocumentChunk)
    assert top_chunk.hybrid_score is not None
    assert top_chunk.dense_score is not None
    assert top_chunk.sparse_score is not None
    assert "attention" in top_chunk.page_content.lower()


def test_cross_encoder_reranker():
    """Test CrossEncoder score computation and descending ranking."""
    reranker = CrossEncoderReranker()
    docs = [
        DocumentChunk(id="1", page_content="The cat sat on the mat.", metadata={"source": "pet.txt"}, hybrid_score=0.4),
        DocumentChunk(id="2", page_content="Attention is all you need with self-attention transformer layers.", metadata={"source": "transformer.txt"}, hybrid_score=0.7),
        DocumentChunk(id="3", page_content="Quantum computing qubits superposition.", metadata={"source": "quantum.txt"}, hybrid_score=0.2),
    ]

    reranked = reranker.rerank("transformer attention layers", docs, top_k=2)

    assert len(reranked) == 2
    assert reranked[0].id == "2"
    assert reranked[0].rerank_rank == 1
    assert reranked[0].rerank_score is not None


def test_crag_agent_happy_path(sample_documents):
    """Test end-to-end CRAG state machine execution on a relevant query."""
    ingestor = DocumentIngestor(chunk_size=300, chunk_overlap=40)
    chunks = ingestor.process_and_chunk(sample_documents)

    emb_wrapper = EmbeddingWrapper()
    retriever = HybridRetriever(
        chunks=chunks,
        embedding_wrapper=emb_wrapper,
        dense_weight=0.6,
        sparse_weight=0.4,
        qdrant_location=":memory:"
    )
    reranker = CrossEncoderReranker()
    llm = SmartRuleBasedLLM()

    pipeline = CRAGPipeline(retriever=retriever, reranker=reranker, llm=llm)
    result = pipeline.run("What is the scaling factor in attention?")

    assert result["query"] == "What is the scaling factor in attention?"
    assert result["generation"] is not None
    assert len(result["generation"]) > 10
    assert result["hallucination_grade"] == "grounded"
    assert len(result["trace_steps"]) >= 4, "Expected retrieve, grade, generate, hallucination_check steps"


def test_crag_agent_rewrite_loop(tmp_path):
    """Test CRAG state machine query rewrite loop on ambiguous/missing initial terms."""
    doc = tmp_path / "obscure.txt"
    doc.write_text("Specific corporate protocol AlphaX requires triple encryption.", encoding="utf-8")

    ingestor = DocumentIngestor()
    chunks = ingestor.process_and_chunk([doc])
    emb_wrapper = EmbeddingWrapper()
    retriever = HybridRetriever(chunks=chunks, embedding_wrapper=emb_wrapper, qdrant_location=":memory:")
    reranker = CrossEncoderReranker()

    class StrictGradingLLM(SmartRuleBasedLLM):
        def _generate(self, messages, **kwargs):
            text = "\n".join([m.content for m in messages if isinstance(m.content, str)])
            if "grade" in text.lower() and "AlphaX" not in text:
                return super()._generate([messages[0], type(messages[0])(content='{"score": "no", "rationale": "Irrelevant"}')])
            return super()._generate(messages, **kwargs)

    pipeline = CRAGPipeline(retriever=retriever, reranker=reranker, llm=StrictGradingLLM())
    result = pipeline.run("What is protocol AlphaX?")

    assert result["generation"] is not None
    assert len(result["trace_steps"]) > 0


def test_crag_agent_fallback(sample_documents):
    """Test CRAG agent fallback routing on unanswerable query outside knowledge base."""
    ingestor = DocumentIngestor()
    chunks = ingestor.process_and_chunk(sample_documents)
    emb_wrapper = EmbeddingWrapper()
    retriever = HybridRetriever(chunks=chunks, embedding_wrapper=emb_wrapper, qdrant_location=":memory:")
    reranker = CrossEncoderReranker()

    class RejectAllLLM(SmartRuleBasedLLM):
        def _generate(self, messages, **kwargs):
            text = "\n".join([m.content for m in messages if isinstance(m.content, str)])
            if "grade" in text.lower():
                from langchain_core.messages import AIMessage
                from langchain_core.outputs import ChatResult, ChatGeneration
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content='{"score": "no", "rationale": "Not found in documents"}'))])
            return super()._generate(messages, **kwargs)

    pipeline = CRAGPipeline(retriever=retriever, reranker=reranker, llm=RejectAllLLM())
    result = pipeline.run("What is the recipe for chocolate chip cookies?")

    assert "cannot provide a reliable answer" in result["generation"].lower() or "insufficient" in result["generation"].lower()
    assert result.get("route") == "fallback"


def test_benchmark_evaluator(tmp_path):
    """Test RAGAS benchmark runner and report generation."""
    evaluator = BenchmarkEvaluator()
    subset = SYNTHETIC_BENCHMARK_DATASET[:3]

    out_file = tmp_path / "eval_test_report.json"
    report = evaluator.evaluate_pipeline(test_dataset=subset, save_report=True, output_path=out_file)

    assert "summary" in report
    assert "detailed_results" in report
    assert len(report["detailed_results"]) == 3
    assert report["summary"]["mean_faithfulness"] > 0.0
    assert report["summary"]["overall_crag_score"] > 0.0
    assert out_file.exists()
    assert out_file.with_suffix(".csv").exists()
