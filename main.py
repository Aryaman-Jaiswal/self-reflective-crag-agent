"""
Main entry point for Self-Reflective Corrective RAG (CRAG) Agent.
Supports CLI execution and FastAPI REST API serving.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel

from src.config import settings
from src.ingestion import DocumentIngestor, EmbeddingWrapper, HybridRetriever
from src.reranker import CrossEncoderReranker
from src.agent_graph import CRAGPipeline
from src.evaluate import BenchmarkEvaluator
from src.llm_factory import get_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CRAG-Main")


# Pydantic Schemas for FastAPI
class QueryRequest(BaseModel):
    query: str
    provider: Optional[str] = None
    model_name: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    rewritten_query: Optional[str]
    generation: str
    retry_count: int
    hallucination_grade: str
    trace_steps: List[dict]


def initialize_agent(data_folder: Optional[Path] = None) -> CRAGPipeline:
    """Initialize ingestor, embeddings, Qdrant hybrid retriever, and LangGraph CRAG agent."""
    ingestor = DocumentIngestor()
    emb_wrapper = EmbeddingWrapper(model_name=settings.embedding_model_name)
    
    docs_dir = data_folder or settings.sample_docs_dir
    files = list(docs_dir.glob("*.*"))
    chunks = ingestor.process_and_chunk(files) if files else []
    
    retriever = HybridRetriever(
        chunks=chunks,
        embedding_wrapper=emb_wrapper,
        dense_weight=settings.dense_weight,
        sparse_weight=settings.sparse_weight,
        collection_name=settings.qdrant_collection_name,
        qdrant_location=settings.qdrant_location
    )
    
    reranker = CrossEncoderReranker(model_name=settings.reranker_model_name)
    llm = get_llm()
    return CRAGPipeline(retriever=retriever, reranker=reranker, llm=llm)


# ------------------ FASTAPI APP CREATION ------------------
def create_app():
    """Build and configure FastAPI instance."""
    from fastapi import FastAPI, HTTPException, UploadFile, File
    
    app = FastAPI(
        title="Self-Reflective Corrective RAG (CRAG) API",
        description="Production API for CRAG Agent powered by LangGraph, Qdrant, and Cross-Encoder reranking.",
        version="1.0.0"
    )

    agent_instance = initialize_agent()

    @app.get("/health")
    def health_check():
        return {"status": "ok", "service": "CRAG Agent"}

    @app.post("/query", response_model=QueryResponse)
    def ask_query(req: QueryRequest):
        try:
            result = agent_instance.run(req.query)
            return QueryResponse(
                query=result["query"],
                rewritten_query=result.get("rewritten_query"),
                generation=result.get("generation", ""),
                retry_count=result.get("retry_count", 0),
                hallucination_grade=result.get("hallucination_grade", "grounded"),
                trace_steps=result.get("trace_steps", [])
            )
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/evaluate")
    def trigger_evaluation():
        evaluator = BenchmarkEvaluator(agent=agent_instance)
        report = evaluator.evaluate_pipeline(save_report=True)
        return report

    return app


# ------------------ CLI INTERFACE ------------------
def cli_main():
    parser = argparse.ArgumentParser(description="Self-Reflective Corrective RAG (CRAG) CLI")
    parser.add_argument("--query", "-q", type=str, help="Question to ask the CRAG Agent")
    parser.add_argument("--evaluate", "-e", action="store_true", help="Run RAGAS benchmark evaluation")
    parser.add_argument("--serve", "-s", action="store_true", help="Run FastAPI REST API server")
    parser.add_argument("--port", "-p", type=int, default=8000, help="API server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="API server host")

    args = parser.parse_args()

    if args.serve:
        import uvicorn
        app = create_app()
        uvicorn.run(app, host=args.host, port=args.port)
        return

    if args.evaluate:
        logger.info("Initializing Agent and running evaluation suite...")
        agent = initialize_agent()
        evaluator = BenchmarkEvaluator(agent=agent)
        report = evaluator.evaluate_pipeline(save_report=True)
        print("\n" + "="*50)
        print("CRAG BENCHMARK EVALUATION SUMMARY")
        print("="*50)
        for k, v in report["summary"].items():
            print(f"  {k}: {v}")
        print("="*50)
        return

    if args.query:
        logger.info(f"Querying CRAG Agent: '{args.query}'")
        agent = initialize_agent()
        result = agent.run(args.query)
        print("\n" + "="*50)
        print("FINAL SYNTHESIZED ANSWER")
        print("="*50)
        print(result.get("generation"))
        print("\n" + "="*50)
        print(f"Retries: {result.get('retry_count')} | Hallucination Grade: {result.get('hallucination_grade')}")
        print("="*50)
        return

    # Default: Show help and run interactive loop
    print("Welcome to CRAG Agent CLI. Type 'exit' or 'quit' to stop.\n")
    agent = initialize_agent()
    while True:
        try:
            q = input("\nEnter Question: ").strip()
            if q.lower() in ("exit", "quit", "q"):
                break
            if not q:
                continue
            res = agent.run(q)
            print("\n--- Answer ---")
            print(res.get("generation"))
            print(f"\n[Trace: {len(res.get('trace_steps', []))} steps | Retries: {res.get('retry_count', 0)}]")
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    cli_main()
