"""
Configuration management for the CRAG pipeline.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
QDRANT_STORAGE_DIR = DATA_DIR / "qdrant_db"


def _get_env_or_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieve config from environment variable or Streamlit secrets."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


class Settings(BaseModel):
    """Global configuration settings for CRAG agent and components."""

    # Project directories
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    sample_docs_dir: Path = SAMPLE_DOCS_DIR
    qdrant_storage_dir: Path = QDRANT_STORAGE_DIR

    # Text Splitting & Parsing
    chunk_size: int = Field(default=600, description="Recursive character text splitter chunk size")
    chunk_overlap: int = Field(default=80, description="Chunk overlap")

    # Embeddings & Vector Store
    embedding_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Dense embedding model identifier"
    )
    qdrant_collection_name: str = Field(
        default="crag_knowledge_base",
        description="Collection name for Qdrant vector store"
    )
    qdrant_location: str = Field(
        default=":memory:",
        description="':memory:' for in-memory or a path string for persistent local storage"
    )

    # Hybrid Search Weights
    dense_weight: float = Field(default=0.6, description="Weight for dense retrieval in ensemble")
    sparse_weight: float = Field(default=0.4, description="Weight for BM25 sparse retrieval in ensemble")
    top_k_retrieval: int = Field(default=10, description="Top-K documents fetched by hybrid retrieval")

    # Reranking
    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-Encoder model for reranking"
    )
    top_k_rerank: int = Field(default=3, description="Final number of top chunks kept after reranking")

    # LangGraph Agent Settings
    max_retries: int = Field(default=2, description="Maximum query rewrite retries before fallback")
    llm_provider: str = Field(
        default_factory=lambda: _get_env_or_secret("LLM_PROVIDER", "openai"),
        description="LLM provider: 'openai', 'gemini', 'anthropic', 'groq', or 'local'"
    )
    llm_model_name: str = Field(
        default_factory=lambda: _get_env_or_secret("LLM_MODEL", "gpt-4o-mini"),
        description="LLM model identifier"
    )
    llm_temperature: float = Field(default=0.0, description="LLM temperature")
    
    # API Keys
    openai_api_key: Optional[str] = Field(default_factory=lambda: _get_env_or_secret("OPENAI_API_KEY"))
    google_api_key: Optional[str] = Field(default_factory=lambda: _get_env_or_secret("GOOGLE_API_KEY") or _get_env_or_secret("GEMINI_API_KEY"))
    anthropic_api_key: Optional[str] = Field(default_factory=lambda: _get_env_or_secret("ANTHROPIC_API_KEY"))
    groq_api_key: Optional[str] = Field(default_factory=lambda: _get_env_or_secret("GROQ_API_KEY"))

    model_config = {"arbitrary_types_allowed": True}


# Global settings instance
settings = Settings()
