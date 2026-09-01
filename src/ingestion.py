"""
Document ingestion, layout-aware Markdown parsing, chunking, and hybrid indexing (Qdrant + BM25).
"""

import os
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from src.config import settings
from src.reranker import DocumentChunk

logger = logging.getLogger(__name__)


class EmbeddingWrapper:
    """
    Unified embedding model interface supporting FastEmbed, SentenceTransformers,
    and a robust fallback embedding generator for offline/testing scenarios.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._fastembed = None
        self._st_model = None
        self._dim = 384
        self._initialized = False

    def _init_model(self):
        if self._initialized:
            return
        
        # 1. Try FastEmbed
        try:
            from fastembed import TextEmbedding
            logger.info(f"Initializing FastEmbed model: {self.model_name}")
            self._fastembed = TextEmbedding(model_name=self.model_name)
            self._initialized = True
            logger.info("FastEmbed initialized successfully.")
            return
        except Exception as e:
            logger.debug(f"FastEmbed init failed: {e}. Trying sentence-transformers.")

        # 2. Try SentenceTransformers
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Initializing SentenceTransformer: {self.model_name}")
            self._st_model = SentenceTransformer(self.model_name)
            self._dim = self._st_model.get_sentence_embedding_dimension()
            self._initialized = True
            logger.info("SentenceTransformer initialized successfully.")
            return
        except Exception as e:
            logger.warning(f"SentenceTransformer init failed: {e}. Using deterministic local embedding fallback.")
            self._initialized = True

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self._init_model()
        if not texts:
            return []

        if self._fastembed is not None:
            try:
                embeddings = list(self._fastembed.embed(texts))
                return [e.tolist() for e in embeddings]
            except Exception as e:
                logger.error(f"FastEmbed batch embed error: {e}")

        if self._st_model is not None:
            try:
                embeddings = self._st_model.encode(texts, normalize_embeddings=True)
                return [e.tolist() for e in embeddings]
            except Exception as e:
                logger.error(f"SentenceTransformer embed error: {e}")

        # Deterministic lightweight hashing embedding fallback
        return [self._hash_embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        self._init_model()
        if self._fastembed is not None:
            try:
                embeddings = list(self._fastembed.embed([text]))
                return embeddings[0].tolist()
            except Exception as e:
                logger.error(f"FastEmbed query embed error: {e}")

        if self._st_model is not None:
            try:
                embedding = self._st_model.encode(text, normalize_embeddings=True)
                return embedding.tolist()
            except Exception as e:
                logger.error(f"SentenceTransformer query embed error: {e}")

        return self._hash_embed(text)

    def _hash_embed(self, text: str) -> List[float]:
        """Deterministic 384-dimensional normalized bag-of-characters vector for offline fallback."""
        vec = np.zeros(self._dim, dtype=np.float32)
        words = text.lower().split()
        for w in words:
            h = hash(w) % self._dim
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


class HybridRetriever:
    """
    Ensemble hybrid search retriever combining Dense Vector (Qdrant) and Sparse Keyword (BM25).
    """

    def __init__(
        self,
        chunks: List[Document],
        embedding_wrapper: EmbeddingWrapper,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        collection_name: str = "crag_knowledge_base",
        qdrant_location: str = ":memory:"
    ):
        self.chunks = chunks
        self.embedding_wrapper = embedding_wrapper
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.collection_name = collection_name
        self.qdrant_location = qdrant_location
        self.qdrant_client = None
        
        # Build tokenized corpus for BM25
        self.corpus_tokens = [self._tokenize(doc.page_content) for doc in self.chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None
        
        # Initialize and populate Qdrant
        self._setup_qdrant()

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower().strip(".,!?;:\"'()[]{}") for w in text.split() if w.strip()]

    def _setup_qdrant(self):
        if not self.chunks:
            return
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as rest_models
            
            if self.qdrant_location == ":memory:":
                self.qdrant_client = QdrantClient(location=":memory:")
            else:
                os.makedirs(self.qdrant_location, exist_ok=True)
                self.qdrant_client = QdrantClient(path=self.qdrant_location)

            dim = self.embedding_wrapper.dimension
            if self.qdrant_client.collection_exists(self.collection_name):
                self.qdrant_client.delete_collection(self.collection_name)

            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest_models.VectorParams(
                    size=dim,
                    distance=rest_models.Distance.COSINE
                )
            )

            # Generate embeddings and upload points
            texts = [c.page_content for c in self.chunks]
            embeddings = self.embedding_wrapper.embed_documents(texts)
            
            points = []
            for idx, (chunk, emb) in enumerate(zip(self.chunks, embeddings)):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk.metadata.get('source', '')}_{idx}"))
                payload = {
                    "index": idx,
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata
                }
                points.append(
                    rest_models.PointStruct(
                        id=point_id,
                        vector=emb,
                        payload=payload
                    )
                )

            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Indexed {len(points)} chunks into Qdrant collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Error setting up Qdrant vector store: {e}")
            self.qdrant_client = None

    def retrieve(self, query: str, top_k: int = 10) -> List[DocumentChunk]:
        """
        Perform hybrid retrieval combining Qdrant dense search and BM25 sparse search.
        """
        if not self.chunks:
            return []

        num_chunks = len(self.chunks)
        top_k = min(top_k, num_chunks)

        # 1. Sparse Scores (BM25)
        query_tokens = self._tokenize(query)
        bm25_raw_scores = self.bm25.get_scores(query_tokens) if self.bm25 else np.zeros(num_chunks)
        
        # Normalize BM25 scores to [0, 1]
        max_bm25 = np.max(bm25_raw_scores) if len(bm25_raw_scores) > 0 and np.max(bm25_raw_scores) > 0 else 1.0
        bm25_norm_scores = (bm25_raw_scores / max_bm25) if max_bm25 > 0 else np.zeros(num_chunks)

        # 2. Dense Scores (Qdrant)
        dense_scores = np.zeros(num_chunks)
        if self.qdrant_client is not None:
            try:
                query_emb = self.embedding_wrapper.embed_query(query)
                if hasattr(self.qdrant_client, "query_points"):
                    query_res = self.qdrant_client.query_points(
                        collection_name=self.collection_name,
                        query=query_emb,
                        limit=num_chunks
                    )
                    hits = query_res.points
                else:
                    hits = self.qdrant_client.search(
                        collection_name=self.collection_name,
                        query_vector=query_emb,
                        limit=num_chunks
                    )
                for hit in hits:
                    idx = hit.payload.get("index")
                    if idx is not None and 0 <= idx < num_chunks:
                        # Cosine similarity is usually in [-1, 1], normalize to [0, 1]
                        dense_scores[idx] = max(0.0, (hit.score + 1.0) / 2.0)
            except Exception as e:
                logger.error(f"Qdrant search error: {e}")
        else:
            # Fallback: compute cosine similarity via embeddings directly
            query_emb = np.array(self.embedding_wrapper.embed_query(query))
            chunk_embs = np.array(self.embedding_wrapper.embed_documents([c.page_content for c in self.chunks]))
            if len(chunk_embs) > 0 and np.linalg.norm(query_emb) > 0:
                norms = np.linalg.norm(chunk_embs, axis=1) * np.linalg.norm(query_emb)
                sims = np.dot(chunk_embs, query_emb) / np.maximum(norms, 1e-9)
                dense_scores = np.maximum(0.0, (sims + 1.0) / 2.0)

        # 3. Hybrid Ensemble Fusion
        hybrid_candidates: List[DocumentChunk] = []
        for idx, chunk in enumerate(self.chunks):
            d_score = float(dense_scores[idx])
            s_score = float(bm25_norm_scores[idx])
            h_score = (self.dense_weight * d_score) + (self.sparse_weight * s_score)
            
            chunk_obj = DocumentChunk(
                id=str(idx),
                page_content=chunk.page_content,
                metadata=chunk.metadata,
                dense_score=round(d_score, 4),
                sparse_score=round(s_score, 4),
                hybrid_score=round(h_score, 4)
            )
            hybrid_candidates.append(chunk_obj)

        # Sort by hybrid score descending and return top_k
        sorted_candidates = sorted(hybrid_candidates, key=lambda x: x.hybrid_score or 0.0, reverse=True)
        return sorted_candidates[:top_k]


class DocumentIngestor:
    """
    Parses complex documents (PDFs via PyMuPDF4LLM, Markdown, Text) into structured chunks.
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
        )

    def load_file(self, file_path: Path) -> List[Document]:
        """
        Load a file using layout-aware Markdown extraction for PDF, or native Markdown/Text loader.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        docs: List[Document] = []

        if suffix == ".pdf":
            try:
                import pymupdf4llm
                # Extract markdown text preserving tables and headers
                md_text = pymupdf4llm.to_markdown(str(file_path))
                docs.append(Document(
                    page_content=md_text,
                    metadata={"source": file_path.name, "type": "pdf_markdown"}
                ))
            except Exception as e:
                logger.warning(f"PyMuPDF4LLM failed for {file_path.name}: {e}. Falling back to pypdf.")
                import pypdf
                reader = pypdf.PdfReader(str(file_path))
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        docs.append(Document(
                            page_content=text,
                            metadata={"source": file_path.name, "page": page_num + 1, "type": "pdf"}
                        ))
        elif suffix in [".md", ".markdown"]:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            docs.append(Document(
                page_content=text,
                metadata={"source": file_path.name, "type": "markdown"}
            ))
        elif suffix in [".txt", ".log", ".csv"]:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            docs.append(Document(
                page_content=text,
                metadata={"source": file_path.name, "type": "text"}
            ))
        else:
            # General text fallback
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            docs.append(Document(
                page_content=text,
                metadata={"source": file_path.name, "type": "unknown"}
            ))

        return docs

    def process_and_chunk(self, file_paths: List[Path]) -> List[Document]:
        """
        Ingest a list of files and split them into semantic chunks.
        """
        all_raw_docs: List[Document] = []
        for p in file_paths:
            docs = self.load_file(p)
            all_raw_docs.extend(docs)

        chunks = self.text_splitter.split_documents(all_raw_docs)
        # Clean chunks
        cleaned_chunks = []
        for idx, chunk in enumerate(chunks):
            content = chunk.page_content.strip()
            if len(content) > 20:  # filter noise/blank chunks
                chunk.metadata["chunk_id"] = idx
                cleaned_chunks.append(chunk)

        logger.info(f"Ingested {len(file_paths)} files into {len(cleaned_chunks)} chunks.")
        return cleaned_chunks
