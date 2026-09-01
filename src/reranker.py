"""
Cross-Encoder Reranker module for re-scoring retrieved candidates.
"""

from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Standard container for text chunks and retrieval metadata."""
    id: str
    page_content: str
    metadata: Dict[str, Any]
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    rerank_score: Optional[float] = None
    rerank_rank: Optional[int] = None
    relevance_grade: Optional[str] = None  # 'relevant' | 'irrelevant'
    grade_rationale: Optional[str] = None


class CrossEncoderReranker:
    """
    Reranker using Cross-Encoder models (e.g., ms-marco-MiniLM-L-6-v2)
    to compute fine-grained semantic similarity between query and candidate chunks.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        self._initialized = False

    def _load_model(self):
        """Lazy load cross-encoder model to speed up startup."""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            self._initialized = True
            logger.info("Cross-Encoder model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load CrossEncoder from sentence-transformers: {e}. Using fallback heuristic scorer.")
            self._model = None
            self._initialized = True

    def rerank(
        self,
        query: str,
        documents: List[DocumentChunk],
        top_k: int = 3
    ) -> List[DocumentChunk]:
        """
        Rerank a list of DocumentChunks against the given query.

        Args:
            query: The user search query.
            documents: Candidate DocumentChunks.
            top_k: Number of highest-ranked chunks to return.

        Returns:
            List of reranked and filtered DocumentChunks sorted by rerank_score descending.
        """
        if not documents:
            return []

        self._load_model()

        if self._model is not None:
            try:
                pairs = [(query, doc.page_content) for doc in documents]
                scores = self._model.predict(pairs)
                
                # If scores are raw logits, apply min-max or sigmoid scaling for friendly UI display
                import numpy as np
                scores = np.array(scores, dtype=float)
                
                # Apply sigmoid if logits
                sigmoid_scores = 1.0 / (1.0 + np.exp(-scores))
                
                for idx, doc in enumerate(documents):
                    doc.rerank_score = float(sigmoid_scores[idx])
            except Exception as e:
                logger.error(f"Error during cross-encoder prediction: {e}. Falling back to hybrid score.")
                for doc in documents:
                    doc.rerank_score = doc.hybrid_score or 0.5
        else:
            # Fallback heuristic: combine term-overlap + hybrid score
            query_terms = set(query.lower().split())
            for doc in documents:
                content_lower = doc.page_content.lower()
                overlap = sum(1 for t in query_terms if t in content_lower) / max(len(query_terms), 1)
                base = doc.hybrid_score if doc.hybrid_score is not None else 0.5
                doc.rerank_score = round(0.5 * base + 0.5 * overlap, 4)

        # Sort descending by rerank_score
        reranked = sorted(documents, key=lambda d: d.rerank_score if d.rerank_score is not None else -1.0, reverse=True)
        
        # Assign ranks
        for rank, doc in enumerate(reranked, start=1):
            doc.rerank_rank = rank

        return reranked[:top_k]
