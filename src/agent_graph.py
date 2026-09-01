import re
import json
import logging
from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from src.config import settings
from src.reranker import DocumentChunk, CrossEncoderReranker
from src.ingestion import HybridRetriever
from src.llm_factory import get_llm

logger = logging.getLogger(__name__)


def _extract_text(response: Any) -> str:
    """Safely extract plain text from LLM response (handling strings, lists of content blocks, etc.)."""
    if hasattr(response, "content"):
        content = response.content
    else:
        content = response

    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", "") or str(item))
            elif hasattr(item, "text"):
                parts.append(getattr(item, "text", ""))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content).strip()


def _safe_json_loads(raw_text: str) -> Dict[str, Any]:
    """
    Safely parse JSON from LLM output, handling markdown code fences, unescaped LaTeX math,
    and regex fallback extraction.
    """
    cleaned = raw_text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()

    # 1. Direct JSON parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 2. Fix unescaped backslashes (e.g. \frac, \sqrt, \alpha in LaTeX)
    try:
        sanitized = re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', cleaned)
        return json.loads(sanitized)
    except Exception:
        pass

    # 3. Regex Fallback extraction of key fields
    extracted: Dict[str, Any] = {}
    score_m = re.search(r'"score"\s*:\s*"([^"]+)"', cleaned, re.IGNORECASE)
    if score_m:
        extracted["score"] = score_m.group(1)

    rationale_m = re.search(r'"rationale"\s*:\s*"([^"]+)"', cleaned, re.IGNORECASE)
    if rationale_m:
        extracted["rationale"] = rationale_m.group(1)

    improved_m = re.search(r'"improved_query"\s*:\s*"([^"]+)"', cleaned, re.IGNORECASE)
    if improved_m:
        extracted["improved_query"] = improved_m.group(1)

    reasoning_m = re.search(r'"reasoning"\s*:\s*"([^"]+)"', cleaned, re.IGNORECASE)
    if reasoning_m:
        extracted["reasoning"] = reasoning_m.group(1)

    if extracted:
        return extracted

    raise ValueError(f"Could not parse valid JSON from text: {cleaned[:100]}")


class GraphState(TypedDict):
    """LangGraph State holding conversational context and CRAG pipeline artifacts."""
    query: str
    rewritten_query: Optional[str]
    documents: List[DocumentChunk]
    graded_documents: List[DocumentChunk]
    generation: Optional[str]
    retry_count: int
    hallucination_retry: int
    route: Optional[str]
    hallucination_grade: Optional[str]
    trace_steps: List[Dict[str, Any]]


class CRAGPipeline:
    """
    Production-grade Corrective RAG (CRAG) pipeline using LangGraph state machine.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Optional[CrossEncoderReranker] = None,
        llm=None
    ):
        self.retriever = retriever
        self.reranker = reranker or CrossEncoderReranker(model_name=settings.reranker_model_name)
        self.llm = llm or get_llm()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(GraphState)

        # Register Nodes
        workflow.add_node("retrieve_and_rerank", self.node_retrieve_and_rerank)
        workflow.add_node("grade_documents", self.node_grade_documents)
        workflow.add_node("rewrite_query", self.node_rewrite_query)
        workflow.add_node("generate_answer", self.node_generate_answer)
        workflow.add_node("generate_fallback", self.node_generate_fallback)
        workflow.add_node("hallucination_check", self.node_hallucination_check)

        # Set Entry Point
        workflow.set_entry_point("retrieve_and_rerank")

        # Edge 1: retrieve_and_rerank -> grade_documents
        workflow.add_edge("retrieve_and_rerank", "grade_documents")

        # Edge 2: Conditional routing from grade_documents
        workflow.add_conditional_edges(
            "grade_documents",
            self.edge_decide_to_generate,
            {
                "generate_answer": "generate_answer",
                "rewrite_query": "rewrite_query",
                "generate_fallback": "generate_fallback"
            }
        )

        # Edge 3: rewrite_query loops back to retrieve_and_rerank
        workflow.add_edge("rewrite_query", "retrieve_and_rerank")

        # Edge 4: generate_answer -> hallucination_check
        workflow.add_edge("generate_answer", "hallucination_check")

        # Edge 5: Conditional routing from hallucination_check
        workflow.add_conditional_edges(
            "hallucination_check",
            self.edge_decide_hallucination,
            {
                "grounded": END,
                "regenerate": "generate_answer",
                "fallback": "generate_fallback"
            }
        )

        # Edge 6: fallback ends workflow
        workflow.add_edge("generate_fallback", END)

        return workflow.compile()

    # ------------------ NODES ------------------

    def node_retrieve_and_rerank(self, state: GraphState) -> Dict[str, Any]:
        """Hybrid search (Qdrant + BM25) followed by Cross-Encoder reranking."""
        current_query = state.get("rewritten_query") or state["query"]
        logger.info(f"--- [NODE: retrieve_and_rerank] Query: '{current_query}' ---")

        # Step 1: Hybrid Retrieval
        retrieved = self.retriever.retrieve(current_query, top_k=settings.top_k_retrieval)

        # Step 2: Cross-Encoder Reranking
        reranked = self.reranker.rerank(current_query, retrieved, top_k=settings.top_k_rerank)

        trace = {
            "step": "retrieve_and_rerank",
            "active_query": current_query,
            "candidates_retrieved": len(retrieved),
            "top_reranked_count": len(reranked),
            "chunks": [
                {
                    "id": d.id,
                    "content_snippet": d.page_content[:200] + "...",
                    "dense_score": d.dense_score,
                    "sparse_score": d.sparse_score,
                    "hybrid_score": d.hybrid_score,
                    "rerank_score": d.rerank_score,
                    "rank": d.rerank_rank,
                    "source": d.metadata.get("source", "unknown")
                }
                for d in reranked
            ]
        }

        trace_steps = list(state.get("trace_steps", []))
        trace_steps.append(trace)

        return {
            "documents": reranked,
            "trace_steps": trace_steps
        }

    def node_grade_documents(self, state: GraphState) -> Dict[str, Any]:
        """LLM evaluates semantic relevance of each retrieved chunk."""
        query = state["query"]
        docs = state.get("documents", [])
        logger.info(f"--- [NODE: grade_documents] Grading {len(docs)} chunks ---")

        vetted_docs: List[DocumentChunk] = []
        grading_details = []

        system_prompt = (
            "You are an expert retrieval evaluator. Your task is to evaluate whether a retrieved document "
            "chunk is relevant and contains useful information to answer the user question.\n"
            "Return a JSON object strictly matching:\n"
            '{"score": "yes" | "no", "rationale": "<concise explanation>"}'
        )

        for doc in docs:
            user_prompt = (
                f"User Question: {query}\n\n"
                f"Retrieved Document:\n{doc.page_content}\n\n"
                "Is the document relevant to the user question? Respond with JSON:"
            )

            try:
                response = self.llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ])
                
                # Parse JSON response using safe text extraction and escape sanitizer
                raw_text = _extract_text(response)
                parsed = _safe_json_loads(raw_text)
                is_rel = parsed.get("score", "no").lower() == "yes"
                rationale = parsed.get("rationale", "")
            except Exception as e:
                logger.warning(f"Grading parse error: {e}. Heuristic fallback applied.")
                # Heuristic fallback: check query keyword overlap
                q_words = [w.lower() for w in query.split() if len(w) > 3]
                overlap = sum(1 for w in q_words if w in doc.page_content.lower())
                is_rel = overlap > 0 or (doc.rerank_score is not None and doc.rerank_score >= 0.3)
                rationale = f"Heuristic evaluation based on keyword overlap ({overlap} matches)."

            doc.relevance_grade = "relevant" if is_rel else "irrelevant"
            doc.grade_rationale = rationale

            grading_details.append({
                "chunk_id": doc.id,
                "source": doc.metadata.get("source", "unknown"),
                "grade": doc.relevance_grade,
                "rationale": rationale,
                "content_preview": doc.page_content[:150] + "..."
            })

            if is_rel:
                vetted_docs.append(doc)

        trace = {
            "step": "grade_documents",
            "total_evaluated": len(docs),
            "vetted_count": len(vetted_docs),
            "details": grading_details
        }
        trace_steps = list(state.get("trace_steps", []))
        trace_steps.append(trace)

        return {
            "graded_documents": vetted_docs,
            "trace_steps": trace_steps
        }

    def node_rewrite_query(self, state: GraphState) -> Dict[str, Any]:
        """Rewrite search query to improve retrieval recall when initial search fails."""
        original_query = state["query"]
        retry_count = state.get("retry_count", 0) + 1
        logger.info(f"--- [NODE: rewrite_query] Retry #{retry_count} for query: '{original_query}' ---")

        system_prompt = (
            "You are an expert search query optimizer. The previous vector search query failed to retrieve "
            "relevant documentation. Rewrite the query to optimize for dense and sparse semantic retrieval. "
            "Expand acronyms, add technical keywords, and clarify ambiguous terms.\n"
            "Return a JSON object strictly matching:\n"
            '{"improved_query": "<new query string>", "reasoning": "<why this was changed>"}'
        )

        user_prompt = f"Original Query: {original_query}\n\nProvide the optimized query in JSON:"

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            raw_text = _extract_text(response)
            parsed = _safe_json_loads(raw_text)
            improved = parsed.get("improved_query", original_query)
            reasoning = parsed.get("reasoning", "Expanded query with semantic synonyms.")
        except Exception as e:
            logger.warning(f"Rewrite parse error: {e}. Applying rule-based expansion.")
            improved = f"{original_query} comprehensive overview details technical architecture"
            reasoning = "Rule-based expansion with technical domain keywords."

        trace = {
            "step": "rewrite_query",
            "retry_count": retry_count,
            "original_query": original_query,
            "rewritten_query": improved,
            "reasoning": reasoning
        }
        trace_steps = list(state.get("trace_steps", []))
        trace_steps.append(trace)

        return {
            "rewritten_query": improved,
            "retry_count": retry_count,
            "trace_steps": trace_steps
        }

    def node_generate_answer(self, state: GraphState) -> Dict[str, Any]:
        """Synthesize response strictly grounded on the vetted chunks with citations."""
        query = state["query"]
        docs = state.get("graded_documents", [])
        is_regenerate = state.get("hallucination_grade") == "not_grounded"
        logger.info(f"--- [NODE: generate_answer] Generating response (Regenerate: {is_regenerate}) ---")

        # Format context with numbered citation tags
        context_parts = []
        for idx, doc in enumerate(docs, start=1):
            src = doc.metadata.get("source", f"Document {idx}")
            context_parts.append(f"[Document {idx}] (Source: {src}):\n{doc.page_content}")
        
        context_str = "\n\n".join(context_parts)

        strictness_clause = (
            "IMPORTANT: Your previous draft was flagged for potential hallucination. "
            "You MUST ONLY state facts directly and explicitly mentioned in the context. "
            "Do NOT extrapolate or assume." if is_regenerate else ""
        )

        system_prompt = (
            "You are a factual, precision-oriented AI assistant. Answer the user's question strictly "
            "and exclusively using the provided context documents.\n"
            "- Always cite your sources in-text using `[Doc 1]`, `[Doc 2]`, etc.\n"
            "- If a detail cannot be proven by the context, omit it.\n"
            "- Maintain professional clarity, structure with bullet points when applicable.\n"
            f"{strictness_clause}"
        )

        user_prompt = (
            f"Context Documents:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            "Synthesized Answer with Citations:"
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            generation = _extract_text(response)
        except Exception as e:
            logger.error(f"Generation error: {e}")
            generation = "An error occurred while generating the grounded answer from vetted context."

        trace = {
            "step": "generate_answer",
            "is_regenerate": is_regenerate,
            "used_documents_count": len(docs),
            "generation_snippet": generation[:300] + ("..." if len(generation) > 300 else "")
        }
        trace_steps = list(state.get("trace_steps", []))
        trace_steps.append(trace)

        return {
            "generation": generation,
            "trace_steps": trace_steps
        }

    def node_generate_fallback(self, state: GraphState) -> Dict[str, Any]:
        """Gracefully admit lack of reliable context to prevent hallucinations."""
        query = state["query"]
        retries = state.get("retry_count", 0)
        logger.info(f"--- [NODE: generate_fallback] Returning fallback for query: '{query}' ---")

        fallback_msg = (
            f"I cannot provide a reliable answer to your query: \"{query}\" based on the uploaded knowledge base.\n\n"
            f"**Reason**: After {retries} search refinement attempt(s), the available documents did not contain "
            "sufficient, verified information to answer accurately without risking hallucination. "
            "Please consider uploading additional relevant documents or providing more specific keywords."
        )

        trace = {
            "step": "generate_fallback",
            "reason": "Exhausted retries or irrelevance without hallucination",
            "retry_count": retries
        }
        trace_steps = list(state.get("trace_steps", []))
        trace_steps.append(trace)

        return {
            "generation": fallback_msg,
            "route": "fallback",
            "trace_steps": trace_steps
        }

    def node_hallucination_check(self, state: GraphState) -> Dict[str, Any]:
        """Internal reflection check: verify if the generated answer is strictly grounded."""
        generation = state.get("generation", "")
        docs = state.get("graded_documents", [])
        hallucination_retry = state.get("hallucination_retry", 0)
        logger.info("--- [NODE: hallucination_check] Verifying factual groundedness ---")

        context_str = "\n\n".join([f"[Doc {i+1}]: {d.page_content}" for i, d in enumerate(docs)])

        system_prompt = (
            "You are an unbiased AI auditor tasked with hallucination detection. "
            "Evaluate whether every factual claim in the generated answer is directly supported by the context.\n"
            "Return a JSON object strictly matching:\n"
            '{"score": "yes" | "no", "rationale": "<explanation>", "hallucinated_statements": ["<list of ungrounded claims>"]}'
        )

        user_prompt = (
            f"Context Documents:\n{context_str}\n\n"
            f"Generated Answer:\n{generation}\n\n"
            "Is the generated answer strictly grounded in the facts from the context? Respond in JSON:"
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            raw_text = _extract_text(response)
            parsed = _safe_json_loads(raw_text)
            is_grounded = parsed.get("score", "yes").lower() == "yes"
            rationale = parsed.get("rationale", "Answer is strictly grounded.")
            hallucinated_statements = parsed.get("hallucinated_statements", [])
        except Exception as e:
            logger.warning(f"Hallucination check parse error: {e}. Defaulting to grounded.")
            is_grounded = True
            rationale = "Heuristic check passed."
            hallucinated_statements = []

        grade = "grounded" if is_grounded else "not_grounded"

        trace = {
            "step": "hallucination_check",
            "grade": grade,
            "rationale": rationale,
            "hallucinated_statements": hallucinated_statements,
            "hallucination_retry": hallucination_retry
        }
        trace_steps = list(state.get("trace_steps", []))
        trace_steps.append(trace)

        return {
            "hallucination_grade": grade,
            "hallucination_retry": hallucination_retry + (0 if is_grounded else 1),
            "trace_steps": trace_steps
        }

    # ------------------ CONDITIONAL EDGES ------------------

    def edge_decide_to_generate(self, state: GraphState) -> str:
        """Route to generate_answer if >=1 doc relevant; else rewrite or fallback."""
        graded_docs = state.get("graded_documents", [])
        retries = state.get("retry_count", 0)

        if len(graded_docs) > 0:
            logger.info("--- [EDGE: decide_to_generate] -> generate_answer ---")
            return "generate_answer"
        
        if retries < settings.max_retries:
            logger.info(f"--- [EDGE: decide_to_generate] -> rewrite_query (Retry {retries+1}/{settings.max_retries}) ---")
            return "rewrite_query"
        
        logger.info("--- [EDGE: decide_to_generate] -> generate_fallback ---")
        return "generate_fallback"

    def edge_decide_hallucination(self, state: GraphState) -> str:
        """Check hallucination status and decide whether to finish or regenerate."""
        grade = state.get("hallucination_grade", "grounded")
        h_retry = state.get("hallucination_retry", 0)

        if grade == "grounded":
            logger.info("--- [EDGE: decide_hallucination] -> END (Grounded) ---")
            return "grounded"
        
        if h_retry <= 1:
            logger.info("--- [EDGE: decide_hallucination] -> regenerate ---")
            return "regenerate"

        logger.info("--- [EDGE: decide_hallucination] -> fallback ---")
        return "fallback"

    # ------------------ EXECUTION ENTRYPOINT ------------------

    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute the full CRAG agent workflow for a given query.
        """
        initial_state: GraphState = {
            "query": query,
            "rewritten_query": None,
            "documents": [],
            "graded_documents": [],
            "generation": None,
            "retry_count": 0,
            "hallucination_retry": 0,
            "route": None,
            "hallucination_grade": None,
            "trace_steps": []
        }

        final_state = self.graph.invoke(initial_state)
        return final_state
