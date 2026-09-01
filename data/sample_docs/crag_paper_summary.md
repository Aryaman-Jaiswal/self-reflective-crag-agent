# Corrective Retrieval Augmented Generation (CRAG)

## 1. Overview and Motivation
Standard Retrieval-Augmented Generation (RAG) models assume that retrieved documents are consistently accurate and relevant. In practice, retrieval systems frequently fetch noisy, irrelevant, or misleading passages, leading generative language models to produce ungrounded hallucinations. Corrective RAG (CRAG) introduces self-reflection and dynamic retrieval evaluation to alleviate this limitation.

## 2. Core Architecture & Pipeline Components

### A. Retrieval Evaluator (Confidence Estimator)
CRAG employs a lightweight retrieval evaluator designed to evaluate the overall relevance quality of retrieved documents for a given input query. The evaluator assigns confidence scores to categorize retrieval into three discrete outcomes:
1. **Correct**: The retrieved documents are relevant and sufficient. The system proceeds directly to document refinement and answer generation.
2. **Incorrect**: The retrieved documents are irrelevant or erroneous. The system discards the irrelevant documents and triggers external query rewriting or search fallback.
3. **Ambiguous**: The retrieval confidence is intermediate. The system integrates internal knowledge with external search refinement.

### B. Decompose-Then-Recompose Document Refinement
To eliminate internal noise within retrieved passages, CRAG breaks documents down into fine-grained knowledge strips (sentences or sub-chunks), filters out irrelevant strips, and recombines the key factual sentences into a compact, vetted prompt.

### C. Query Rewriting and Search Correction
When retrieval quality is graded as low or ambiguous, the agent activates query transformation. The query is re-engineered using semantic expansion, entity disambiguation, and synonym enrichment to maximize recall across the dense and sparse indexes.

### D. Self-Reflective Hallucination Guardrails
Before returning a final answer to the user, an internal faithfulness audit verifies whether the claims in the generated response are strictly grounded in the vetted context strips. If any claim is ungrounded, the generation node is re-prompted with strict grounding constraints.
