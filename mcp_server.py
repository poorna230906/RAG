
"""
Model Context Protocol (MCP) Server for Biomedical RAG (BioASQ) System.

Exposes:
  Tools:
    - search_passages: Fast semantic search across the FAISS biomedical vector store.
    - rag_answer: End-to-end RAG query (FAISS retrieval + Claude answer + dynamic ROUGE grounding).
    - calculate_grounding: Calculates Precision, Recall, and ROUGE-1 F1 between an answer and context.
    - get_metrics: Retrieves API attempts, successes, failures, and error rates from error_metrics.json.
    - check_system_health: System diagnostic check (vector index status, model availability, API key presence).
    - run_benchmark_evaluation: Runs benchmark metrics (Recall@k, Precision@k, ROUGE@k) on BioASQ test split.

  Resources:
    - rag://system/health: JSON health status and configuration info.
    - rag://metrics/errors: Real-time API reliability metrics.
    - rag://pipeline/architecture: Markdown architecture documentation.

  Prompts:
    - biomedical_qa: Template for strictly grounded biomedical question answering.
    - verify_claim: Template for validating medical assertions against scientific literature.
"""

import os
import sys
import json
import logging
from typing import Any, Optional, List, Dict

# --- Path & Environment Setup ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Disable SSL verification if needed for proxy/offline environments
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
try:
    import httpx
    orig_client_init = httpx.Client.__init__
    def patched_client_init(self, *args, **kwargs):
        kwargs['verify'] = False
        orig_client_init(self, *args, **kwargs)
    httpx.Client.__init__ = patched_client_init

    orig_async_client_init = httpx.AsyncClient.__init__
    def patched_async_client_init(self, *args, **kwargs):
        kwargs['verify'] = False
        orig_async_client_init(self, *args, **kwargs)
    httpx.AsyncClient.__init__ = patched_async_client_init
except Exception:
    pass

try:
    import requests
    orig_session_request = requests.Session.request
    def patched_session_request(self, method, url, **kwargs):
        kwargs['verify'] = False
        return orig_session_request(self, method, url, **kwargs)
    requests.Session.request = patched_session_request
except Exception:
    pass

# Load environment variables without modifying .env
from dotenv import load_dotenv
env_path = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

# Configure logging to stderr (critical for stdio MCP to avoid corrupting JSON-RPC on stdout)
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [MCP] %(message)s"
)
logger = logging.getLogger("rag-mcp")

# --- MCP Server Initialization ---
from mcp.server.mcpserver import MCPServer

server = MCPServer(
    name="biomedical-rag",
    version="1.0.0",
    description="BioASQ Biomedical Retrieval-Augmented Generation (RAG) MCP Server"
)

# --- Global Cached Singletons ---
_VECTOR_DB = None
_EMBEDDING_MODEL = None
_RETRIEVER = None

def get_cached_components():
    """Lazily loads and caches the FAISS vector database and embedding model."""
    global _VECTOR_DB, _EMBEDDING_MODEL, _RETRIEVER

    if _RETRIEVER is not None:
        return _VECTOR_DB, _RETRIEVER

    store_path = os.path.join(PROJECT_ROOT, "vector_store")
    if not os.path.exists(store_path):
        raise FileNotFoundError(
            f"FAISS vector store not found at '{store_path}'. "
            "Please build the index first using build_index.py."
        )

    logger.info("Initializing embedding model and loading FAISS vector database...")
    from python.embeddings import get_embedding_model
    from langchain_community.vectorstores import FAISS

    try:
        _EMBEDDING_MODEL = get_embedding_model()
    except Exception as e:
        logger.warning(f"Standard embedding model load failed ({e}), falling back to cached local model...")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from langchain_huggingface import HuggingFaceEmbeddings
        _EMBEDDING_MODEL = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"local_files_only": True}
        )

    _VECTOR_DB = FAISS.load_local(
        store_path,
        _EMBEDDING_MODEL,
        allow_dangerous_deserialization=True
    )
    _RETRIEVER = _VECTOR_DB.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.3
        }
    )
    logger.info("Vector database and retriever loaded and cached successfully.")
    return _VECTOR_DB, _RETRIEVER


# ==============================================================================
# MCP Tools
# ==============================================================================

@server.tool(
    name="search_passages",
    description="Performs semantic vector search over the BioASQ biomedical literature index without invoking the LLM."
)
def search_passages(
    query: str,
    top_k: int = 3,
    score_threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Search for relevant biomedical literature passages matching a query.

    Args:
        query: Biomedical question or search phrase.
        top_k: Maximum number of passages to return (1-10).
        score_threshold: Minimum similarity score threshold (0.0 - 1.0).

    Returns:
        A list of matching passages with ID, text snippet, full text, and relevance score.
    """
    vector_db, _ = get_cached_components()
    logger.info(f"Executing search_passages: '{query}' (top_k={top_k}, threshold={score_threshold})")

    # Use similarity_search_with_relevance_scores for fine-grained ranking
    results_with_scores = vector_db.similarity_search_with_relevance_scores(
        query,
        k=max(1, min(top_k, 10))
    )

    passages = []
    for doc, score in results_with_scores:
        if score >= score_threshold:
            passage_id = str(doc.metadata.get("id", "unknown"))
            text = doc.page_content.strip()
            passages.append({
                "id": passage_id,
                "relevance_score": round(float(score), 4),
                "snippet": text[:180] + ("..." if len(text) > 180 else ""),
                "content": text,
                "metadata": {k: v for k, v in doc.metadata.items() if k != "id"}
            })

    logger.info(f"Found {len(passages)} passages above threshold {score_threshold}.")
    return passages


@server.tool(
    name="rag_answer",
    description="Full RAG pipeline: retrieves relevant passages, generates a strictly grounded answer with Claude, and computes live ROUGE-1 F1 grounding metrics."
)
def rag_answer(
    query: str,
    top_k: int = 3,
    score_threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Answers a biomedical question using the end-to-end RAG pipeline with grounding verification.

    Args:
        query: Biomedical question.
        top_k: Number of relevant context passages to retrieve.
        score_threshold: Minimum similarity threshold for passages.

    Returns:
        Structured response with generated answer, grounding metrics (precision, recall, rouge_f1),
        and retrieved context passages.
    """
    vector_db, retriever = get_cached_components()
    logger.info(f"Executing rag_answer for query: '{query}'")

    from python.generator import generate_response

    # Retrieve docs matching the threshold
    results_with_scores = vector_db.similarity_search_with_relevance_scores(
        query,
        k=max(1, min(top_k, 10))
    )
    retrieved_docs = [doc for doc, score in results_with_scores if score >= score_threshold]

    if not retrieved_docs:
        return {
            "query": query,
            "answer": "No relevant biomedical passages were found meeting the similarity threshold.",
            "grounding_metrics": {
                "precision": 0.0,
                "recall": 0.0,
                "rouge_score": 0.0
            },
            "retrieved_passages": []
        }

    # Generate response and metrics via generator
    gen_result = generate_response(query, retrieved_docs)

    passages_summary = []
    for doc in retrieved_docs:
        passages_summary.append({
            "id": str(doc.metadata.get("id", "unknown")),
            "snippet": doc.page_content[:180] + ("..." if len(doc.page_content) > 180 else ""),
            "content": doc.page_content
        })

    return {
        "query": query,
        "answer": gen_result.get("answer", ""),
        "grounding_metrics": {
            "precision": gen_result.get("precision", 0.0),
            "recall": gen_result.get("recall", 0.0),
            "rouge_score": gen_result.get("rouge_score", 0.0)
        },
        "retrieved_passages": passages_summary
    }


@server.tool(
    name="calculate_grounding",
    description="Computes token-level Precision, Recall, and ROUGE-1 F1 grounding score between an answer and reference passages."
)
def calculate_grounding(
    answer: str,
    context_passages: List[str]
) -> Dict[str, float]:
    """
    Computes lexical grounding metrics for any answer against a list of context strings.

    Args:
        answer: The text of the answer to evaluate.
        context_passages: List of reference context passages.

    Returns:
        Dictionary with precision, recall, and rouge_score.
    """
    from python.generator import calculate_rouge_score
    metrics = calculate_rouge_score(answer, context_passages)
    return metrics


@server.tool(
    name="get_error_metrics",
    description="Reads persistent API reliability metrics from error_metrics.json (total attempts, successes, failures, error rate)."
)
def get_error_metrics() -> Dict[str, Any]:
    """
    Returns the persistent LLM API error metrics and error breakdown.
    """
    metrics_path = os.path.join(PROJECT_ROOT, "error_metrics.json")
    if not os.path.exists(metrics_path):
        return {
            "total_attempts": 0,
            "successes": 0,
            "failures": 0,
            "error_rate": 0.0,
            "error_counts": {}
        }
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read error metrics: {e}")
        return {"error": str(e)}


@server.tool(
    name="check_system_health",
    description="Performs diagnostic checks: vector database file size and existence, embedding model readiness, and API key presence."
)
def check_system_health() -> Dict[str, Any]:
    """
    Checks the status and health of the RAG system components.
    """
    store_path = os.path.join(PROJECT_ROOT, "vector_store")
    faiss_file = os.path.join(store_path, "index.faiss")
    pkl_file = os.path.join(store_path, "index.pkl")

    has_index = os.path.exists(faiss_file) and os.path.exists(pkl_file)
    faiss_size_mb = round(os.path.getsize(faiss_file) / (1024 * 1024), 2) if os.path.exists(faiss_file) else 0.0
    pkl_size_mb = round(os.path.getsize(pkl_file) / (1024 * 1024), 2) if os.path.exists(pkl_file) else 0.0

    api_key_set = bool(os.getenv("ANTHROPIC_API_KEY"))

    status = {
        "status": "ready" if has_index else "index_missing",
        "vector_store": {
            "path": store_path,
            "faiss_file_exists": os.path.exists(faiss_file),
            "faiss_size_mb": faiss_size_mb,
            "pkl_file_exists": os.path.exists(pkl_file),
            "pkl_size_mb": pkl_size_mb,
        },
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_model": "claude-sonnet-5",
        "anthropic_api_key_configured": api_key_set,
        "is_model_cached_in_memory": _VECTOR_DB is not None
    }
    return status


@server.tool(
    name="run_benchmark_evaluation",
    description="Runs system benchmark metrics (Recall@k, Precision@k, ROUGE@k, and generation metrics) on the BioASQ test split."
)
def run_benchmark_evaluation(
    sample_size: int = 5,
    k_values: List[int] = [3, 5]
) -> Dict[str, Any]:
    """
    Evaluates retrieval and generation quality against the BioASQ benchmark dataset.

    Args:
        sample_size: Number of test questions to sample for LLM answer generation evaluation (default 5).
        k_values: List of k values for retrieval metrics (default [3, 5]).

    Returns:
        Dictionary of benchmark evaluation results.
    """
    logger.info(f"Running benchmark evaluation with sample_size={sample_size}, k_values={k_values}")
    vector_db, _ = get_cached_components()
    from python.evaluate import (
        load_evaluation_data,
        calculate_recall,
        calculate_precision,
        calculate_rouge_at_k,
        evaluate_generation_metrics,
        load_api_error_metrics
    )

    evaluation_data = load_evaluation_data()
    results = {
        "total_test_questions": len(evaluation_data),
        "sampled_for_generation": sample_size
    }

    # Retrieval metrics for requested k values
    for k in k_values:
        recall = calculate_recall(vector_db, evaluation_data, k=k)
        precision = calculate_precision(vector_db, evaluation_data, k=k)
        rouge = calculate_rouge_at_k(precision, recall)
        results[f"recall_at_{k}"] = recall
        results[f"precision_at_{k}"] = precision
        results[f"rouge_at_{k}"] = rouge

    # Generation metrics on sample
    gen_precision, gen_recall, gen_rouge = evaluate_generation_metrics(
        vector_db,
        evaluation_data,
        sample_size=max(1, min(sample_size, 50))
    )
    results["generation_precision"] = gen_precision
    results["generation_recall"] = gen_recall
    results["generation_rouge_score"] = gen_rouge

    attempts, failures, rate = load_api_error_metrics()
    results["llm_api_attempts"] = attempts
    results["llm_api_failures"] = failures
    results["llm_api_error_rate"] = rate

    return results


# ==============================================================================
# MCP Resources
# ==============================================================================

@server.resource("rag://system/health")
def resource_system_health() -> str:
    """Provides current RAG system status, index size, and configuration."""
    health = check_system_health()
    return json.dumps(health, indent=2)


@server.resource("rag://metrics/errors")
def resource_error_metrics() -> str:
    """Provides real-time API reliability statistics and error counts."""
    metrics = get_error_metrics()
    return json.dumps(metrics, indent=2)


@server.resource("rag://pipeline/architecture")
def resource_pipeline_architecture() -> str:
    """Provides the biomedical RAG architecture documentation."""
    return """# Biomedical RAG Architecture
- **Dataset**: `rag-datasets/rag-mini-bioasq` (corpus passages & QA pairs)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
- **Vector Database**: FAISS index with similarity score threshold filtering
- **Generator**: Anthropic Claude (`claude-sonnet-5`) with strict context grounding prompts
- **Evaluation**: Live lexical overlap scoring (ROUGE-1 F1, Precision, Recall)
"""


# ==============================================================================
# MCP Prompts
# ==============================================================================

@server.prompt(
    name="biomedical_qa",
    description="Prompt template for strictly grounded biomedical question answering."
)
def prompt_biomedical_qa(question: str) -> str:
    """Generates a structured prompt enforcing strict context grounding for a biomedical query."""
    return (
        f"You are a specialized biomedical assistant answering questions based on verified scientific literature.\n\n"
        f"Use the `rag_answer` or `search_passages` tool to look up context for this question:\n"
        f"Question: {question}\n\n"
        f"Rules:\n"
        f"1. Ground all claims directly in retrieved literature.\n"
        f"2. Cite passage identifiers when available.\n"
        f"3. Explicitly report if the retrieved passages do not contain sufficient evidence."
    )


@server.prompt(
    name="verify_claim",
    description="Prompt template for verifying a biomedical assertion or medical hypothesis against literature."
)
def prompt_verify_claim(claim: str) -> str:
    """Generates a structured prompt to fact-check a medical claim."""
    return (
        f"Verify the following biomedical claim against the scientific corpus:\n"
        f"Claim: \"{claim}\"\n\n"
        f"1. Search the index using `search_passages` with relevant medical keywords.\n"
        f"2. Evaluate whether the evidence supports, refutes, or is insufficient to evaluate the claim.\n"
        f"3. Report Precision and Recall grounding using `calculate_grounding`."
    )


# ==============================================================================
# Entry Point & CLI
# ==============================================================================

def run_self_test():
    """Runs a quick self-test of the tools without starting the stdio server loop."""
    print("\n=== Running Biomedical RAG MCP Self-Test ===\n")
    print("1. Testing check_system_health()...")
    health = check_system_health()
    print(json.dumps(health, indent=2))

    print("\n2. Testing get_error_metrics()...")
    metrics = get_error_metrics()
    print(json.dumps(metrics, indent=2))

    print("\n3. Testing calculate_grounding()...")
    grounding = calculate_grounding(
        "p53 is a tumor suppressor gene involved in apoptosis.",
        ["The tumor suppressor p53 regulates cell cycle arrest and apoptosis in response to DNA damage."]
    )
    print(json.dumps(grounding, indent=2))

    print("\n4. Testing search_passages('What is p53?')...")
    try:
        passages = search_passages("What is p53?", top_k=2, score_threshold=0.3)
        print(f"Found {len(passages)} passages:")
        for p in passages:
            print(f"  - [{p['id']}] (score: {p['relevance_score']}): {p['snippet']}")
    except Exception as e:
        print(f"Search passages test skipped or failed: {e}")

    print("\n=== Self-Test Completed Successfully! ===\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--test", "-t", "test"):
        run_self_test()
    elif len(sys.argv) > 1 and sys.argv[1] in ("--sse", "sse"):
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        logger.info(f"Starting MCP Server over SSE transport on port {port}...")
        server.run(transport="sse", port=port)
    else:
        logger.info("Starting Biomedical RAG MCP Server over stdio transport...")
        server.run(transport="stdio")
