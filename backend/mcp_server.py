import os
import sys
import json
import argparse

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mcp.server.mcpserver import MCPServer

from python.retriever import get_retriever
from python.generator import generate_response
from python.load_data import DATASET_NAME, CONFIGS

# Initialize Model Context Protocol Server
mcp_server = MCPServer(
    name="python-copilot-rag",
    version="1.0.0",
    instructions=(
        "You are connected to the Python Copilot RAG MCP server. "
        "Use 'ask_python_copilot' to answer questions grounded in the indexed Python codebase. "
        "Use 'search_code_snippets' to find raw classes, functions, and architecture definitions."
    )
)

VECTOR_STORE_PATH = os.path.join(PROJECT_ROOT, "vector_store")
METRICS_PATH = os.path.join(PROJECT_ROOT, "error_metrics.json")

def _get_active_retriever():
    """Lazily loads the retriever."""
    return get_retriever(store_path=VECTOR_STORE_PATH)

# --- MCP Tool 1: Full RAG Copilot Query ---
@mcp_server.tool()
def ask_python_copilot(question: str) -> str:
    """
    Asks a question about Python codebase architecture, class definitions,
    or method implementations. Retrieves relevant context from the FAISS database
    and returns a context-grounded response formulated with Claude.

    Args:
        question: The user query or coding question.
    """
    try:
        retriever = _get_active_retriever()
        docs = retriever.invoke(question)
        if not docs:
            return "No relevant context found in the vector database for this question."

        result = generate_response(question, docs)
        answer = result.get("answer", "")
        rouge = result.get("rouge_score", 0.0)
        precision = result.get("precision", 0.0)
        recall = result.get("recall", 0.0)

        output = f"Answer:\n{answer}\n\n[Evaluation Metrics: Precision: {precision:.4f}, Recall: {recall:.4f}, ROUGE-1 F1: {rouge:.4f}]"
        return output
    except Exception as e:
        return f"Error executing copilot query: {str(e)}"

# --- MCP Tool 2: Raw Code Search ---
@mcp_server.tool()
def search_code_snippets(query: str, k: int = 3) -> str:
    """
    Searches the FAISS vector database directly for matching Python code snippets,
    class specifications, and method signatures without invoking the LLM (zero token cost).

    Args:
        query: The code symbol, class name, or technical search term.
        k: Number of matching passages to return (default: 3).
    """
    try:
        retriever = _get_active_retriever()
        results = retriever.vectorstore.similarity_search_with_relevance_scores(query, k=k)
        if not results:
            return f"No code snippets found matching '{query}'."

        snippets = []
        for idx, (doc, score) in enumerate(results):
            name = doc.metadata.get("name", "Unknown")
            path = doc.metadata.get("file_path", "Unknown")
            repo = doc.metadata.get("repo", "Unknown")
            score_str = f"{score:.4f}" if score is not None else "N/A"
            header = f"--- [Result {idx + 1}] {name} ({path}) | Repo: {repo} | Similarity Score: {score_str} ---"
            snippets.append(f"{header}\n{doc.page_content.strip()}")

        return "\n\n".join(snippets)
    except Exception as e:
        return f"Error retrieving snippets: {str(e)}"

# --- MCP Tool 3: Telemetry & Error Metrics ---
@mcp_server.tool()
def get_system_metrics() -> str:
    """
    Inspects API telemetry and persistent error logs from error_metrics.json,
    showing total LLM attempts, successes, failures, and error rates.
    """
    if not os.path.exists(METRICS_PATH):
        return "No metrics file found yet. No API requests have been logged."

    try:
        with open(METRICS_PATH, "r") as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error reading metrics: {str(e)}"

# --- MCP Tool 4: Dataset & Vector Store Info ---
@mcp_server.tool()
def get_dataset_info() -> str:
    """
    Returns information about the active dataset, vector store path,
    and available repository splits that can be indexed.
    """
    try:
        total_chunks = "Unknown"
        if os.path.exists(VECTOR_STORE_PATH):
            retriever = _get_active_retriever()
            if hasattr(retriever.vectorstore, "index"):
                total_chunks = retriever.vectorstore.index.ntotal

        info = {
            "dataset_name": DATASET_NAME,
            "vector_store_path": VECTOR_STORE_PATH,
            "total_chunks_indexed": total_chunks,
            "available_configs": list(CONFIGS.keys()) + ["all"],
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
        }
        return json.dumps(info, indent=2)
    except Exception as e:
        return f"Error fetching dataset info: {str(e)}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python Copilot RAG MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio", help="MCP transport (default: stdio)")
    parser.add_argument("--port", type=int, default=8001, help="Port for SSE transport (default: 8001)")
    args = parser.parse_args()

    print(f"Starting MCP server using transport: '{args.transport}'...", file=sys.stderr)
    if args.transport == "stdio":
        mcp_server.run(transport="stdio")
    else:
        mcp_server.run(transport="sse", port=args.port)
