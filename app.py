import os
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

try:
    import httpx
    # Monkeypatch httpx Clients to disable SSL verification
    original_client_init = httpx.Client.__init__
    def patched_client_init(self, *args, **kwargs):
        kwargs['verify'] = False
        original_client_init(self, *args, **kwargs)
    httpx.Client.__init__ = patched_client_init
    
    original_async_client_init = httpx.AsyncClient.__init__
    def patched_async_client_init(self, *args, **kwargs):
        kwargs['verify'] = False
        original_async_client_init(self, *args, **kwargs)
    httpx.AsyncClient.__init__ = patched_async_client_init
except Exception:
    pass

try:
    import requests
    # Monkeypatch requests to disable SSL verification
    original_session_request = requests.Session.request
    def patched_session_request(self, method, url, **kwargs):
        kwargs['verify'] = False
        return original_session_request(self, method, url, **kwargs)
    requests.Session.request = patched_session_request
except Exception:
    pass

from python.retriever import get_retriever
from python.generator import generate_response

def run_app():
    """
    Coordinates Pipeline 2:
    - Loads the FAISS retriever
    - Loops to accept user queries
    - Retrieves context documents matching queries
    - Generates responses
    """
    print("Starting Pipeline 2: RAG Application Chat\n")

    store_path = "vector_store"
    if not os.path.exists(store_path):
        print(f"Error: Vector store not found at '{store_path}'.")
        print("Please build the index first by running: python build_index.py\n")
        return

    print("[Pipeline] Loading vector database retriever...")
    try:
        retriever = get_retriever(store_path=store_path)
        print("[Pipeline] Retriever loaded successfully.")
    except Exception as e:
        print(f"Error loading retriever: {e}")
        return

    print("\nSystem ready! Type 'exit' or 'quit' to close the application.\n")

    while True:
        try:
            query = input("Ask a question: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if query.strip().lower() in ("exit", "quit"):
            print("Exiting RAG Application. Goodbye!")
            break

        if not query.strip():
            continue

        print("\n[RAG] Searching for relevant context passages...")
        try:
            retrieved_docs = retriever.invoke(query)
            if not retrieved_docs:
                print("[RAG] No relevant passages found based on the score threshold.")
                continue
            print(f"[RAG] Found {len(retrieved_docs)} matching passages.")
            
            # Print matching passages metadata/contents briefly
            for idx, doc in enumerate(retrieved_docs):
                passage_id = doc.metadata.get("id", "Unknown")
                snippet = doc.page_content[:120].replace('\n', ' ')
                print(f"  - Chunk {idx + 1} (ID: {passage_id}): \"{snippet}...\"")
            
            print("\n[RAG] Generating response...")

            result = generate_response(query, retrieved_docs)

            print("\n--- Answer ---")
            print(result["answer"])

            print("\n--- Grounding Evaluation ---")
            print("Status:", result["grounding"])
            print("Score:", result["score"])
        except Exception as e:
            print(f"An error occurred while processing query: {e}\n")

if __name__ == "__main__":
    run_app()
