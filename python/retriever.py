import sys
import os
from langchain_community.vectorstores import FAISS

def get_retriever(store_path="vector_store"):
    """
    Loads the saved FAISS index and returns it as a retriever.
    Args:
        store_path (str): Local path to the saved FAISS store.
    Returns:
        VectorStoreRetriever: LangChain retriever instance.
    """
    # Import inside function to avoid circular import if needed
    from python.embeddings import get_embedding_model
    
    if not os.path.exists(store_path):
        raise FileNotFoundError(f"FAISS index folder not found at '{store_path}'. Please run build_index.py first.")

    print(f"Loading local FAISS index from '{store_path}'...")
    embedding_model = get_embedding_model()
    vector_db = FAISS.load_local(
        store_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return vector_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.4
    }
)

# Test the retriever
if __name__ == "__main__":
    # Add parent directory to path so we can import python.* when run directly
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print(" Test: Loading and Querying Retriever ")
    try:
        # Looking for store_path relative to project root or test parent directory
        store_path = "../vector_store" if os.path.exists("../vector_store") else "vector_store"
        retriever = get_retriever(store_path=store_path)
        
        query = "What is BioASQ?"
        print(f"Query: '{query}'")
        
        results = retriever.invoke(query)
        print(f"Retrieved {len(results)} chunks:")
        for idx, doc in enumerate(results):
            print(f"[{idx+1}] ID: {doc.metadata.get('id', 'N/A')}")
            print(f"    Content: {doc.page_content[:150]}...")
    except FileNotFoundError as e:
        print(f"Test skipped or failed: {e}")
        print("Build the index first using build_index.py or running vector_db.py test.")
