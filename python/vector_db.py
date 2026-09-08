import sys
import os
from langchain_community.vectorstores import FAISS

def build_faiss_index(documents, embedding_model, store_path="vector_store"):
    """
    Creates and saves a FAISS index locally.
    Args:
        documents (list[Document]): List of chunked documents to index.
        embedding_model: Embedding model instance.
        store_path (str): Path to save the FAISS vector store.
    Returns:
        FAISS: Loaded FAISS vector database.
    """
    print(f"Creating FAISS index from {len(documents)} document chunks...")
    vector_db = FAISS.from_documents(
        documents,
        embedding_model
    )

    vector_db.save_local(store_path)
    return vector_db

# Test the FAISS index creation
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from python.load_data import load_data
    from python.preprocess import preprocess_data
    from python.embeddings import get_embedding_model

    print(" Test: Building FAISS Index ")
    print("Loading documents...")
    corpus = load_data()
    documents = preprocess_data(corpus)

    print("Loading embedding model...")
    embedding_model = get_embedding_model()

    print("Creating FAISS index...")
    vector_db = build_faiss_index(documents, embedding_model, store_path="../vector_store")

    print("\nFAISS Index Created and Saved Successfully in Test Mode!")
