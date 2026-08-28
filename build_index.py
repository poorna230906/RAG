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

from python.load_data import load_data
from python.preprocess import preprocess_data
from python.embeddings import get_embedding_model
from python.vector_db import build_faiss_index

def build_pipeline():
    """
    Coordinates Pipeline 1:
    - Loads raw dataset
    - Cleans and chunks passages
    - Initializes the embedding model
    - Indexes chunks into FAISS vector database
    - Saves index locally
    """
    print("Starting Pipeline 1: Build Vector Index\n")

    # Step 1: Load Raw Dataset
    print("[Pipeline] Step 1: Loading raw dataset corpus...")
    corpus = load_data()

    # Step 2: Clean and Chunk passages
    print("\n[Pipeline] Step 2: Cleaning and chunking passages...")
    chunks = preprocess_data(corpus)

    # Step 3: Load embedding model
    print("\n[Pipeline] Step 3: Loading embedding model...")
    embedding_model = get_embedding_model()

    # Step 4: Build FAISS Vector Database and save
    print("\n[Pipeline] Step 4: Building and saving FAISS index...")
    vector_db = build_faiss_index(chunks, embedding_model, store_path="vector_store")

    print("Pipeline 1 Completed Successfully!")
    print("Vector database is saved under 'vector_store/'")

if __name__ == "__main__":
    build_pipeline()
