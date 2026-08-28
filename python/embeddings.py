import os
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

try:
    import httpx
    
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

    original_session_request = requests.Session.request
    def patched_session_request(self, method, url, **kwargs):
        kwargs['verify'] = False
        return original_session_request(self, method, url, **kwargs)
    requests.Session.request = patched_session_request
except Exception:
    pass

if __name__ == "__main__" and __package__ is None:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    """
    Initializes and returns the HuggingFaceEmbeddings model.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model

# Testing the embedding model
#if __name__ == "__main__":
 #   print("Loading embedding model...")
 #   embedding_model = get_embedding_model()

  #  text = "i am poorna chandra , living in bangalore."
  #  embedding = embedding_model.embed_query(text)

  #  print("\nEmbedding model loaded successfully!")
   # print("Embedding Length:", len(embedding))
    #print("First 10 Values:")
    #print(embedding[:10])
