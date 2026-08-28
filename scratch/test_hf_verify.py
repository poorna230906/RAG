import httpx
import huggingface_hub.utils._http as h

# Define custom client factories that disable SSL verification
def custom_client_factory() -> httpx.Client:
    return httpx.Client(
        event_hooks={"request": [h.hf_request_event_hook]},
        follow_redirects=True,
        timeout=None,
        verify=False
    )

def custom_async_client_factory() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        event_hooks={"request": [h.async_hf_request_event_hook]},
        event_hooks_response={"response": [h.async_hf_response_event_hook]},
        follow_redirects=True,
        timeout=None,
        verify=False
    )

h.set_client_factory(custom_client_factory)
h.set_async_client_factory(custom_async_client_factory)

# Now try importing sentence-transformers and downloading the model
print("Trying to load SentenceTransformer model...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded successfully!")
