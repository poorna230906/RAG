import os
import sys
import json
from contextlib import asynccontextmanager
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ChunkItem,
    RetrieveRequest,
    RetrieveResponse,
    DatasetLoadRequest,
    DatasetLoadResponse,
    DatasetInfoResponse,
    MetricsResponse,
    EvaluateRequest,
    EvaluateResponse
)

from python.load_data import load_data, DATASET_NAME, CONFIGS
from python.preprocess import preprocess_data
from python.embeddings import get_embedding_model
from python.vector_db import build_faiss_index
from python.retriever import get_retriever
from python.generator import generate_response
from python.evaluate import evaluate_system

# Global state
VECTOR_STORE_PATH = os.path.join(PROJECT_ROOT, "vector_store")
METRICS_PATH = os.path.join(PROJECT_ROOT, "error_metrics.json")
state = {
    "retriever": None,
    "vector_db": None,
    "active_config": "andromeda",
    "total_chunks": 0
}

def reload_vector_store():
    """Reloads the local FAISS index into memory."""
    if os.path.exists(VECTOR_STORE_PATH):
        try:
            retriever = get_retriever(store_path=VECTOR_STORE_PATH)
            state["retriever"] = retriever
            state["vector_db"] = retriever.vectorstore
            state["total_chunks"] = state["vector_db"].index.ntotal if hasattr(state["vector_db"], "index") else 0
            print(f"[FastAPI] Vector store loaded successfully. Total chunks: {state['total_chunks']}")
        except Exception as e:
            print(f"[FastAPI] Warning: Failed to load vector store: {e}")
    else:
        print(f"[FastAPI] Warning: Vector store directory '{VECTOR_STORE_PATH}' does not exist.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the FAISS vector database
    print("[FastAPI] Initializing application & loading vector store...")
    reload_vector_store()
    yield
    # Shutdown
    print("[FastAPI] Shutting down application.")

# Initialize FastAPI App
app = FastAPI(
    title="Python Copilot RAG Backend API",
    description="Context-grounded Retrieval-Augmented Generation API for Python architecture, code comprehension, and copilot queries.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware (allows web frontends on any port to communicate with the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def root():
    """Redirects to interactive Swagger API documentation."""
    return RedirectResponse(url="/docs")

# 1. Health & Status
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns server health, vector database status, and active model."""
    is_loaded = state["vector_db"] is not None
    model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    return HealthResponse(
        status="healthy",
        vector_store_loaded=is_loaded,
        total_chunks=state.get("total_chunks", 0),
        model=model_name
    )

# 2. Main RAG Query Endpoint
@app.post("/api/query", response_model=QueryResponse, tags=["RAG"])
def query_copilot(req: QueryRequest):
    """
    Main RAG Endpoint:
    Takes a question, searches FAISS for matching code context, invokes Claude,
    calculates real-time ROUGE scores, and returns the response.
    """
    if not state["vector_db"]:
        reload_vector_store()
    if not state["vector_db"]:
        raise HTTPException(
            status_code=503,
            detail="Vector store is not loaded. Please build the index first using /api/dataset/load."
        )

    try:
        # Search using custom k and score_threshold
        results = state["vector_db"].similarity_search_with_relevance_scores(
            req.question,
            k=req.k
        )

        filtered_docs = []
        chunk_items = []
        for doc, score in results:
            if score >= req.score_threshold:
                filtered_docs.append(doc)
                chunk_items.append(
                    ChunkItem(
                        id=str(doc.metadata.get("id", "N/A")),
                        name=doc.metadata.get("name"),
                        file_path=doc.metadata.get("file_path"),
                        repo=doc.metadata.get("repo"),
                        src_object=doc.metadata.get("src_object"),
                        snippet=doc.page_content[:150].replace("\n", " ") + "...",
                        content=doc.page_content,
                        score=round(float(score), 4)
                    )
                )

        # Generate response using generator.py
        result = generate_response(req.question, filtered_docs)

        return QueryResponse(
            answer=result["answer"],
            retrieved_chunks=chunk_items,
            precision=result.get("precision", 0.0),
            recall=result.get("recall", 0.0),
            rouge_score=result.get("rouge_score", result.get("score", 0.0))
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")

# 3. Context Search Only Endpoint
@app.post("/api/retrieve", response_model=RetrieveResponse, tags=["Search"])
def retrieve_code(req: RetrieveRequest):
    """
    Search-Only Endpoint:
    Directly queries FAISS and returns matching code passages without invoking Claude (zero token cost).
    """
    if not state["vector_db"]:
        reload_vector_store()
    if not state["vector_db"]:
        raise HTTPException(
            status_code=503,
            detail="Vector store is not loaded. Please build the index first using /api/dataset/load."
        )

    try:
        results = state["vector_db"].similarity_search_with_relevance_scores(
            req.query,
            k=req.k
        )

        chunk_items = []
        for doc, score in results:
            if score >= req.score_threshold:
                chunk_items.append(
                    ChunkItem(
                        id=str(doc.metadata.get("id", "N/A")),
                        name=doc.metadata.get("name"),
                        file_path=doc.metadata.get("file_path"),
                        repo=doc.metadata.get("repo"),
                        src_object=doc.metadata.get("src_object"),
                        snippet=doc.page_content[:150].replace("\n", " ") + "...",
                        content=doc.page_content,
                        score=round(float(score), 4)
                    )
                )

        return RetrieveResponse(
            total_found=len(chunk_items),
            chunks=chunk_items
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

# 4. Dataset Management Endpoints
@app.get("/api/dataset/info", response_model=DatasetInfoResponse, tags=["Dataset"])
def get_dataset_info():
    """Returns metadata about the active dataset, indexed chunks, and available repository splits."""
    return DatasetInfoResponse(
        dataset_name=DATASET_NAME,
        available_configs=list(CONFIGS.keys()) + ["all"],
        active_config=state.get("active_config", "andromeda"),
        total_chunks=state.get("total_chunks", 0),
        vector_store_path=VECTOR_STORE_PATH,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )

@app.post("/api/dataset/load", response_model=DatasetLoadResponse, tags=["Dataset"])
def load_and_rebuild_dataset(req: DatasetLoadRequest):
    """
    Ingests the specified dataset split from Hugging Face and rebuilds the FAISS vector index.
    """
    try:
        print(f"[FastAPI] Loading dataset split: config='{req.config}', split='{req.split}'...")
        corpus = load_data(config=req.config, split=req.split)
        chunks = preprocess_data(corpus)
        embedding_model = get_embedding_model()

        # Build and save FAISS index
        vector_db = build_faiss_index(chunks, embedding_model, store_path=VECTOR_STORE_PATH)

        # Update in-memory state
        state["vector_db"] = vector_db
        state["retriever"] = get_retriever(store_path=VECTOR_STORE_PATH)
        state["active_config"] = req.config
        state["total_chunks"] = len(chunks)

        return DatasetLoadResponse(
            status="success",
            dataset_name=DATASET_NAME,
            config=req.config,
            split=req.split,
            documents_loaded=len(corpus),
            chunks_indexed=len(chunks)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {str(e)}")

# 5. Observability & Error Metrics Endpoint
@app.get("/api/metrics", response_model=MetricsResponse, tags=["Observability"])
def get_metrics():
    """Reads persistent telemetry from error_metrics.json."""
    if not os.path.exists(METRICS_PATH):
        return MetricsResponse(
            total_attempts=0,
            successes=0,
            failures=0,
            error_rate=0.0,
            error_counts={}
        )

    try:
        with open(METRICS_PATH, "r") as f:
            data = json.load(f)
        return MetricsResponse(
            total_attempts=data.get("total_attempts", 0),
            successes=data.get("successes", 0),
            failures=data.get("failures", 0),
            error_rate=data.get("error_rate", 0.0),
            error_counts=data.get("error_counts", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read metrics: {str(e)}")

# 6. Benchmark Evaluation Endpoint
@app.post("/api/evaluate", response_model=EvaluateResponse, tags=["Observability"])
def run_evaluation(req: EvaluateRequest):
    """
    Runs the automated evaluation benchmark across the test dataset and returns metrics.
    """
    try:
        results = evaluate_system()
        return EvaluateResponse(**results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
