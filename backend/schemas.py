from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# --- Health & Status Schemas ---
class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    vector_store_loaded: bool = Field(..., example=True)
    total_chunks: int = Field(..., example=1009)
    model: str = Field(..., example="claude-3-5-sonnet-20241022")

# --- Query Schemas ---
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question or prompt about Python architecture or code")
    k: int = Field(3, ge=1, le=20, description="Number of top chunks to retrieve")
    score_threshold: float = Field(0.2, ge=0.0, le=1.0, description="Minimum similarity score threshold")

class ChunkItem(BaseModel):
    id: str
    name: Optional[str] = None
    file_path: Optional[str] = None
    repo: Optional[str] = None
    src_object: Optional[str] = None
    snippet: str
    content: str
    score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: List[ChunkItem]
    precision: float
    recall: float
    rouge_score: float

# --- Retrieval-Only Schemas ---
class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query or symbol")
    k: int = Field(3, ge=1, le=20, description="Number of top chunks to retrieve")
    score_threshold: float = Field(0.2, ge=0.0, le=1.0, description="Minimum similarity score threshold")

class RetrieveResponse(BaseModel):
    total_found: int
    chunks: List[ChunkItem]

# --- Dataset Management Schemas ---
class DatasetLoadRequest(BaseModel):
    config: str = Field("andromeda", description="Repository configuration name (andromeda, swarms, zeta, etc., or all)")
    split: str = Field("train", description="Split to load (train or test)")

class DatasetLoadResponse(BaseModel):
    status: str
    dataset_name: str
    config: str
    split: str
    documents_loaded: int
    chunks_indexed: int

class DatasetInfoResponse(BaseModel):
    dataset_name: str
    available_configs: List[str]
    active_config: str
    total_chunks: int
    vector_store_path: str
    embedding_model: str

# --- Observability & Evaluation Schemas ---
class MetricsResponse(BaseModel):
    total_attempts: int
    successes: int
    failures: int
    error_rate: float
    error_counts: Dict[str, int]

class EvaluateRequest(BaseModel):
    sample_size: int = Field(10, ge=1, le=100, description="Number of test queries to benchmark")

class EvaluateResponse(BaseModel):
    precision_at_3: float
    recall_at_3: float
    rouge_at_3: float
    precision_at_5: float
    recall_at_5: float
    rouge_at_5: float
    generation_precision: float
    generation_recall: float
    generation_rouge_score: float
    error_rate_at_3: float
    llm_api_attempts: int
    llm_api_failures: int
    llm_api_error_rate: float
