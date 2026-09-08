import os
import json
import ast
from sklearn.metrics import recall_score
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from python.embeddings import get_embedding_model
def load_evaluation_data():
    """Load BioASQ questions and their relevant passage IDs."""
    dataset = load_dataset(
        "rag-datasets/rag-mini-bioasq",
        "question-answer-passages",
        split="test"
    )
    return dataset
def parse_relevant_ids(value):
    """Convert relevant passage IDs into a set of strings."""
    if isinstance(value, list):
        return {str(x) for x in value}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return {str(x) for x in parsed}
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return {str(x) for x in parsed}
            except (ValueError, SyntaxError):
                pass
    return set()
def load_vector_database(store_path="vector_store"):
    """Load the existing FAISS vector database."""
    if not os.path.exists(store_path):
        raise FileNotFoundError(
            f"FAISS index not found at '{store_path}'. "
            "Please run build_index.py first."
        )
    embedding_model = get_embedding_model()
    vector_db = FAISS.load_local(
        store_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return vector_db
def calculate_recall(vector_db, evaluation_data, k):
    """
    Calculate Recall@k using the BioASQ relevant passage IDs.
    """
    y_true = []
    y_pred = []
    for row in evaluation_data:
        question = row["question"]
        relevant_ids = parse_relevant_ids(
            row["relevant_passage_ids"]
        )
        if not relevant_ids:
            continue
        retrieved_docs = vector_db.similarity_search(
            question,
            k=k
        )
        retrieved_ids = {
            str(doc.metadata.get("id"))
            for doc in retrieved_docs
        }
        for passage_id in relevant_ids:
            y_true.append(1)
            if passage_id in retrieved_ids:
                y_pred.append(1)
            else:
                y_pred.append(0)
    if not y_true:
        return 0.0
    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )
    return round(recall, 4)
def load_api_error_metrics():
    """Read Claude API error metrics."""
    metrics_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "error_merics.json"
    )
    attempts = 0
    failures = 0
    error_rate = 0.0
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                data = json.load(f)
            attempts = data.get("total_attempts", 0)
            failures = data.get("failures", 0)
            error_rate = data.get("error_rate", 0.0)
        except Exception as e:
            print(
                f"[Warning] Failed to read error metrics: {e}"
            )
    return attempts, failures, error_rate
def evaluate_system():
    """Evaluate the RAG retrieval system."""
    print("Evaluating RAG system...")
    # Load evaluation questions
    evaluation_data = load_evaluation_data()
    print(
        f"Evaluation questions: {len(evaluation_data)}"
    )
    # Load FAISS database
    vector_db = load_vector_database()
    # Calculate Recall@3
    print("Calculating Recall@3...")
    recall_at_3 = calculate_recall(
        vector_db,
        evaluation_data,
        k=3
    )
    # Calculate Recall@5
    print("Calculating Recall@5...")
    recall_at_5 = calculate_recall(
        vector_db,
        evaluation_data,
        k=5
    )
    # Calculate retrieval error rates
    error_rate_at_3 = round(
        1 - recall_at_3,
        4
    )
    error_rate_at_5 = round(
        1 - recall_at_5,
        4
    )
    # Load actual API error metrics
    (
        llm_attempts,
        llm_failures,
        llm_error_rate
    ) = load_api_error_metrics()
    # Final results
    results = {
        "recall_at_3": recall_at_3,
        "recall_at_5": recall_at_5,
        "error_rate_at_3": error_rate_at_3,
        "error_rate_at_5": error_rate_at_5,
        "llm_api_attempts": llm_attempts,
        "llm_api_failures": llm_failures,
        "llm_api_error_rate": llm_error_rate
    }
    return results
if __name__ == "__main__":

    print("\n--- RAG Evaluation ---")

    metrics = evaluate_system()

    print("\nResults:")

    for metric, value in metrics.items():
        print(f"{metric}: {value}")