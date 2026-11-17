import os
import sys
import json
import ast
import re
from sklearn.metrics import recall_score
from datasets import load_dataset
from langchain_community.vectorstores import FAISS

try:
    from python.embeddings import get_embedding_model
    from python.generator import calculate_rouge_score, generate_response
    from python.load_data import load_data
except ModuleNotFoundError:
    try:
        from embeddings import get_embedding_model
        from generator import calculate_rouge_score, generate_response
        from load_data import load_data
    except ModuleNotFoundError:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from python.embeddings import get_embedding_model
        from python.generator import calculate_rouge_score, generate_response
        from python.load_data import load_data

def load_evaluation_data():
    """
    Loads test questions and target identifiers from the Python Copilot dataset:
    matlok/multimodal-python-copilot-training-overview (andromeda test split).
    """
    print("Loading evaluation data from matlok Python Copilot test split...")
    raw_dataset = load_data(config="andromeda", split="test")

    evaluation_data = []
    for idx, row in enumerate(raw_dataset):
        desc = str(row.get("desc") or "")
        name = str(row.get("name") or row.get("class_name") or "")
        file_path = str(row.get("file_path") or "")

        match = re.search(r"prompt:\s*(.*?)\s*Respond with", desc, re.DOTALL)
        if match:
            question = match.group(1).replace("\n", " ").strip()
        else:
            question = f"How to use and implement {name} in {file_path}?"

        target_ids = []
        if file_path and name:
            target_ids.append(f"{file_path}::{name}")
        if name:
            target_ids.append(name)
        if not target_ids:
            target_ids.append(str(idx))

        evaluation_data.append({
            "question": question,
            "relevant_passage_ids": target_ids,
            "name": name,
            "file_path": file_path
        })
    return evaluation_data

def parse_relevant_ids(value):
    """Convert relevant passage IDs into a set of strings."""
    if isinstance(value, (list, set)):
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
        return {value}
    return set()

def load_vector_database(store_path="vector_store"):
    """Load the existing FAISS vector database."""
    if not os.path.exists(store_path):
        candidate = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vector_store")
        if os.path.exists(candidate):
            store_path = candidate
        else:
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
    Calculate Recall@k across evaluation queries.
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
        retrieved_ids = set()
        for doc in retrieved_docs:
            if doc.metadata.get("id"):
                retrieved_ids.add(str(doc.metadata.get("id")))
            if doc.metadata.get("name"):
                retrieved_ids.add(str(doc.metadata.get("name")))
            if doc.metadata.get("file_path") and doc.metadata.get("file_path") == row.get("file_path"):
                retrieved_ids.update(relevant_ids)

        y_true.append(1)
        if relevant_ids.intersection(retrieved_ids):
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

def calculate_precision(vector_db, evaluation_data, k):
    """
    Calculate Precision@k across evaluation queries.
    """
    precisions = []
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
        hits = 0
        for doc in retrieved_docs:
            doc_id = str(doc.metadata.get("id") or "")
            doc_name = str(doc.metadata.get("name") or "")
            doc_file = str(doc.metadata.get("file_path") or "")
            if (doc_id in relevant_ids) or (doc_name in relevant_ids) or (doc_file and doc_file == row.get("file_path")):
                hits += 1
        precisions.append(hits / k)
    if not precisions:
        return 0.0
    return round(sum(precisions) / len(precisions), 4)

def calculate_rouge_at_k(precision, recall):
    """Calculate ROUGE F1 score from Precision and Recall."""
    if precision + recall > 0:
        return round((2 * precision * recall) / (precision + recall), 4)
    return 0.0

def evaluate_generation_metrics(vector_db, evaluation_data, sample_size=10):
    """
    Evaluates LLM answer generation Precision, Recall, and ROUGE metrics by calling generator.py.
    """
    precisions = []
    recalls = []
    rouge_scores = []

    total = len(evaluation_data)
    limit = min(sample_size, total)
    for i in range(limit):
        row = evaluation_data[i]
        question = row["question"]
        retrieved_docs = vector_db.similarity_search(question, k=3)
        if not retrieved_docs:
            continue
        result = generate_response(question, retrieved_docs)
        precisions.append(result.get("precision", 0.0))
        recalls.append(result.get("recall", 0.0))
        rouge_scores.append(result.get("rouge_score", 0.0))

    if not precisions:
        return 0.0, 0.0, 0.0

    avg_precision = round(sum(precisions) / len(precisions), 4)
    avg_recall = round(sum(recalls) / len(recalls), 4)
    avg_rouge = round(sum(rouge_scores) / len(rouge_scores), 4)
    return avg_precision, avg_recall, avg_rouge

def load_api_error_metrics():
    """Read Claude API error metrics."""
    metrics_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "error_metrics.json"
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
            print(f"[Warning] Failed to read error metrics: {e}")
    return attempts, failures, error_rate

def evaluate_system():
    """Evaluate the RAG retrieval and generation system."""
    print("Evaluating RAG system...")
    evaluation_data = load_evaluation_data()
    print(f"Evaluation questions: {len(evaluation_data)}")

    vector_db = load_vector_database()

    print("Calculating retrieval metrics @ k=3...")
    recall_at_3 = calculate_recall(vector_db, evaluation_data, k=3)
    precision_at_3 = calculate_precision(vector_db, evaluation_data, k=3)
    rouge_at_3 = calculate_rouge_at_k(precision_at_3, recall_at_3)

    print("Calculating retrieval metrics @ k=5...")
    recall_at_5 = calculate_recall(vector_db, evaluation_data, k=5)
    precision_at_5 = calculate_precision(vector_db, evaluation_data, k=5)
    rouge_at_5 = calculate_rouge_at_k(precision_at_5, recall_at_5)

    print("Evaluating LLM answer generation metrics (calls generator.py)...")
    gen_precision, gen_recall, gen_rouge = evaluate_generation_metrics(
        vector_db,
        evaluation_data,
        sample_size=10
    )

    error_rate_at_3 = round(1 - recall_at_3, 4)

    llm_attempts, llm_failures, llm_error_rate = load_api_error_metrics()

    results = {
        "precision_at_3": precision_at_3,
        "recall_at_3": recall_at_3,
        "rouge_at_3": rouge_at_3,
        "precision_at_5": precision_at_5,
        "recall_at_5": recall_at_5,
        "rouge_at_5": rouge_at_5,
        "generation_precision": gen_precision,
        "generation_recall": gen_recall,
        "generation_rouge_score": gen_rouge,
        "error_rate_at_3": error_rate_at_3,
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
