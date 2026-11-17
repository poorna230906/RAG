import os
import json
import time
import re
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from sklearn.metrics.pairwise import cosine_similarity

from python.embeddings import get_embedding_model

load_dotenv()

# Error metrics file
METRICS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "error_metrics.json"
)
def log_api_attempt(success: bool, error_msg: str = None):
    """Logs API attempts, successes, failures and error rate."""

    try:
        data = {
            "total_attempts": 0,
            "successes": 0,
            "failures": 0,
            "error_rate": 0.0,
            "error_counts": {}
        }

        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    pass

        data["total_attempts"] += 1

        if success:
            data["successes"] += 1
        else:
            data["failures"] += 1

            if error_msg:
                short_error = error_msg.split("\n")[0][:100]
                data["error_counts"][short_error] = (
                    data["error_counts"].get(short_error, 0) + 1
                )

        if data["total_attempts"] > 0:
            data["error_rate"] = round(
                data["failures"] / data["total_attempts"],
                4
            )

        with open(METRICS_FILE, "w") as f:
            json.dump(data, f, indent=4)

    except Exception as e:
        print(f"[Warning] Failed to log API metrics: {e}")


# Claude model
llm = ChatAnthropic(
    model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
)

def tokenize(text: str) -> list:
    """Tokenize text into lowercase words, ignoring punctuation."""
    if not text:
        return []
    return re.findall(r'\b\w+\b', text.lower())


def calculate_rouge_score(answer: str, retrieved_docs) -> dict:
    """
    Calculates dynamic ROUGE score (ROUGE-1 F1), Precision, and Recall
    between the generated answer and the retrieved context documents.

    - Precision: Fraction of answer tokens present in retrieved context.
    - Recall: Fraction of context tokens present in answer.
    - ROUGE Score: Harmonic mean (F1 score) of Precision and Recall.
    """
    if not answer or not retrieved_docs:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "rouge_score": 0.0
        }

    answer_tokens = tokenize(answer)
    if not answer_tokens:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "rouge_score": 0.0
        }

    # Extract context text from retrieved documents
    context_passages = []
    for doc in retrieved_docs:
        if hasattr(doc, 'page_content') and doc.page_content:
            context_passages.append(doc.page_content.strip())
        elif isinstance(doc, str):
            context_passages.append(doc.strip())

    if not context_passages:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "rouge_score": 0.0
        }

    # Combine full context for precision matching
    full_context_text = " ".join(context_passages)
    full_context_tokens = tokenize(full_context_text)

    if not full_context_tokens:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "rouge_score": 0.0
        }

    from collections import Counter
    answer_counts = Counter(answer_tokens)
    full_context_counts = Counter(full_context_tokens)

    # Calculate token overlap between answer and context
    overlap_count = sum(
        min(count, full_context_counts[token])
        for token, count in answer_counts.items()
    )

    # Precision: proportion of generated answer tokens supported by context
    precision = overlap_count / len(answer_tokens)

    # Recall: max recall across context passages or full context
    best_passage_recall = 0.0
    for passage in context_passages:
        p_tokens = tokenize(passage)
        if not p_tokens:
            continue
        p_counts = Counter(p_tokens)
        p_overlap = sum(
            min(count, p_counts[token])
            for token, count in answer_counts.items()
        )
        p_recall = p_overlap / len(p_tokens)
        if p_recall > best_passage_recall:
            best_passage_recall = p_recall

    full_context_recall = overlap_count / len(full_context_tokens)
    recall = max(best_passage_recall, full_context_recall)

    # ROUGE F1 Score
    if precision + recall > 0:
        rouge_score = (2 * precision * recall) / (precision + recall)
    else:
        rouge_score = 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "rouge_score": round(rouge_score, 4)
    }


def calculate_grounding_score(answer, retrieved_docs):
    """Legacy alias for calculate_rouge_score."""
    metrics = calculate_rouge_score(answer, retrieved_docs)
    return metrics["rouge_score"]


def generate_response(query: str, retrieved_docs) -> dict:
    """
    Generates an answer using retrieved documents.

    Includes:
    - Claude API call
    - API error tracking
    - Retry mechanism
    - Fallback answer
    - Dynamic Precision, Recall, and ROUGE score
    """

    # No relevant documents
    if not retrieved_docs:
        return {
            "answer": (
                "I could not find any relevant information "
                "to answer your question."
            ),
            "precision": 0.0,
            "recall": 0.0,
            "rouge_score": 0.0,
            "score": 0.0
        }

    # Prepare context
    context = "\n".join(
        [f"- {doc.page_content}" for doc in retrieved_docs]
    )
    # Prompt for Claude
    prompt = (
        "You are an AI Python code assistant and copilot.\n"
        "Your task is to answer the user's question using ONLY the "
        "information provided in the retrieved context.\n\n"
        "Follow these rules strictly:\n"
        "1.Use only information present in the retrieved context.\n"
        "2.Do not use outside knowledge, assumptions, or personal "
        "reasoning that is not supported by the context.\n"
        "3.Do not invent, guess, or add facts that are not present "
        "in the context.\n"
        "4.If the retrieved context does not contain enough "
        "information to answer the question, clearly state that "
        "there is not enough relevant information in the provided "
        "context.\n"
        "5.Keep the answer direct, clear, and concise.\n"
        "6.Answer exactly what the user asked and provide code examples, "
        "arguments, returns, or architectural explanations if given in context.\n"
        "7.When possible, use the exact identifiers, classes, methods, and facts given "
        "in the context.\n"
        "8.If multiple retrieved passages provide relevant "
        "information, combine them carefully without adding "
        "unsupported information.\n\n"
        "Retrieved Context:\n"
        f"{context}\n\n"

        "User Question:\n"
        f"{query}\n\n"
        "Answer:"
    )

    # API retry configuration
    max_retries = 2
    delay = 1.0

    response_text = None
    success = False
    last_exception = None
    api_key = os.getenv("ANTHROPIC_API_KEY")

    # Claude API call
    for attempt in range(max_retries + 1):
        try:
            response = llm.invoke(prompt)

            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)

            success = True
            break

        except (ValueError, TypeError) as e:
            last_exception = e
            break

        except Exception as e:
            last_exception = e

            if attempt < max_retries:
                time.sleep(delay * (2 ** attempt))
            else:
                break

    # Successful Claude response
    if success:
        log_api_attempt(success=True)

        refusal_keywords = [
            "no relevant information",
            "could not find any relevant",
            "don't have enough information",
            "no relevant information found"
        ]

        answer_lower = response_text.lower()

        if any(keyword in answer_lower for keyword in refusal_keywords):
            return {
                "answer": response_text,
                "precision": 0.0,
                "recall": 0.0,
                "rouge_score": 0.0,
                "score": 0.0
            }

        # Calculate Precision, Recall, and ROUGE score dynamically
        metrics = calculate_rouge_score(
            response_text,
            retrieved_docs
        )

        return {
            "answer": response_text,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "rouge_score": metrics["rouge_score"],
            "score": metrics["rouge_score"]
        }

    # API failure
    err_msg = str(last_exception)

    log_api_attempt(
        success=False,
        error_msg=err_msg
    )

    # Fallback answer
    print(f"\n[Notice] LLM API call failed ({err_msg[:120]}). Using retrieved context as fallback.")
    fallback_answer = (
        "[Fallback Answer - API Error Mitigated]\n"
        "An error occurred while generating the LLM response, "
        "but here is the information retrieved from the database:\n"
    )

    for idx, doc in enumerate(retrieved_docs):
        fallback_answer += (
            f"- Chunk {idx + 1}: "
            f"{doc.page_content.strip()}\n"
        )

    fallback_metrics = calculate_rouge_score(
        fallback_answer,
        retrieved_docs
    )

    return {
        "answer": fallback_answer,
        "precision": fallback_metrics["precision"],
        "recall": fallback_metrics["recall"],
        "rouge_score": fallback_metrics["rouge_score"],
        "score": fallback_metrics["rouge_score"]
    }