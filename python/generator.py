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
    model="claude-sonnet-5",
)

# Embedding model for grounding calculation
embedding_model = get_embedding_model()


def calculate_grounding_score(answer, retrieved_docs):
    """
    Calculates grounding score using semantic similarity.

    Formula:
    Grounding Score =
    Average of the highest similarity score
    for each answer sentence.
    """

    if not answer or not retrieved_docs:
        return 0.0

    # Split answer into sentences
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    if not sentences:
        return 0.0

    # Get retrieved context
    context_chunks = [
        doc.page_content.strip()
        for doc in retrieved_docs
        if doc.page_content.strip()
    ]

    if not context_chunks:
        return 0.0

    # Create embeddings
    answer_embeddings = embedding_model.embed_documents(sentences)
    context_embeddings = embedding_model.embed_documents(context_chunks)

    # Calculate cosine similarity
    similarities = cosine_similarity(
        answer_embeddings,
        context_embeddings
    )

    sentence_scores = []

    # Find the best matching context for each answer sentence
    for row in similarities:
        best_similarity = max(row)

        best_similarity = max(
            0.0,
            min(1.0, float(best_similarity))
        )

        sentence_scores.append(best_similarity)

    # Calculate average grounding score
    grounding_score = (
        sum(sentence_scores) / len(sentence_scores)
    )

    return round(grounding_score, 4)


def generate_response(query: str, retrieved_docs) -> dict:
    """
    Generates an answer using retrieved documents.

    Includes:
    - Claude API call
    - API error tracking
    - Retry mechanism
    - Fallback answer
    - Dynamic grounding score
    """

    # No relevant documents
    if not retrieved_docs:
        return {
            "answer": (
                "I could not find any relevant information "
                "to answer your question."
            ),
            "grounding": "Not Grounded",
            "score": 0.0
        }

    # Prepare context
    context = "\n".join(
        [f"- {doc.page_content}" for doc in retrieved_docs]
    )
    # Prompt for Claude
    prompt = (
        "You are a biomedical question-answering assistant.\n"
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
        "6.Answer exactly what the user asked and avoid "
        "unnecessary details.\n"
        "7.When possible, use the terminology and facts given "
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
    api_key = os.getenv("")

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
                "grounding": "Not Grounded",
                "score": 0.0
            }

        # Calculate grounding score dynamically
        score = calculate_grounding_score(
            response_text,
            retrieved_docs
        )

        # Determine grounding status from calculated score
        if score >= 0.70:
            grounding = "Grounded"
        else:
            grounding = "Not Grounded"

        return {
            "answer": response_text,
            "grounding": grounding,
            "score": score
        }

    # API failure
    err_msg = str(last_exception)

    log_api_attempt(
        success=False,
        error_msg=err_msg
    )

    # Fallback answer
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

    fallback_score = calculate_grounding_score(
        fallback_answer,
        retrieved_docs
    )

    if fallback_score >= 0.70:
        fallback_grounding = "Grounded"
    else:
        fallback_grounding = "Not Grounded"

    return {
        "answer": fallback_answer,
        "grounding": f"Mitigated Fallback ({fallback_grounding})",
        "score": fallback_score
    }