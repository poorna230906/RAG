from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

llm = ChatAnthropic(
    model="claude-sonnet-5",
    
)


def generate_response(query: str, retrieved_docs) -> str:
    """
    Generates a response using the retrieved documents as context.
    """

    if not retrieved_docs:
        return "I could not find any relevant information to answer your question."

    context = "\n".join(
        [f"- {doc.page_content}" for doc in retrieved_docs]
    )

    prompt = (
    "Answer the question using ONLY the provided context.\n"
    "Do not use outside knowledge or make up information.\n"
    "If the context does not contain enough information to answer "
    "the question, say: 'No relevant information found in the provided context.'\n"
    "Give a direct and concise answer.\n\n"
    f"Context:\n{context}\n\n"
    f"Question: {query}\n"
    "Answer:"
)

    response = llm.invoke(prompt)
    return {
    "answer": response,
    "grounding": grounding,
    "score": score
}


# Test the generator
if __name__ == "__main__":
    from langchain_core.documents import Document

    print(" Test: Generator ")

    mock_docs = [
        Document(
            page_content="BioASQ is a series of challenges on biomedical semantic indexing and question answering.",
            metadata={"id": "1"}
        ),
        Document(
            page_content="The BioASQ tasks include semantic indexing, question answering, and information retrieval.",
            metadata={"id": "2"}
        )
    ]

    query = "What is the BioASQ challenge?"

    answer = generate_response(query, mock_docs)

    print("\n Answer ")
    print(answer)

