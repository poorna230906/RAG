import sys
import os
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def clean_text(text):
    """Normalize whitespace and strip text."""
    if not text:
        return ""
    text = text.strip()
    text = " ".join(text.split())
    return text

def _safe_str(val):
    """Safely converts potential NA or None values to stripped string."""
    if val is None or pd.isna(val):
        return ""
    return str(val).strip()

def preprocess_data(corpus):
    """
    Cleans each record from the Python copilot corpus, formats structured code & documentation,
    wraps it in a Document object, and splits it into manageable chunks.

    Args:
        corpus: Hugging Face dataset split containing the code records or passages.
    Returns:
        list[Document]: Splitted document chunks with metadata.
    """
    documents = []

    for idx, row in enumerate(corpus):
        # Support legacy passage schema if present
        if "passage" in row and row["passage"]:
            text = clean_text(row["passage"])
            doc = Document(
                page_content=text,
                metadata={"id": str(row.get("id", idx))}
            )
            documents.append(doc)
            continue

        # Multimodal Python Copilot schema
        file_path = _safe_str(row.get("file_path"))
        repo = _safe_str(row.get("repo"))
        name = _safe_str(row.get("name")) or _safe_str(row.get("class_name"))
        src_object = _safe_str(row.get("src_object")) or "code"
        method_names = _safe_str(row.get("method_names"))
        docstr = _safe_str(row.get("docstr"))
        code = _safe_str(row.get("code")).replace("__LOKCDR__", "\n")
        desc = _safe_str(row.get("desc"))

        # Extract answer / specification from description payload
        if "_ANSWER:" in desc:
            spec = desc.split("_ANSWER:", 1)[1].strip()
        else:
            spec = desc

        parts = [f"Repository: {repo} | File: {file_path} | {src_object.capitalize()}: {name}"]
        if method_names:
            parts.append(f"Methods: {method_names}")
        if docstr:
            parts.append(f"Docstring:\n{docstr}")
        if spec:
            parts.append(f"Architecture Specification & Usage:\n{spec}")
        if code:
            parts.append(f"Implementation Code:\n{code}")

        full_content = "\n\n".join(parts)
        doc_id = f"{file_path}::{name}" if file_path and name else str(row.get("id", idx))

        doc = Document(
            page_content=full_content,
            metadata={
                "id": doc_id,
                "file_path": file_path,
                "name": name,
                "repo": repo,
                "src_object": src_object,
                "method_names": method_names
            }
        )
        documents.append(doc)

    print(f"Number of Documents: {len(documents)}")

    # Split long documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)
    print(f"Number of Chunks: {len(chunks)}")

    return chunks

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from python.load_data import load_data

    print("\nLoading and preprocessing dataset corpus...")
    corpus = load_data(config="andromeda", split="train")
    chunks = preprocess_data(corpus)

    if len(chunks) > 0:
        print("\nFirst Chunk Preview:\n")
        print(chunks[0].page_content[:300])
        print("\nFirst Chunk Metadata:")
        print(chunks[0].metadata)
    else:
        print("No chunks were created.")
