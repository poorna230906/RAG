import sys
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Clean the text
def clean_text(text):
    if not text:
        return ""
    text = text.strip()                 # Remove spaces at the beginning and end
    text = " ".join(text.split())       # Remove extra spaces and newlines
    return text

# Preprocess the dataset
def preprocess_data(corpus):
    """
    Cleans each passage from the corpus, wraps it in a Document object, and splits it into chunks.
    Args:
        corpus: Hugging Face dataset split containing the passages.
    Returns:
        list[Document]: Splitted document chunks.
    """
    documents = []

    # Clean each passage and convert it to a Document
    for row in corpus:
        text = clean_text(row["passage"])
        doc = Document(
            page_content=text,
            metadata={"id": row["id"]}
        )
        documents.append(doc)

    print("Number of Documents:", len(documents))

    # Split long documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=384,
        chunk_overlap=75
    )

    chunks = splitter.split_documents(documents)
    print("Number of Chunks:", len(chunks))

    return chunks

# # Test the preprocessing
# if __name__ == "__main__":
#     # Add parent directory to path so we can import python.load_data when run directly
#     sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#     from python.load_data import load_data
# 
#     sample_text = """
#        i am poorna chandra , living in bangalore.
# 
#         i am a student.
#     """
# 
#     print("Testing clean_text()")
#     print("Original Text:", repr(sample_text))
#     print("Processed Text:", repr(clean_text(sample_text)))
# 
#     print("\nLoading and preprocessing dataset corpus...")
#     corpus = load_data()
#     chunks = preprocess_data(corpus)
# 
#     if len(chunks) > 0:
#         print("\nFirst Chunk:\n")
#         print(chunks[0].page_content)
#         print("\nMetadata:")
#         print(chunks[0].metadata)
#     else:
#         print("No chunks were created.")
