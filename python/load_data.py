import os
# Check if executing directly or as package
if __name__ == "__main__" and __package__ is None:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

def load_data():
    """
    Loads the BioASQ corpus dataset.
    Returns:
        Dataset: Hugging Face dataset split for the text-corpus (passages).
    """
    print("Loading rag-datasets/rag-mini-bioasq (text-corpus)...")
    corpus = load_dataset(
        "rag-datasets/rag-mini-bioasq",
        "text-corpus",
        #ONLY 10000 PASSAGES TO REDUCE TIME
        split="passages"#[:10000]
    )
    return corpus

if __name__ == "__main__":
    corpus = load_data()
    print("Columns:", corpus.column_names)
    print("First Record:", corpus[0])
