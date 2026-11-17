import os
import sys

# SSL bypass for environments with custom certificates / proxies
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

try:
    import httpx
    original_client_init = httpx.Client.__init__
    def patched_client_init(self, *args, **kwargs):
        kwargs['verify'] = False
        original_client_init(self, *args, **kwargs)
    httpx.Client.__init__ = patched_client_init

    original_async_client_init = httpx.AsyncClient.__init__
    def patched_async_client_init(self, *args, **kwargs):
        kwargs['verify'] = False
        original_async_client_init(self, *args, **kwargs)
    httpx.AsyncClient.__init__ = patched_async_client_init
except Exception:
    pass

try:
    import requests
    original_session_request = requests.Session.request
    def patched_session_request(self, method, url, **kwargs):
        kwargs['verify'] = False
        return original_session_request(self, method, url, **kwargs)
    requests.Session.request = patched_session_request
except Exception:
    pass

if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datasets import Dataset

# Reference Overview Hub:
# https://huggingface.co/datasets/matlok/multimodal-python-copilot-training-overview
# Underlying Parquet file repository curated by Matlok:
BASE_URL = "https://huggingface.co/datasets/matlok/python-text-copilot-training-instruct-ai-research-2024-02-03/resolve/main"

# Config mapping for individual AI research repository instruction splits
CONFIGS = {
    "andromeda": {
        "train": f"{BASE_URL}/train/train-0001-andromeda-andromeda_torch.parquet",
        "test": f"{BASE_URL}/test/train-0002-andromeda-tests.parquet",
    },
    "swarms": {
        "train": f"{BASE_URL}/train/train-0004-swarms-swarms.parquet",
        "test": f"{BASE_URL}/test/train-0005-swarms-tests.parquet",
    },
    "swarms_pytorch": {
        "train": f"{BASE_URL}/train/train-0006-swarms-pytorch-swarms_torch.parquet",
        "test": f"{BASE_URL}/test/train-0007-swarms-pytorch-tests.parquet",
    },
    "longnet": {
        "train": f"{BASE_URL}/train/train-0009-longnet-long_net.parquet",
        "test": f"{BASE_URL}/test/train-0010-longnet-tests.parquet",
    },
    "zeta": {
        "train": f"{BASE_URL}/train/train-0011-zeta-zeta.parquet",
        "test": f"{BASE_URL}/test/train-0012-zeta-tests.parquet",
    },
}

DATASET_NAME = "matlok/multimodal-python-copilot-training-overview"

def load_data(config="andromeda", split="train"):
    """
    Loads Python Copilot training data indexed under:
    https://huggingface.co/datasets/matlok/multimodal-python-copilot-training-overview

    Args:
        config (str): Config name ('andromeda', 'swarms', 'swarms_pytorch', 'longnet', 'zeta', or 'all').
                      Defaults to 'andromeda' for fast and high-quality local RAG indexing.
        split (str): 'train' (training corpus) or 'test' (evaluation suite).

    Returns:
        Dataset: Hugging Face Dataset split containing Python copilot source code,
                 architecture specifications, class/function definitions, and metadata.
    """
    print(f"Loading dataset: {DATASET_NAME}")
    print(f"[Dataset] Config: '{config}', Split: '{split}'")

    if config == "all":
        selected_configs = list(CONFIGS.keys())
    elif config in CONFIGS:
        selected_configs = [config]
    else:
        raise ValueError(
            f"Unknown config '{config}'. Choose from: {list(CONFIGS.keys())} or 'all'"
        )

    dfs = []
    for cfg in selected_configs:
        file_url = CONFIGS[cfg].get(split)
        if not file_url:
            raise ValueError(f"Split '{split}' not found for config '{cfg}'.")
        print(f"[Dataset] Downloading shard for '{cfg}' ({split})...")
        df = pd.read_parquet(file_url)
        dfs.append(df)

    if len(dfs) == 1:
        combined_df = dfs[0]
    else:
        combined_df = pd.concat(dfs, ignore_index=True)

    dataset = Dataset.from_pandas(combined_df)
    print(f"[Dataset] Loaded {len(dataset)} records successfully.")
    return dataset

if __name__ == "__main__":
    corpus = load_data(config="andromeda", split="train")
    print("\nDataset loaded successfully!")
    print("Columns:", corpus.column_names)
    print("Number of records:", len(corpus))
    print("First Record Name:", corpus[0].get("name"))
    print("First Record File:", corpus[0].get("file_path"))
