# Biomedical RAG Question Answering System

A Retrieval-Augmented Generation (RAG) system for biomedical question answering using the BioASQ dataset.The system retrieves relevant biomedical passages using FAISS and generates answers using Anthropic Claude.

# Features

1 Biomedical question answering using the BioASQ dataset
2 Text cleaning and preprocessing
3 Text chunking using Recursive Character Text Splitter
4 Sentence Transformer embeddings
5 FAISS vector database for semantic search
6 Similarity threshold-based retrieval
7 Anthropic Claude for answer generation
8 Context-only prompting to reduce hallucination
9 API retry mechanism
10 Fallback response for API failures
11 Dynamic grounding score
12 Recall@3 and Recall@5 evaluation
13 Retrieval error rate calculation
14 API error-rate tracking
15 Error mitigation score

## Architecture

    text
BioASQ Dataset
      |
      v
Load Passages
      |
      v
Text Preprocessing
      |
      v
Text Chunking
      |
      v
Sentence Transformer Embeddings
      |
      v
FAISS Vector Database
      |
      v
User Question
      |
      v
Semantic Retrieval
      |
      v
Similarity Threshold
      |
      v
Retrieved Context
      |
      v
Anthropic Claude
      |
      +------ API Error ------> Retry
      |                           |
      |                           v
      |                       Fallback
      |
      v
Generated Answer
      |
      v
Grounding Evaluation