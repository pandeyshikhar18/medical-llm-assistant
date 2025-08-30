import os
import numpy as np
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# ---------------------------
# Helper / Retriever adapter
# ---------------------------
def _retrieve(retriever, query):
    """Unified retriever interface: prefer get_relevant_documents, then callable, then retrieve()."""
    if retriever is None:
        return []
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    try:
        # some retrievers are callables
        return retriever(query)
    except Exception:
        if hasattr(retriever, "retrieve"):
            return retriever.retrieve(query)
    return []

# ---------------------------
# Metrics
# ---------------------------
def recall_at_k(results, relevant, k):
    return 1 if any(os.path.basename(r.metadata.get("source", "")) == relevant for r in results[:k]) else 0

def ndcg_at_k(results, relevant, k):
    for i, r in enumerate(results[:k]):
        if os.path.basename(r.metadata.get("source", "")) == relevant:
            return 1 / np.log2(i + 2)
    return 0
                

# ---------------------------
# Evaluation
# ---------------------------
def evaluate(vectorstore, retriever, dataset, k=10):
    """
    Evaluate retrieval on `dataset` (list of {"query":..., "relevant": "<filename>"}).
    Returns a dict with recall_at_k and ndcg_at_k (averages).
    """
    recalls, ndcgs = [], []
    for item in dataset:
        results = _retrieve(retriever, item["query"]) or []
        recalls.append(recall_at_k(results, item["relevant"], k))
        ndcgs.append(ndcg_at_k(results, item["relevant"], k))

    out = {
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "k": k,
        "queries_evaluated": len(dataset),
    }

    # print a compact summary for CLI usage
    print("📊 Evaluation Results")
    print(f"Recall@{k}: {out['recall_at_k']:.4f}")
    print(f"NDCG@{k}:   {out['ndcg_at_k']:.4f}")
    return out

# ---------------------------
# Example Usage (script)
# ---------------------------
if __name__ == "__main__":
    load_dotenv()

    eval_data = [
        {"query": "What are the side effects of COVID-19 vaccines?", "relevant": "vaccines.pdf"},
        {"query": "Summarize findings on COVID-19 treatments", "relevant": "covid19.pdf"},
    ]

    # Load documents from data/ (adjust paths if needed)
    all_docs = []
    for file in ["data/vaccines.pdf", "data/covid19.pdf"]:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Expected file not found: {file}")
        loader = PyPDFLoader(file)
        all_docs.extend(loader.load())

    # Split
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    # Embeddings & vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    # Run evaluation and print/return results
    metrics = evaluate(vectorstore, retriever, eval_data, k=10)
    print(metrics)
