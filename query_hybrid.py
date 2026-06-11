import json
import os
import re
import sys
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Paths
json_path = "space_books.json"
db_path = "qdrant_db"
collection_name = "space_books"

# 1. Load data
if not os.path.exists(json_path):
    print(f"Error: {json_path} not found. Please run embed_books.py first or ensure the file is in the workspace directory.")
    sys.exit(1)

with open(json_path, "r", encoding="utf-8") as f:
    books = json.load(f)

books_by_id = {b["id"]: b for b in books}

# Tokenization helper for BM25
def tokenize(text):
    return re.findall(r"\w+", text.lower())

# Build the tokenized corpus for BM25 indexing (Title + Content)
corpus = [
    tokenize(f"Title: {b.get('title', '')}\nContent: {b.get('content', '')}")
    for b in books
]
bm25 = BM25Okapi(corpus)

# 2. Initialize Clients
client = QdrantClient(path=db_path)
model = SentenceTransformer("all-MiniLM-L6-v2")

def hybrid_search(query, k=60, top_n=50):
    """
    Performs hybrid search on space books.
    
    1. Queries BM25 keyword index.
    2. Queries local Qdrant collection for semantic vector matches.
    3. Fuses results using Reciprocal Rank Fusion (RRF).
    """
    # 1. Keyword (BM25) search
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked = sorted(
        [(idx, score) for idx, score in enumerate(bm25_scores) if score > 0],
        key=lambda x: x[1],
        reverse=True,
    )

    # 2. Semantic (Qdrant) search
    query_vector = model.encode(query).tolist()
    
    try:
        semantic_response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_n,
        )
        semantic_points = semantic_response.points
    except Exception as e:
        print(f"Error querying Qdrant collection '{collection_name}': {e}")
        print("Please verify that embed_books.py has been run and the database is populated.")
        sys.exit(1)

    # 3. Reciprocal Rank Fusion (RRF)
    # Formula: RRF(d) = sum( 1 / (k + rank) )
    rrf_scores = {}
    bm25_ranks = {}
    semantic_ranks = {}

    # Rank BM25 results
    for rank, (doc_idx, score) in enumerate(bm25_ranked[:top_n], start=1):
        book_id = books[doc_idx]["id"]
        rrf_scores[book_id] = rrf_scores.get(book_id, 0.0) + (1.0 / (k + rank))
        bm25_ranks[book_id] = rank

    # Rank semantic results
    for rank, point in enumerate(semantic_points, start=1):
        book_id = point.id
        rrf_scores[book_id] = rrf_scores.get(book_id, 0.0) + (1.0 / (k + rank))
        semantic_ranks[book_id] = rank

    # Sort candidates by combined RRF score
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top_3 = sorted_rrf[:3]

    # Display results
    print(f"\n=========================================")
    print(f"Hybrid Search Query: '{query}'")
    print(f"=========================================")

    # Print BM25 Top 3
    print("\n--- Top 3 Keyword (BM25) Matches ---")
    if not bm25_ranked:
        print("No keyword matches found.")
    for rank, (doc_idx, score) in enumerate(bm25_ranked[:3], start=1):
        book = books[doc_idx]
        print(f"{rank}. [ID {book['id']}] {book['title']} (Score: {score:.2f})")

    # Print Semantic Top 3
    print("\n--- Top 3 Semantic (Qdrant) Matches ---")
    if not semantic_points:
        print("No semantic matches found.")
    for rank, point in enumerate(semantic_points[:3], start=1):
        print(f"{rank}. [ID {point.id}] {point.payload.get('title')} (Score: {point.score:.4f})")

    # Print Combined RRF Top 3
    print("\n--- Top 3 Combined (RRF Hybrid) Matches ---")
    if not top_3:
        print("No combined results found.")
    for idx, (book_id, rrf_score) in enumerate(top_3, start=1):
        book = books_by_id.get(book_id)
        if book:
            bm25_rank_str = f"Rank {bm25_ranks[book_id]}" if book_id in bm25_ranks else "N/A"
            semantic_rank_str = f"Rank {semantic_ranks[book_id]}" if book_id in semantic_ranks else "N/A"
            print(f"\n{idx}. [ID {book_id}] {book['title']} (RRF Score: {rrf_score:.5f})")
            print(f"   BM25 Rank: {bm25_rank_str} | Semantic Rank: {semantic_rank_str}")
            content = book.get("content", "")
            excerpt = content[:150] + "..." if len(content) > 150 else content
            print(f"   Excerpt: {excerpt}")

    return top_3

if __name__ == "__main__":
    # If a query is provided as command-line arguments, use it; otherwise fallback to a test query
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])
    else:
        test_query = "NASA mission studying water on a moon"
    
    hybrid_search(test_query)
