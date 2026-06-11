import json
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

def main():
    json_path = "space_books.json"
    db_path = "qdrant_db"
    collection_name = "space_books"
    model_name = "all-MiniLM-L6-v2"

    print("Loading space_books.json...")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find {json_path}")
        
    with open(json_path, "r", encoding="utf-8") as f:
        books = json.load(f)
    print(f"Loaded {len(books)} stories.")

    print(f"Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)
    vector_size = 384  # dimension of all-MiniLM-L6-v2 embeddings
    print("Model loaded successfully.")

    print(f"Initializing local Qdrant client at path: '{db_path}'...")
    client = QdrantClient(path=db_path)

    # Recreate the collection (delete if exists, then create)
    print(f"Setting up collection '{collection_name}'...")
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"Collection '{collection_name}' created.")

    print("Generating embeddings and upserting points in batches...")
    batch_size = 100
    for i in range(0, len(books), batch_size):
        batch = books[i : i + batch_size]
        # We embed the combination of Title and Content for richer semantics
        texts = [f"Title: {b.get('title', '')}\nContent: {b.get('content', '')}" for b in batch]
        embeddings = model.encode(texts).tolist()

        points = []
        for j, b in enumerate(batch):
            point_id = b.get("id")
            if point_id is None:
                point_id = i + j
                
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embeddings[j],
                    payload={
                        "title": b.get("title", ""),
                        "content": b.get("content", ""),
                        "book_id": point_id,
                    },
                )
            )

        client.upsert(collection_name=collection_name, points=points)
        print(f"Upserted batch {i // batch_size + 1}/{(len(books) - 1) // batch_size + 1} ({len(points)} stories)...")

    print("\nAll space stories have been successfully embedded and saved to the Qdrant database.")

    # Perform a quick test query to verify everything works!
    test_query = "NASA mission studying water on a moon"
    print(f"\n--- Performing Verification Query: '{test_query}' ---")
    query_vector = model.encode(test_query).tolist()
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=3
    )

    for idx, result in enumerate(response.points):
        print(f"\nResult #{idx+1} (Score: {result.score:.4f}):")
        print(f"Title: {result.payload.get('title')}")
        print(f"Content: {result.payload.get('content')}")

if __name__ == "__main__":
    main()
