import endee
from typing import List, Dict, Any

class EndeeStore:
    def __init__(self, collection_name: str = "doc_chunks"):
        """Initializes the Endee Vector Client and collection."""
        self.client = endee.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            dimension=384  # Matches sentence-transformers 384 dimensions
        )

    def add_documents(self, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        """Inserts or updates document chunks and vector embeddings into Endee."""
        self.collection.upsert(
            ids=ids,
            vectors=vectors,
            payloads=payloads
        )

    def query_similar(self, query_vector: List[float], top_k: int = 3):
        """Queries Endee for top matching document vectors."""
        return self.collection.search(
            vector=query_vector,
            limit=top_k
        )