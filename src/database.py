from endee import Endee
from typing import List, Dict, Any

class EndeeStore:
    def __init__(self, collection_name: str = "doc_chunks"):
        """Initializes Endee client and creates the index."""
        self.client = Endee()
        self.collection_name = collection_name
        
        # Create vector index if it doesn't exist
        try:
            self.client.create_index(
                name=self.collection_name,
                dimension=384,
                space_type="cosine"
            )
        except Exception:
            pass  # Index already exists
            
        self.index = self.client.get_index(self.collection_name)

    def add_documents(self, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        """Upsert document vector embeddings into Endee index."""
        items = [
            {
                "id": doc_id,
                "vector": vec,
                "meta": payload
            }
            for doc_id, vec, payload in zip(ids, vectors, payloads)
        ]
        self.index.upsert(items)

    def query_similar(self, query_vector: List[float], top_k: int = 3):
        """Query Endee index for top matching document vectors."""
        return self.index.query(
            vector=query_vector,
            top_k=top_k
        )