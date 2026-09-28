from endee import Endee
from typing import List, Dict, Any

class EndeeStore:
    def __init__(self, collection_name: str = "doc_chunks"):
        """Initializes the Endee client and sets up the index target."""
        self.collection_name = collection_name
        self.client = Endee()
        
        # Create index if supported by current SDK version
        if hasattr(self.client, "create_index"):
            try:
                self.client.create_index(
                    name=self.collection_name,
                    dimension=384,
                    space_type="cosine"
                )
            except Exception:
                pass  # Index already exists or auto-managed

        # Resolve index object reference safely
        if hasattr(self.client, "index"):
            self.index = self.client.index(self.collection_name)
        elif hasattr(self.client, "get_index"):
            self.index = self.client.get_index(self.collection_name)
        else:
            self.index = self.client

    def add_documents(self, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        """Upsert document vector embeddings and metadata into Endee."""
        items = [
            {
                "id": doc_id,
                "vector": vec,
                "meta": payload
            }
            for doc_id, vec, payload in zip(ids, vectors, payloads)
        ]
        
        target = self.index if hasattr(self.index, "upsert") else self.client

        try:
            target.upsert(items)
        except TypeError:
            # Alternate parameter signature fallback
            target.upsert(ids=ids, vectors=vectors, payloads=payloads)

    def query_similar(self, query_vector: List[float], top_k: int = 3):
        """Query Endee for top matching document vectors."""
        target = self.index if (hasattr(self.index, "query") or hasattr(self.index, "search")) else self.client

        if hasattr(target, "query"):
            try:
                return target.query(vector=query_vector, top_k=top_k)
            except TypeError:
                return target.query(vector=query_vector, limit=top_k)
        elif hasattr(target, "search"):
            try:
                return target.search(vector=query_vector, limit=top_k)
            except TypeError:
                return target.search(vector=query_vector, top_k=top_k)
        return []