from endee import Endee
from typing import List, Dict, Any

class EndeeStore:
    def __init__(self, collection_name: str = "doc_chunks"):
        """Initializes Endee client and index connection."""
        self.collection_name = collection_name
        self.client = Endee()
        
        # 1. Attempt to create index/collection if method exists
        for create_fn in ["create_index", "create_collection", "create"]:
            if hasattr(self.client, create_fn):
                try:
                    getattr(self.client, create_fn)(
                        name=self.collection_name,
                        dimension=384,
                        space_type="cosine"
                    )
                except Exception:
                    pass
                break

        # 2. Retrieve index reference if separate from client
        self.index = None
        for get_fn in ["get_index", "index", "get_collection", "collection"]:
            if hasattr(self.client, get_fn):
                try:
                    self.index = getattr(self.client, get_fn)(self.collection_name)
                    break
                except Exception:
                    pass
        
        if self.index is None:
            self.index = self.client

    def add_documents(self, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        """Upsert document vector embeddings and metadata into Endee."""
        items = [
            {"id": doc_id, "vector": vec, "meta": payload}
            for doc_id, vec, payload in zip(ids, vectors, payloads)
        ]
        
        targets = [self.index, self.client]
        method_names = ["upsert", "insert", "add", "add_documents", "upsert_vectors"]
        
        for target in targets:
            if target is None:
                continue
            for method_name in method_names:
                if hasattr(target, method_name):
                    method = getattr(target, method_name)
                    try:
                        method(items)
                        return
                    except TypeError:
                        try:
                            method(ids=ids, vectors=vectors, payloads=payloads)
                            return
                        except Exception:
                            pass
                    except Exception:
                        pass

        # Print available methods if no matching API signature was found
        available = [m for m in dir(self.client) if not m.startswith("_")]
        raise AttributeError(f"Could not find an insertion method on Endee client. Available methods: {available}")

    def query_similar(self, query_vector: List[float], top_k: int = 3):
        """Query Endee for top matching document vectors."""
        targets = [self.index, self.client]
        method_names = ["query", "search", "query_vectors", "similarity_search"]
        
        for target in targets:
            if target is None:
                continue
            for method_name in method_names:
                if hasattr(target, method_name):
                    method = getattr(target, method_name)
                    try:
                        return method(vector=query_vector, top_k=top_k)
                    except TypeError:
                        try:
                            return method(vector=query_vector, limit=top_k)
                        except TypeError:
                            try:
                                return method(query_vector, top_k)
                            except Exception:
                                pass
                    except Exception:
                        pass
        return []