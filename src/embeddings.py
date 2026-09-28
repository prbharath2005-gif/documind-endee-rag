from sentence_transformers import SentenceTransformer

class EmbeddingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Loads a lightweight sentence-transformer model.
        Outputs 384-dimensional dense vector embeddings.
        """
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        """Converts a list of text chunks into vector embeddings."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def encode_query(self, query: str) -> list[float]:
        """Converts a single user query string into a vector embedding."""
        return self.model.encode(query).tolist()