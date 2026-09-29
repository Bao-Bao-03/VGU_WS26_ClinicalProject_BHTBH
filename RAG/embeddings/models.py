from typing import List
import numpy as np

class EmbeddingModel:
    # define an interface for embedding models
    def encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

class BioLinkerBERTEmbedder(EmbeddingModel):
    #  we use BioLinkerBERT model (recommended for medical text)
    def __init__(self):
        from sentence_transformers import SentenceTransformer
      
        self.model = SentenceTransformer('dmis-lab/biobert-base-cased-v1.1')
        self.dim = 768

  # converts text to vectors
    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

class OpenAIEmbedder(EmbeddingModel):
    # define OpenAI's embedding model
    def __init__(self, api_key: str):
        import openai
        openai.api_key = api_key
        self.model = "text-embedding-3-small"
        self.dim = 1536

    def encode(self, texts: List[str]) -> np.ndarray:
        import openai
        result = openai.Embedding.create(
            input=texts, model=self.model
        )
        return np.array([item['embedding'] for item in result['data']])
