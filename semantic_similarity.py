"""Semantic similarity checking module."""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, Any, List, Optional
from config import Config

class SemanticSimilarityChecker:
    """Checks semantic similarity between LLM output and reference text."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or Config.DEFAULT_EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
        self.threshold = Config.SEMANTIC_SIMILARITY_THRESHOLD
    
    def compute_similarity(
        self,
        text1: str,
        text2: str,
        return_embeddings: bool = False
    ) -> Dict[str, Any]:
        """
        Compute semantic similarity between two texts.
        
        Args:
            text1: First text (usually LLM output)
            text2: Second text (usually reference/ground truth)
            return_embeddings: Whether to return embedding vectors
            
        Returns:
            Dictionary with similarity scores and details
        """
        # Generate embeddings
        embedding1 = self.model.encode(text1, convert_to_numpy=True)
        embedding2 = self.model.encode(text2, convert_to_numpy=True)
        
        # Compute cosine similarity
        similarity_score = cosine_similarity(
            embedding1.reshape(1, -1),
            embedding2.reshape(1, -1)
        )[0][0]
        
        result = {
            "similarity_score": float(similarity_score),
            "is_similar": similarity_score >= self.threshold,
            "threshold": self.threshold
        }
        
        if return_embeddings:
            result["embeddings"] = {
                "text1": embedding1.tolist(),
                "text2": embedding2.tolist()
            }
        
        return result
    
    def compute_similarity_batch(
        self,
        texts1: List[str],
        texts2: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Compute similarity for multiple text pairs.
        
        Args:
            texts1: List of first texts
            texts2: List of second texts (must match length of texts1)
            
        Returns:
            List of similarity results
        """
        if len(texts1) != len(texts2):
            raise ValueError("texts1 and texts2 must have the same length")
        
        # Batch encode for efficiency
        embeddings1 = self.model.encode(texts1, convert_to_numpy=True)
        embeddings2 = self.model.encode(texts2, convert_to_numpy=True)
        
        # Compute similarities
        similarities = cosine_similarity(embeddings1, embeddings2)
        
        results = []
        for i, similarity_score in enumerate(similarities.diagonal()):
            results.append({
                "similarity_score": float(similarity_score),
                "is_similar": similarity_score >= self.threshold,
                "threshold": self.threshold,
                "text1": texts1[i],
                "text2": texts2[i]
            })
        
        return results
    
    def find_most_similar(
        self,
        query_text: str,
        candidate_texts: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find most similar texts to a query text.
        
        Args:
            query_text: Text to find similarities for
            candidate_texts: List of candidate texts
            top_k: Number of top results to return
            
        Returns:
            List of results sorted by similarity (highest first)
        """
        query_embedding = self.model.encode(query_text, convert_to_numpy=True)
        candidate_embeddings = self.model.encode(candidate_texts, convert_to_numpy=True)
        
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            candidate_embeddings
        )[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "text": candidate_texts[idx],
                "similarity_score": float(similarities[idx]),
                "rank": len(results) + 1
            })
        
        return results

