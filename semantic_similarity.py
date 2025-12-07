"""Semantic similarity checking module with lightweight fallback."""
import os
import numpy as np
from typing import Dict, Any, List, Optional
from config import Config

# Try to import sentence_transformers, but don't fail if not available
SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    # Only import if explicitly enabled (for local development)
    if os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true":
        from sentence_transformers import SentenceTransformer
        SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


class SemanticSimilarityChecker:
    """
    Checks semantic similarity between LLM output and reference text.
    
    Uses lightweight TF-IDF similarity by default (works on free hosting tiers).
    Can optionally use sentence-transformers for better accuracy (requires more RAM).
    """
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or Config.DEFAULT_EMBEDDING_MODEL
        self.threshold = Config.SEMANTIC_SIMILARITY_THRESHOLD
        self.use_local_model = SENTENCE_TRANSFORMERS_AVAILABLE and os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
        self._local_model = None
        
        # TF-IDF vectorizer for lightweight similarity
        self._tfidf_vectorizer = None
    
    def _get_local_model(self):
        """Lazy load the local sentence transformer model."""
        if self._local_model is None and self.use_local_model:
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer(self.model_name)
        return self._local_model
    
    def _compute_tfidf_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity using TF-IDF (lightweight, no heavy dependencies).
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Create or reuse vectorizer
        if self._tfidf_vectorizer is None:
            self._tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=5000
            )
        
        try:
            # Fit and transform both texts
            tfidf_matrix = self._tfidf_vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception:
            # Fallback to simple word overlap
            return self._compute_word_overlap(text1, text2)
    
    def _compute_word_overlap(self, text1: str, text2: str) -> float:
        """
        Simple word overlap similarity (Jaccard-like).
        Used as ultimate fallback.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _compute_local_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity using local sentence-transformers model."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        model = self._get_local_model()
        embedding1 = model.encode(text1, convert_to_numpy=True)
        embedding2 = model.encode(text2, convert_to_numpy=True)
        
        similarity_score = cosine_similarity(
            embedding1.reshape(1, -1),
            embedding2.reshape(1, -1)
        )[0][0]
        
        return float(similarity_score)
    
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
            return_embeddings: Whether to return embedding vectors (only with local model)
            
        Returns:
            Dictionary with similarity scores and details
        """
        method_used = "tfidf"
        
        try:
            if self.use_local_model:
                similarity_score = self._compute_local_similarity(text1, text2)
                method_used = "sentence_transformers"
            else:
                similarity_score = self._compute_tfidf_similarity(text1, text2)
        except Exception as e:
            # Ultimate fallback
            similarity_score = self._compute_word_overlap(text1, text2)
            method_used = "word_overlap"
        
        result = {
            "similarity_score": similarity_score,
            "is_similar": similarity_score >= self.threshold,
            "threshold": self.threshold,
            "method": method_used
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
        
        results = []
        for i, (t1, t2) in enumerate(zip(texts1, texts2)):
            sim_result = self.compute_similarity(t1, t2)
            sim_result["text1"] = t1
            sim_result["text2"] = t2
            results.append(sim_result)
        
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
        # Compute similarities for all candidates
        similarities = []
        for candidate in candidate_texts:
            result = self.compute_similarity(query_text, candidate)
            similarities.append(result["similarity_score"])
        
        # Get top k indices
        similarities = np.array(similarities)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "text": candidate_texts[idx],
                "similarity_score": float(similarities[idx]),
                "rank": len(results) + 1
            })
        
        return results
