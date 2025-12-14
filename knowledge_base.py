"""Custom Knowledge Base for fact verification."""
import os
import json
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config import Config


class KnowledgeBase:
    """
    Custom knowledge base for storing and retrieving verified facts.
    
    Supports:
    - JSON file-based storage
    - TF-IDF based semantic search
    - Multiple knowledge domains
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the knowledge base.
        
        Args:
            base_path: Path to knowledge base directory
        """
        self.base_path = base_path or Config.KNOWLEDGE_BASE_PATH
        self.facts: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = None
        self._ensure_directory()
        self._load_all_facts()
    
    def _ensure_directory(self):
        """Ensure knowledge base directory exists."""
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
            # Create a sample knowledge base file
            self._create_sample_kb()
    
    def _create_sample_kb(self):
        """Create a sample knowledge base file."""
        sample_facts = {
            "domain": "general",
            "version": "1.0",
            "facts": [
                {
                    "id": "1",
                    "statement": "The Earth orbits around the Sun.",
                    "category": "science",
                    "verified": True,
                    "source": "NASA"
                },
                {
                    "id": "2",
                    "statement": "Water boils at 100 degrees Celsius at sea level.",
                    "category": "science",
                    "verified": True,
                    "source": "Physics textbook"
                },
                {
                    "id": "3",
                    "statement": "Python is a programming language created by Guido van Rossum.",
                    "category": "technology",
                    "verified": True,
                    "source": "Python.org"
                }
            ]
        }
        
        sample_path = os.path.join(self.base_path, "sample_kb.json")
        with open(sample_path, 'w') as f:
            json.dump(sample_facts, f, indent=2)
    
    def _load_all_facts(self):
        """Load all facts from JSON files in the knowledge base directory."""
        self.facts = []
        
        if not os.path.exists(self.base_path):
            return
        
        for filename in os.listdir(self.base_path):
            if filename.endswith('.json'):
                filepath = os.path.join(self.base_path, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        domain = data.get('domain', 'unknown')
                        for fact in data.get('facts', []):
                            fact['domain'] = domain
                            fact['source_file'] = filename
                            self.facts.append(fact)
                except (json.JSONDecodeError, IOError):
                    continue
        
        # Build TF-IDF matrix if we have facts
        if self.facts:
            self._build_index()
    
    def _build_index(self):
        """Build TF-IDF index for semantic search."""
        if not self.facts:
            return
        
        statements = [fact.get('statement', '') for fact in self.facts]
        self.tfidf_matrix = self.vectorizer.fit_transform(statements)
    
    def search(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search for relevant facts using TF-IDF similarity.
        
        Args:
            query: The search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of matching facts with similarity scores
        """
        if not self.facts or self.tfidf_matrix is None:
            return []
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top results above threshold
        results = []
        indices = np.argsort(similarities)[::-1]
        
        for idx in indices[:top_k]:
            if similarities[idx] >= threshold:
                fact = self.facts[idx].copy()
                fact['similarity'] = float(similarities[idx])
                results.append(fact)
        
        return results
    
    def verify_claim(self, claim: str) -> Dict[str, Any]:
        """
        Verify a claim against the knowledge base.
        
        Args:
            claim: The claim to verify
            
        Returns:
            Verification result with confidence score
        """
        matches = self.search(claim, top_k=3, threshold=0.4)
        
        if not matches:
            return {
                "claim": claim,
                "verified": False,
                "confidence": 0.0,
                "message": "No matching facts found in knowledge base",
                "matches": []
            }
        
        # Calculate overall verification score
        top_match = matches[0]
        confidence = top_match.get('similarity', 0.0)
        
        # Check if the top match supports the claim
        if top_match.get('verified', False):
            return {
                "claim": claim,
                "verified": True,
                "confidence": confidence,
                "message": f"Claim supported by knowledge base",
                "source": top_match.get('source', 'Unknown'),
                "matched_fact": top_match.get('statement', ''),
                "matches": matches
            }
        else:
            return {
                "claim": claim,
                "verified": False,
                "confidence": confidence,
                "message": "Matched fact is not verified",
                "matches": matches
            }
    
    def add_fact(self, statement: str, category: str = "custom", 
                 source: str = "user", domain: str = "custom",
                 verified: bool = True) -> Dict[str, Any]:
        """
        Add a new fact to the knowledge base.
        
        Args:
            statement: The factual statement
            category: Category of the fact
            source: Source of the fact
            domain: Knowledge domain
            verified: Whether the fact is verified
            
        Returns:
            The added fact
        """
        import uuid
        
        fact = {
            "id": str(uuid.uuid4()),
            "statement": statement,
            "category": category,
            "source": source,
            "domain": domain,
            "verified": verified
        }
        
        # Add to memory
        self.facts.append(fact)
        
        # Save to file
        self._save_custom_facts()
        
        # Rebuild index
        self._build_index()
        
        return fact
    
    def _save_custom_facts(self):
        """Save custom (user-added) facts to file."""
        custom_facts = [f for f in self.facts if f.get('domain') == 'custom']
        
        if not custom_facts:
            return
        
        data = {
            "domain": "custom",
            "version": "1.0",
            "facts": custom_facts
        }
        
        filepath = os.path.join(self.base_path, "custom_facts.json")
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        domains = {}
        categories = {}
        
        for fact in self.facts:
            domain = fact.get('domain', 'unknown')
            category = fact.get('category', 'unknown')
            
            domains[domain] = domains.get(domain, 0) + 1
            categories[category] = categories.get(category, 0) + 1
        
        return {
            "total_facts": len(self.facts),
            "domains": domains,
            "categories": categories,
            "indexed": self.tfidf_matrix is not None
        }
    
    def reload(self):
        """Reload all facts from disk."""
        self._load_all_facts()


# Singleton instance
_knowledge_base = None

def get_knowledge_base() -> KnowledgeBase:
    """Get or create KnowledgeBase instance."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase()
    return _knowledge_base

