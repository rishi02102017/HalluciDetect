"""Rule-based scoring for hallucination detection."""
import re
from typing import Dict, Any, List, Optional

class RuleBasedScorer:
    """Rule-based scoring system for detecting hallucinations."""
    
    def __init__(self):
        self.suspicious_patterns = [
            # Overly confident statements without evidence
            r'\b(always|never|all|every|none|no one)\b',
            # Vague qualifiers that might indicate uncertainty
            r'\b(probably|maybe|perhaps|might|could|possibly)\b',
            # Contradictory statements
            r'\b(but|however|although|despite|yet)\b.*\b(but|however|although|despite|yet)\b',
            # Numbers that seem made up
            r'\b\d{1,2}(?:,\d{3})*(?:\.\d+)?%\b',  # Percentages
            # Dates that might be fabricated
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        ]
        
        self.positive_indicators = [
            # Citations or references
            r'\[.*?\]',  # Bracketed citations
            r'\(.*?\d{4}.*?\)',  # Parenthetical citations with years
            r'according to|as stated in|as reported by|source:|reference:',
            # Specific details
            r'\b\d{4}\b',  # Years
            r'\b(?:study|research|paper|article|report)\b',
        ]
    
    def score(self, text: str, reference_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Score text for potential hallucinations using rule-based methods.
        
        Args:
            text: Text to score
            reference_text: Optional reference text for comparison
            
        Returns:
            Dictionary with rule-based scores and details
        """
        scores = {
            "suspicious_patterns": [],
            "positive_indicators": [],
            "suspicious_score": 0.0,
            "positive_score": 0.0,
            "overall_score": 0.0,
            "details": {}
        }
        
        # Check for suspicious patterns
        suspicious_count = 0
        for pattern in self.suspicious_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                suspicious_count += len(matches)
                scores["suspicious_patterns"].append({
                    "pattern": pattern,
                    "matches": matches,
                    "count": len(matches)
                })
        
        # Check for positive indicators
        positive_count = 0
        for pattern in self.positive_indicators:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                positive_count += len(matches)
                scores["positive_indicators"].append({
                    "pattern": pattern,
                    "matches": matches,
                    "count": len(matches)
                })
        
        # Normalize scores
        text_length = len(text.split())
        if text_length > 0:
            scores["suspicious_score"] = min(suspicious_count / (text_length / 10), 1.0)
            scores["positive_score"] = min(positive_count / (text_length / 20), 1.0)
        
        # Overall score: higher suspicious = more likely hallucination
        # Lower positive = more likely hallucination
        # Combine: (suspicious_score - (1 - positive_score)) / 2, normalized to 0-1
        scores["overall_score"] = max(0.0, min(1.0, 
            (scores["suspicious_score"] + (1 - scores["positive_score"])) / 2
        ))
        
        # Additional checks
        scores["details"] = {
            "word_count": text_length,
            "sentence_count": len(re.split(r'[.!?]+', text)),
            "has_citations": bool(re.search(r'\[.*?\]|\(.*?\d{4}.*?\)', text)),
            "has_specific_numbers": len(re.findall(r'\b\d+\b', text)),
            "has_dates": len(re.findall(r'\b\d{4}\b', text)),
        }
        
        # Check consistency if reference text provided
        if reference_text:
            consistency = self._check_consistency(text, reference_text)
            scores["consistency_score"] = consistency
            scores["overall_score"] = (scores["overall_score"] + (1 - consistency)) / 2
        
        return scores
    
    def _check_consistency(self, text: str, reference_text: str) -> float:
        """Check consistency between text and reference."""
        # Extract key entities (numbers, dates, names)
        text_entities = self._extract_entities(text)
        ref_entities = self._extract_entities(reference_text)
        
        if not text_entities and not ref_entities:
            return 1.0  # No entities to compare
        
        # Calculate overlap
        text_set = set(text_entities)
        ref_set = set(ref_entities)
        
        if not text_set:
            return 0.5  # No entities in text
        
        overlap = len(text_set & ref_set)
        total = len(text_set)
        
        return overlap / total if total > 0 else 0.0
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities from text."""
        entities = []
        
        # Extract numbers
        entities.extend(re.findall(r'\b\d+(?:\.\d+)?\b', text))
        
        # Extract years
        entities.extend(re.findall(r'\b(19|20)\d{2}\b', text))
        
        # Extract percentages
        entities.extend(re.findall(r'\b\d+(?:\.\d+)?%', text))
        
        return entities

