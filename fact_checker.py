"""Fact-checking module using external APIs and rule-based checks."""
import requests
import re
from typing import Dict, Any, List, Tuple, Optional
from config import Config

class FactChecker:
    """Checks factual claims in LLM outputs."""
    
    def __init__(self):
        self.api_key = Config.FACTCHECK_API_KEY
        self.api_url = Config.FACTCHECK_API_URL
        self.confidence_threshold = Config.FACT_CHECK_CONFIDENCE_THRESHOLD
    
    def check_facts(self, text: str, reference_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Check facts in the given text.
        
        Args:
            text: Text to fact-check
            reference_text: Optional reference text for comparison
            
        Returns:
            Dictionary with fact-check results
        """
        # Extract factual claims from text
        claims = self._extract_claims(text)
        
        # Check each claim
        checked_claims = []
        total_claims = len(claims)
        verified_claims = 0
        disputed_claims = 0
        
        for claim in claims:
            claim_result = self._check_single_claim(claim, reference_text)
            checked_claims.append(claim_result)
            
            if claim_result.get("verified", False):
                verified_claims += 1
            elif claim_result.get("disputed", False):
                disputed_claims += 1
        
        # Calculate overall score
        if total_claims > 0:
            fact_check_score = verified_claims / total_claims
        else:
            fact_check_score = 1.0  # No claims to check = perfect score
        
        return {
            "score": fact_check_score,
            "total_claims": total_claims,
            "verified_claims": verified_claims,
            "disputed_claims": disputed_claims,
            "unverified_claims": total_claims - verified_claims - disputed_claims,
            "claims": checked_claims,
            "confidence": min(fact_check_score, 1.0)
        }
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text."""
        # Simple heuristic: extract sentences that contain numbers, dates, or specific facts
        sentences = re.split(r'[.!?]+', text)
        claims = []
        
        # Patterns that suggest factual claims
        fact_patterns = [
            r'\d{4}',  # Years
            r'\d+%',   # Percentages
            r'\$\d+',  # Money
            r'\d+\.\d+',  # Decimals
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)',  # Months
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Check if sentence contains factual indicators
            if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in fact_patterns):
                claims.append(sentence)
            # Also include sentences with specific factual keywords
            elif any(keyword in sentence.lower() for keyword in [
                'according to', 'research shows', 'study found', 'data indicates',
                'statistics', 'reported', 'found that', 'discovered'
            ]):
                claims.append(sentence)
        
        return claims[:10]  # Limit to top 10 claims
    
    def _check_single_claim(self, claim: str, reference_text: Optional[str] = None) -> Dict[str, Any]:
        """Check a single factual claim."""
        result = {
            "claim": claim,
            "verified": False,
            "disputed": False,
            "confidence": 0.0,
            "source": None,
            "method": "rule_based"
        }
        
        # If API is configured, try to use it
        if self.api_key and self.api_url:
            try:
                api_result = self._check_via_api(claim)
                if api_result:
                    return api_result
            except Exception as e:
                # Fall back to rule-based checking
                pass
        
        # Rule-based checking
        if reference_text:
            # Check if claim appears in reference text (simple keyword matching)
            claim_keywords = set(claim.lower().split())
            ref_keywords = set(reference_text.lower().split())
            
            # Calculate overlap
            overlap = len(claim_keywords & ref_keywords)
            total_unique = len(claim_keywords | ref_keywords)
            
            if total_unique > 0:
                similarity = overlap / total_unique
                result["confidence"] = similarity
                result["verified"] = similarity > self.confidence_threshold
        
        # Default: mark as unverified if no reference
        if not result["verified"] and not result["disputed"]:
            result["confidence"] = 0.5  # Neutral confidence
        
        return result
    
    def _check_via_api(self, claim: str) -> Optional[Dict[str, Any]]:
        """Check claim using external fact-check API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "claim": claim,
                "language": "en"
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "claim": claim,
                    "verified": data.get("verified", False),
                    "disputed": data.get("disputed", False),
                    "confidence": data.get("confidence", 0.0),
                    "source": data.get("source"),
                    "method": "api"
                }
        except Exception as e:
            # API call failed, return None to fall back to rule-based
            pass
        
        return None

