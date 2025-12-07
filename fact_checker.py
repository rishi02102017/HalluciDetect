"""Fact-checking module using Wikipedia API, external APIs, and rule-based checks."""
import requests
import re
from typing import Dict, Any, List, Tuple, Optional
from config import Config

# Wikipedia API helper
try:
    import wikipediaapi
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False


class FactChecker:
    """Checks factual claims in LLM outputs using Wikipedia and rule-based methods."""
    
    def __init__(self):
        self.api_key = Config.FACTCHECK_API_KEY
        self.api_url = Config.FACTCHECK_API_URL
        self.confidence_threshold = Config.FACT_CHECK_CONFIDENCE_THRESHOLD
        
        # Initialize Wikipedia API
        self._wiki = None
        if WIKIPEDIA_AVAILABLE:
            self._wiki = wikipediaapi.Wikipedia(
                user_agent='HalluciDetect/1.0 (https://hallucidetect.onrender.com)',
                language='en'
            )
    
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
        
        # Extract entities for Wikipedia lookup
        entities = self._extract_entities(text)
        
        # Check each claim
        checked_claims = []
        total_claims = len(claims)
        verified_claims = 0
        disputed_claims = 0
        
        for claim in claims:
            claim_result = self._check_single_claim(claim, reference_text, entities)
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
            "confidence": min(fact_check_score, 1.0),
            "entities_found": entities[:5] if entities else []
        }
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities (proper nouns, locations, people) from text."""
        entities = []
        
        # Simple regex-based entity extraction
        # Capitalized words that aren't at the start of sentences
        words = text.split()
        for i, word in enumerate(words):
            # Skip first word of sentences
            if i > 0 and words[i-1][-1] in '.!?':
                continue
            
            # Look for capitalized words (potential entities)
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                entities.append(clean_word)
        
        # Also extract quoted terms
        quoted = re.findall(r'"([^"]+)"', text)
        entities.extend(quoted)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for e in entities:
            if e.lower() not in seen:
                seen.add(e.lower())
                unique_entities.append(e)
        
        return unique_entities[:10]  # Limit to 10 entities
    
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
            r'\d+ (?:million|billion|thousand)',  # Large numbers
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
                'statistics', 'reported', 'found that', 'discovered', 'established',
                'founded', 'invented', 'created', 'is the', 'are the', 'was the', 'were the'
            ]):
                claims.append(sentence)
        
        return claims[:10]  # Limit to top 10 claims
    
    def _check_single_claim(self, claim: str, reference_text: Optional[str] = None, 
                           entities: List[str] = None) -> Dict[str, Any]:
        """Check a single factual claim using multiple methods."""
        result = {
            "claim": claim,
            "verified": False,
            "disputed": False,
            "confidence": 0.0,
            "source": None,
            "method": "rule_based"
        }
        
        # Method 1: Check against reference text first (highest priority)
        if reference_text:
            ref_result = self._check_against_reference(claim, reference_text)
            if ref_result["confidence"] > 0.6:
                return ref_result
        
        # Method 2: Try Wikipedia API for entity verification
        if self._wiki and entities:
            wiki_result = self._check_via_wikipedia(claim, entities)
            if wiki_result and wiki_result["confidence"] > 0.5:
                return wiki_result
        
        # Method 3: Try external fact-check API
        if self.api_key and self.api_url:
            try:
                api_result = self._check_via_api(claim)
                if api_result:
                    return api_result
            except Exception:
                pass
        
        # Method 4: Fall back to rule-based heuristics
        result = self._check_via_heuristics(claim)
        return result
    
    def _check_against_reference(self, claim: str, reference_text: str) -> Dict[str, Any]:
        """Check if claim is supported by reference text."""
        claim_keywords = set(claim.lower().split())
        ref_keywords = set(reference_text.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'and',
                     'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                     'by', 'from', 'as', 'into', 'through', 'during', 'before',
                     'after', 'above', 'below', 'between', 'under', 'again',
                     'further', 'then', 'once', 'here', 'there', 'when', 'where',
                     'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other',
                     'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                     'so', 'than', 'too', 'very', 'just', 'can', 'it', 'its'}
        
        claim_keywords = claim_keywords - stop_words
        ref_keywords = ref_keywords - stop_words
        
        if not claim_keywords:
            return {"claim": claim, "verified": False, "disputed": False, 
                   "confidence": 0.5, "source": "reference", "method": "reference"}
        
        # Calculate overlap
        overlap = len(claim_keywords & ref_keywords)
        similarity = overlap / len(claim_keywords) if claim_keywords else 0
        
        return {
            "claim": claim,
            "verified": similarity > 0.6,
            "disputed": False,
            "confidence": similarity,
            "source": "reference",
            "method": "reference"
        }
    
    def _check_via_wikipedia(self, claim: str, entities: List[str]) -> Optional[Dict[str, Any]]:
        """Check claim using Wikipedia API."""
        if not self._wiki:
            return None
        
        for entity in entities[:3]:  # Check top 3 entities
            try:
                page = self._wiki.page(entity)
                
                if page.exists():
                    # Get summary and check if claim keywords appear
                    summary = page.summary.lower()
                    claim_words = set(claim.lower().split())
                    
                    # Remove stop words
                    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
                                 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                                 'of', 'with', 'it', 'its'}
                    claim_words = claim_words - stop_words
                    
                    # Count how many claim words appear in summary
                    matches = sum(1 for word in claim_words if word in summary)
                    confidence = matches / len(claim_words) if claim_words else 0
                    
                    if confidence > 0.3:  # Some relevant content found
                        return {
                            "claim": claim,
                            "verified": confidence > 0.5,
                            "disputed": False,
                            "confidence": min(confidence * 1.2, 1.0),  # Boost slightly
                            "source": f"Wikipedia: {entity}",
                            "method": "wikipedia"
                        }
            except Exception:
                continue
        
        return None
    
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
        except Exception:
            pass
        
        return None
    
    def _check_via_heuristics(self, claim: str) -> Dict[str, Any]:
        """Apply heuristic rules to estimate claim reliability."""
        confidence = 0.5  # Start with neutral confidence
        
        # Boost confidence for specific, verifiable claims
        if re.search(r'\d{4}', claim):  # Contains a year
            confidence += 0.1
        if re.search(r'\d+', claim):  # Contains numbers
            confidence += 0.05
        
        # Reduce confidence for vague claims
        vague_indicators = ['some say', 'many believe', 'it is said', 'reportedly',
                          'allegedly', 'supposedly', 'claimed', 'rumored']
        if any(ind in claim.lower() for ind in vague_indicators):
            confidence -= 0.2
        
        # Reduce confidence for absolute statements
        absolute_indicators = ['always', 'never', 'everyone', 'no one', 'all',
                              'none', 'every', 'completely', 'totally', 'absolutely']
        if any(ind in claim.lower() for ind in absolute_indicators):
            confidence -= 0.1
        
        confidence = max(0.1, min(0.9, confidence))  # Clamp between 0.1 and 0.9
        
        return {
            "claim": claim,
            "verified": confidence > 0.6,
            "disputed": False,
            "confidence": confidence,
            "source": None,
            "method": "heuristic"
        }
