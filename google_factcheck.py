"""Google Fact Check Tools API integration."""
import requests
from typing import List, Dict, Any, Optional


class GoogleFactChecker:
    """
    Google Fact Check Tools API client.
    
    This API is FREE and doesn't require an API key for basic usage.
    Documentation: https://developers.google.com/fact-check/tools/api
    """
    
    BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Google Fact Checker.
        
        Args:
            api_key: Optional API key for higher rate limits
        """
        self.api_key = api_key
    
    def search_claims(self, query: str, language_code: str = "en", 
                     max_age_days: int = None) -> List[Dict[str, Any]]:
        """
        Search for fact-checked claims related to a query.
        
        Args:
            query: The text to search for fact checks
            language_code: Language code (default: "en")
            max_age_days: Maximum age of fact checks in days
            
        Returns:
            List of fact check results
        """
        params = {
            "query": query,
            "languageCode": language_code,
        }
        
        if self.api_key:
            params["key"] = self.api_key
            
        if max_age_days:
            params["maxAgeDays"] = max_age_days
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_claims(data.get("claims", []))
            else:
                return []
                
        except requests.RequestException:
            return []
    
    def _parse_claims(self, claims: List[Dict]) -> List[Dict[str, Any]]:
        """Parse and format claim results."""
        results = []
        
        for claim in claims:
            claim_text = claim.get("text", "")
            claimant = claim.get("claimant", "Unknown")
            
            # Get claim reviews (fact checks)
            reviews = []
            for review in claim.get("claimReview", []):
                publisher = review.get("publisher", {})
                reviews.append({
                    "publisher": publisher.get("name", "Unknown"),
                    "publisher_site": publisher.get("site", ""),
                    "url": review.get("url", ""),
                    "title": review.get("title", ""),
                    "rating": review.get("textualRating", ""),
                    "language": review.get("languageCode", "en"),
                    "review_date": review.get("reviewDate", "")
                })
            
            results.append({
                "claim": claim_text,
                "claimant": claimant,
                "claim_date": claim.get("claimDate", ""),
                "reviews": reviews,
                "review_count": len(reviews)
            })
        
        return results
    
    def check_claim(self, claim: str) -> Dict[str, Any]:
        """
        Check a single claim and return aggregated results.
        
        Args:
            claim: The claim to fact-check
            
        Returns:
            Dictionary with fact check results and confidence score
        """
        results = self.search_claims(claim)
        
        if not results:
            return {
                "claim": claim,
                "found": False,
                "confidence": 0.0,
                "verdict": "unverified",
                "message": "No fact checks found for this claim",
                "sources": []
            }
        
        # Analyze the results
        ratings = []
        sources = []
        
        for result in results:
            for review in result.get("reviews", []):
                rating = review.get("rating", "").lower()
                ratings.append(rating)
                sources.append({
                    "publisher": review.get("publisher"),
                    "rating": review.get("rating"),
                    "url": review.get("url"),
                    "title": review.get("title")
                })
        
        # Calculate confidence and verdict
        verdict, confidence = self._calculate_verdict(ratings)
        
        return {
            "claim": claim,
            "found": True,
            "confidence": confidence,
            "verdict": verdict,
            "fact_checks_found": len(results),
            "sources": sources[:5],  # Top 5 sources
            "all_ratings": ratings
        }
    
    def _calculate_verdict(self, ratings: List[str]) -> tuple:
        """
        Calculate overall verdict from multiple ratings.
        
        Returns:
            Tuple of (verdict, confidence)
        """
        if not ratings:
            return ("unverified", 0.0)
        
        # Common rating keywords
        true_keywords = ["true", "correct", "accurate", "verified", "confirmed"]
        false_keywords = ["false", "incorrect", "inaccurate", "fake", "pants on fire", 
                         "wrong", "misleading", "mostly false", "lie"]
        mixed_keywords = ["partly true", "half true", "mixed", "mostly true", 
                         "partially", "context needed"]
        
        true_count = 0
        false_count = 0
        mixed_count = 0
        
        for rating in ratings:
            rating_lower = rating.lower()
            if any(kw in rating_lower for kw in false_keywords):
                false_count += 1
            elif any(kw in rating_lower for kw in true_keywords):
                true_count += 1
            elif any(kw in rating_lower for kw in mixed_keywords):
                mixed_count += 1
        
        total = true_count + false_count + mixed_count
        
        if total == 0:
            return ("unverified", 0.3)
        
        # Determine verdict
        if false_count > true_count and false_count > mixed_count:
            confidence = false_count / total
            return ("false", min(0.95, confidence))
        elif true_count > false_count and true_count > mixed_count:
            confidence = true_count / total
            return ("true", min(0.95, confidence))
        else:
            confidence = mixed_count / total if mixed_count > 0 else 0.5
            return ("mixed", confidence)


# Singleton instance
_google_fact_checker = None

def get_google_fact_checker(api_key: Optional[str] = None) -> GoogleFactChecker:
    """Get or create GoogleFactChecker instance."""
    global _google_fact_checker
    if _google_fact_checker is None:
        _google_fact_checker = GoogleFactChecker(api_key)
    return _google_fact_checker

