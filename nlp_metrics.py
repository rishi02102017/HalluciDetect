"""
NLP Metrics Module - BLEU, ROUGE, and other standard evaluation metrics.
Phase 2 Feature: Enhanced Scoring Methods
"""
import re
import math
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple


class NLPMetrics:
    """Calculate standard NLP evaluation metrics."""
    
    def __init__(self):
        self._stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'and', 'but', 'if', 'or', 'because', 'until', 'while', 'although',
            'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we',
            'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
            'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
            'themselves', 'what', 'which', 'who', 'whom'
        }
    
    def tokenize(self, text: str, lowercase: bool = True) -> List[str]:
        """Simple tokenization."""
        if lowercase:
            text = text.lower()
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Generate n-grams from tokens."""
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    # ==================== BLEU Score ====================
    
    def bleu_score(
        self, 
        candidate: str, 
        reference: str, 
        max_n: int = 4,
        weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate BLEU score (Bilingual Evaluation Understudy).
        
        Used for: Translation quality, text generation
        
        Args:
            candidate: Generated text to evaluate
            reference: Reference/ground truth text
            max_n: Maximum n-gram order (default 4)
            weights: Weights for each n-gram (default: uniform)
        
        Returns:
            Dict with BLEU score and component scores
        """
        if weights is None:
            weights = [1.0 / max_n] * max_n
        
        candidate_tokens = self.tokenize(candidate)
        reference_tokens = self.tokenize(reference)
        
        if len(candidate_tokens) == 0:
            return {
                "bleu": 0.0,
                "precisions": [0.0] * max_n,
                "brevity_penalty": 0.0,
                "candidate_length": 0,
                "reference_length": len(reference_tokens)
            }
        
        # Calculate precision for each n-gram order
        precisions = []
        for n in range(1, max_n + 1):
            candidate_ngrams = self.get_ngrams(candidate_tokens, n)
            reference_ngrams = self.get_ngrams(reference_tokens, n)
            
            if len(candidate_ngrams) == 0:
                precisions.append(0.0)
                continue
            
            candidate_counts = Counter(candidate_ngrams)
            reference_counts = Counter(reference_ngrams)
            
            # Clipped counts
            clipped_counts = {
                ngram: min(count, reference_counts.get(ngram, 0))
                for ngram, count in candidate_counts.items()
            }
            
            precision = sum(clipped_counts.values()) / len(candidate_ngrams)
            precisions.append(precision)
        
        # Brevity penalty
        c = len(candidate_tokens)
        r = len(reference_tokens)
        
        if c > r:
            bp = 1.0
        elif c == 0:
            bp = 0.0
        else:
            bp = math.exp(1 - r / c)
        
        # Geometric mean of precisions
        if any(p == 0 for p in precisions):
            bleu = 0.0
        else:
            log_precisions = [w * math.log(p) for w, p in zip(weights, precisions)]
            bleu = bp * math.exp(sum(log_precisions))
        
        return {
            "bleu": round(bleu, 4),
            "bleu_percent": round(bleu * 100, 2),
            "precisions": [round(p, 4) for p in precisions],
            "brevity_penalty": round(bp, 4),
            "candidate_length": c,
            "reference_length": r
        }
    
    # ==================== ROUGE Scores ====================
    
    def rouge_n(
        self, 
        candidate: str, 
        reference: str, 
        n: int = 1
    ) -> Dict[str, float]:
        """
        Calculate ROUGE-N score (Recall-Oriented Understudy for Gisting Evaluation).
        
        Used for: Summarization evaluation
        
        Args:
            candidate: Generated summary
            reference: Reference summary
            n: N-gram order (1 for unigrams, 2 for bigrams)
        
        Returns:
            Dict with precision, recall, and F1 score
        """
        candidate_tokens = self.tokenize(candidate)
        reference_tokens = self.tokenize(reference)
        
        candidate_ngrams = self.get_ngrams(candidate_tokens, n)
        reference_ngrams = self.get_ngrams(reference_tokens, n)
        
        if len(reference_ngrams) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        candidate_counts = Counter(candidate_ngrams)
        reference_counts = Counter(reference_ngrams)
        
        # Overlapping n-grams
        overlap = sum(
            min(candidate_counts.get(ngram, 0), count)
            for ngram, count in reference_counts.items()
        )
        
        precision = overlap / len(candidate_ngrams) if candidate_ngrams else 0.0
        recall = overlap / len(reference_ngrams)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }
    
    def rouge_l(self, candidate: str, reference: str) -> Dict[str, float]:
        """
        Calculate ROUGE-L score using Longest Common Subsequence.
        
        Args:
            candidate: Generated text
            reference: Reference text
        
        Returns:
            Dict with precision, recall, and F1 score
        """
        candidate_tokens = self.tokenize(candidate)
        reference_tokens = self.tokenize(reference)
        
        if len(reference_tokens) == 0 or len(candidate_tokens) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # LCS length using dynamic programming
        lcs_length = self._lcs_length(candidate_tokens, reference_tokens)
        
        precision = lcs_length / len(candidate_tokens)
        recall = lcs_length / len(reference_tokens)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "lcs_length": lcs_length
        }
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of Longest Common Subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def rouge_scores(self, candidate: str, reference: str) -> Dict[str, Any]:
        """
        Calculate all ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
        
        Args:
            candidate: Generated text
            reference: Reference text
        
        Returns:
            Dict with all ROUGE variants
        """
        return {
            "rouge_1": self.rouge_n(candidate, reference, n=1),
            "rouge_2": self.rouge_n(candidate, reference, n=2),
            "rouge_l": self.rouge_l(candidate, reference)
        }
    
    # ==================== Additional Metrics ====================
    
    def meteor_score(self, candidate: str, reference: str) -> Dict[str, float]:
        """
        Simplified METEOR score (without WordNet synonyms).
        
        Metric for Evaluation of Translation with Explicit ORdering.
        """
        candidate_tokens = set(self.tokenize(candidate))
        reference_tokens = set(self.tokenize(reference))
        
        if len(reference_tokens) == 0:
            return {"meteor": 0.0, "precision": 0.0, "recall": 0.0}
        
        matches = candidate_tokens & reference_tokens
        
        precision = len(matches) / len(candidate_tokens) if candidate_tokens else 0.0
        recall = len(matches) / len(reference_tokens)
        
        if precision + recall == 0:
            f_mean = 0.0
        else:
            # METEOR uses weighted harmonic mean (recall weighted higher)
            alpha = 0.9
            f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
        
        return {
            "meteor": round(f_mean, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4)
        }
    
    def word_error_rate(self, candidate: str, reference: str) -> Dict[str, Any]:
        """
        Calculate Word Error Rate (WER).
        
        Used for: Speech recognition, OCR evaluation
        Lower is better (0 = perfect match)
        """
        candidate_tokens = self.tokenize(candidate)
        reference_tokens = self.tokenize(reference)
        
        if len(reference_tokens) == 0:
            return {"wer": 0.0 if len(candidate_tokens) == 0 else 1.0}
        
        # Levenshtein distance at word level
        d = [[0] * (len(reference_tokens) + 1) for _ in range(len(candidate_tokens) + 1)]
        
        for i in range(len(candidate_tokens) + 1):
            d[i][0] = i
        for j in range(len(reference_tokens) + 1):
            d[0][j] = j
        
        for i in range(1, len(candidate_tokens) + 1):
            for j in range(1, len(reference_tokens) + 1):
                if candidate_tokens[i-1] == reference_tokens[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(
                        d[i-1][j] + 1,      # Deletion
                        d[i][j-1] + 1,      # Insertion
                        d[i-1][j-1] + 1     # Substitution
                    )
        
        wer = d[len(candidate_tokens)][len(reference_tokens)] / len(reference_tokens)
        
        return {
            "wer": round(wer, 4),
            "wer_percent": round(wer * 100, 2),
            "edit_distance": d[len(candidate_tokens)][len(reference_tokens)],
            "reference_length": len(reference_tokens)
        }
    
    def calculate_all(
        self, 
        candidate: str, 
        reference: str,
        include_bleu: bool = True,
        include_rouge: bool = True,
        include_meteor: bool = True,
        include_wer: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate all available NLP metrics.
        
        Args:
            candidate: Generated text
            reference: Reference text
            include_*: Flags to include specific metrics
        
        Returns:
            Dict with all requested metrics
        """
        result = {}
        
        if include_bleu:
            result["bleu"] = self.bleu_score(candidate, reference)
        
        if include_rouge:
            result["rouge"] = self.rouge_scores(candidate, reference)
        
        if include_meteor:
            result["meteor"] = self.meteor_score(candidate, reference)
        
        if include_wer:
            result["wer"] = self.word_error_rate(candidate, reference)
        
        # Summary scores for quick reference
        result["summary"] = {
            "bleu": result.get("bleu", {}).get("bleu_percent", 0),
            "rouge_1_f1": result.get("rouge", {}).get("rouge_1", {}).get("f1", 0) * 100,
            "rouge_2_f1": result.get("rouge", {}).get("rouge_2", {}).get("f1", 0) * 100,
            "rouge_l_f1": result.get("rouge", {}).get("rouge_l", {}).get("f1", 0) * 100,
            "meteor": result.get("meteor", {}).get("meteor", 0) * 100,
            "wer": result.get("wer", {}).get("wer_percent", 100)
        }
        
        return result


# Singleton instance
_nlp_metrics = None

def get_nlp_metrics() -> NLPMetrics:
    """Get or create NLP metrics instance."""
    global _nlp_metrics
    if _nlp_metrics is None:
        _nlp_metrics = NLPMetrics()
    return _nlp_metrics

