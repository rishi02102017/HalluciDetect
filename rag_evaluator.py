"""
RAG (Retrieval-Augmented Generation) Evaluation Module.
Phase 2 Feature: RAG Evaluation

Evaluates:
- Context Relevance: Is the retrieved context relevant to the query?
- Answer Faithfulness: Is the answer grounded in the context?
- Answer Relevance: Does the answer address the query?
- Groundedness: Are claims in the answer supported by context?
"""
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter


class RAGEvaluator:
    """Evaluate RAG pipeline outputs for quality and faithfulness."""
    
    def __init__(self):
        self._stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'and', 'but', 'if', 'or', 'because', 'this', 'that', 'these', 'those',
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
            'she', 'her', 'it', 'its', 'they', 'them', 'their', 'what', 'which',
            'who', 'whom', 'when', 'where', 'why', 'how'
        }
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize and lowercase text."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def get_content_words(self, text: str) -> List[str]:
        """Get non-stopword tokens."""
        tokens = self.tokenize(text)
        return [t for t in tokens if t not in self._stopwords]
    
    def extract_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    # ==================== Context Relevance ====================
    
    def context_relevance(
        self, 
        query: str, 
        contexts: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate how relevant the retrieved contexts are to the query.
        
        Uses keyword overlap and TF-IDF-style weighting.
        
        Args:
            query: The user's question/query
            contexts: List of retrieved context passages
        
        Returns:
            Dict with relevance scores per context and overall score
        """
        query_words = set(self.get_content_words(query))
        
        if not query_words:
            return {
                "overall_score": 0.0,
                "context_scores": [],
                "error": "Query has no content words"
            }
        
        context_scores = []
        
        for i, context in enumerate(contexts):
            context_words = set(self.get_content_words(context))
            
            if not context_words:
                context_scores.append({
                    "index": i,
                    "score": 0.0,
                    "matched_words": 0,
                    "query_coverage": 0.0
                })
                continue
            
            # Word overlap
            overlap = query_words & context_words
            
            # Query coverage (what % of query words appear in context)
            query_coverage = len(overlap) / len(query_words)
            
            # Context density (what % of context is query-relevant)
            context_density = len(overlap) / len(context_words) if context_words else 0
            
            # Combined score (weighted average)
            score = 0.7 * query_coverage + 0.3 * context_density
            
            context_scores.append({
                "index": i,
                "score": round(score, 4),
                "matched_words": len(overlap),
                "query_coverage": round(query_coverage, 4),
                "context_density": round(context_density, 4),
                "preview": context[:100] + "..." if len(context) > 100 else context
            })
        
        # Sort by score
        context_scores.sort(key=lambda x: x["score"], reverse=True)
        
        # Overall score (average of top contexts or weighted by position)
        if context_scores:
            # Weight earlier contexts more (assuming retrieval ranking)
            weights = [1 / (i + 1) for i in range(len(context_scores))]
            weighted_sum = sum(s["score"] * w for s, w in zip(context_scores, weights))
            overall = weighted_sum / sum(weights)
        else:
            overall = 0.0
        
        return {
            "overall_score": round(overall, 4),
            "overall_percent": round(overall * 100, 2),
            "num_contexts": len(contexts),
            "context_scores": context_scores,
            "most_relevant_index": context_scores[0]["index"] if context_scores else None
        }
    
    # ==================== Answer Faithfulness ====================
    
    def answer_faithfulness(
        self, 
        answer: str, 
        contexts: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate if the answer is grounded in the provided contexts.
        
        Checks if claims/statements in the answer can be traced to contexts.
        
        Args:
            answer: The generated answer
            contexts: List of context passages used for generation
        
        Returns:
            Dict with faithfulness score and analysis
        """
        answer_sentences = self.extract_sentences(answer)
        combined_context = " ".join(contexts).lower()
        context_words = set(self.get_content_words(combined_context))
        
        sentence_analysis = []
        
        for i, sentence in enumerate(answer_sentences):
            sentence_words = set(self.get_content_words(sentence))
            
            if not sentence_words:
                sentence_analysis.append({
                    "index": i,
                    "sentence": sentence,
                    "grounded": True,  # Empty/trivial sentences are OK
                    "support_score": 1.0,
                    "reason": "No content words"
                })
                continue
            
            # Check overlap with context
            overlap = sentence_words & context_words
            support_score = len(overlap) / len(sentence_words)
            
            # Check for specific phrases in context
            sentence_lower = sentence.lower()
            phrase_found = any(
                phrase in combined_context 
                for phrase in self._extract_key_phrases(sentence)
            )
            
            # Boost score if key phrases found
            if phrase_found:
                support_score = min(1.0, support_score + 0.2)
            
            grounded = support_score >= 0.5
            
            sentence_analysis.append({
                "index": i,
                "sentence": sentence[:100] + "..." if len(sentence) > 100 else sentence,
                "grounded": grounded,
                "support_score": round(support_score, 4),
                "matched_words": len(overlap),
                "total_words": len(sentence_words)
            })
        
        # Overall faithfulness
        if sentence_analysis:
            grounded_count = sum(1 for s in sentence_analysis if s["grounded"])
            faithfulness = grounded_count / len(sentence_analysis)
            avg_support = sum(s["support_score"] for s in sentence_analysis) / len(sentence_analysis)
        else:
            faithfulness = 1.0
            avg_support = 1.0
        
        return {
            "faithfulness_score": round(faithfulness, 4),
            "faithfulness_percent": round(faithfulness * 100, 2),
            "average_support": round(avg_support, 4),
            "total_sentences": len(sentence_analysis),
            "grounded_sentences": sum(1 for s in sentence_analysis if s["grounded"]),
            "ungrounded_sentences": [
                s for s in sentence_analysis if not s["grounded"]
            ],
            "sentence_analysis": sentence_analysis
        }
    
    def _extract_key_phrases(self, text: str, min_words: int = 2, max_words: int = 4) -> List[str]:
        """Extract key phrases from text."""
        words = self.tokenize(text)
        phrases = []
        
        for length in range(min_words, max_words + 1):
            for i in range(len(words) - length + 1):
                phrase = " ".join(words[i:i+length])
                # Skip if all stopwords
                if any(w not in self._stopwords for w in words[i:i+length]):
                    phrases.append(phrase)
        
        return phrases[:10]  # Limit to top 10 phrases
    
    # ==================== Answer Relevance ====================
    
    def answer_relevance(
        self, 
        query: str, 
        answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate if the answer addresses the query.
        
        Args:
            query: The user's question
            answer: The generated answer
        
        Returns:
            Dict with relevance score
        """
        query_words = set(self.get_content_words(query))
        answer_words = set(self.get_content_words(answer))
        
        if not query_words:
            return {"relevance_score": 0.0, "error": "Empty query"}
        
        if not answer_words:
            return {"relevance_score": 0.0, "error": "Empty answer"}
        
        # Query term coverage in answer
        overlap = query_words & answer_words
        query_coverage = len(overlap) / len(query_words)
        
        # Check for question-answer patterns
        question_words = {'what', 'when', 'where', 'who', 'why', 'how', 'which'}
        query_tokens = set(self.tokenize(query))
        is_question = bool(query_tokens & question_words) or query.strip().endswith('?')
        
        # Answer should not just repeat the question
        answer_unique = answer_words - query_words
        answer_novelty = len(answer_unique) / len(answer_words) if answer_words else 0
        
        # Combined relevance score
        relevance = 0.5 * query_coverage + 0.5 * answer_novelty
        
        return {
            "relevance_score": round(relevance, 4),
            "relevance_percent": round(relevance * 100, 2),
            "query_coverage": round(query_coverage, 4),
            "answer_novelty": round(answer_novelty, 4),
            "is_question": is_question,
            "query_words_matched": len(overlap),
            "query_words_total": len(query_words)
        }
    
    # ==================== Groundedness (Claim-level) ====================
    
    def groundedness(
        self, 
        answer: str, 
        contexts: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate groundedness at the claim level.
        
        Extracts factual claims from answer and checks against contexts.
        
        Args:
            answer: The generated answer
            contexts: List of context passages
        
        Returns:
            Dict with groundedness score and claim analysis
        """
        # Extract potential claims (sentences with factual patterns)
        claims = self._extract_claims(answer)
        combined_context = " ".join(contexts).lower()
        
        claim_analysis = []
        
        for claim in claims:
            claim_lower = claim.lower()
            claim_words = set(self.get_content_words(claim))
            context_words = set(self.get_content_words(combined_context))
            
            if not claim_words:
                continue
            
            # Word-level support
            overlap = claim_words & context_words
            word_support = len(overlap) / len(claim_words)
            
            # Phrase-level support
            key_phrases = self._extract_key_phrases(claim)
            phrases_found = sum(1 for p in key_phrases if p in combined_context)
            phrase_support = phrases_found / len(key_phrases) if key_phrases else 0
            
            # Combined support score
            support = 0.6 * word_support + 0.4 * phrase_support
            
            claim_analysis.append({
                "claim": claim[:150] + "..." if len(claim) > 150 else claim,
                "supported": support >= 0.4,
                "support_score": round(support, 4),
                "word_support": round(word_support, 4),
                "phrase_support": round(phrase_support, 4)
            })
        
        # Overall groundedness
        if claim_analysis:
            supported_count = sum(1 for c in claim_analysis if c["supported"])
            groundedness = supported_count / len(claim_analysis)
            avg_support = sum(c["support_score"] for c in claim_analysis) / len(claim_analysis)
        else:
            groundedness = 1.0  # No claims = fully grounded (trivially)
            avg_support = 1.0
        
        return {
            "groundedness_score": round(groundedness, 4),
            "groundedness_percent": round(groundedness * 100, 2),
            "average_support": round(avg_support, 4),
            "total_claims": len(claim_analysis),
            "supported_claims": sum(1 for c in claim_analysis if c["supported"]),
            "unsupported_claims": [c for c in claim_analysis if not c["supported"]],
            "claim_analysis": claim_analysis
        }
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text."""
        sentences = self.extract_sentences(text)
        claims = []
        
        # Patterns that indicate factual claims
        factual_patterns = [
            r'\b(is|are|was|were)\b',
            r'\b(has|have|had)\b',
            r'\b(can|could|will|would)\b',
            r'\b\d+\b',  # Numbers often indicate facts
            r'\b(according to|based on|studies show)\b',
        ]
        
        for sentence in sentences:
            # Check if sentence looks like a factual claim
            is_factual = any(re.search(p, sentence.lower()) for p in factual_patterns)
            
            # Exclude questions and commands
            is_question = sentence.strip().endswith('?')
            is_command = sentence.lower().startswith(('please', 'let', 'try'))
            
            if is_factual and not is_question and not is_command:
                claims.append(sentence)
        
        return claims
    
    # ==================== Full RAG Evaluation ====================
    
    def evaluate(
        self, 
        query: str, 
        answer: str, 
        contexts: List[str]
    ) -> Dict[str, Any]:
        """
        Perform full RAG evaluation.
        
        Args:
            query: User's question
            answer: Generated answer
            contexts: Retrieved context passages
        
        Returns:
            Dict with all RAG metrics
        """
        context_rel = self.context_relevance(query, contexts)
        faithfulness = self.answer_faithfulness(answer, contexts)
        answer_rel = self.answer_relevance(query, answer)
        grounded = self.groundedness(answer, contexts)
        
        # Overall RAG score (weighted average)
        weights = {
            "context_relevance": 0.25,
            "faithfulness": 0.30,
            "answer_relevance": 0.25,
            "groundedness": 0.20
        }
        
        overall = (
            weights["context_relevance"] * context_rel["overall_score"] +
            weights["faithfulness"] * faithfulness["faithfulness_score"] +
            weights["answer_relevance"] * answer_rel["relevance_score"] +
            weights["groundedness"] * grounded["groundedness_score"]
        )
        
        return {
            "overall_score": round(overall, 4),
            "overall_percent": round(overall * 100, 2),
            "context_relevance": context_rel,
            "faithfulness": faithfulness,
            "answer_relevance": answer_rel,
            "groundedness": grounded,
            "weights_used": weights,
            "summary": {
                "context_relevance": context_rel["overall_percent"],
                "faithfulness": faithfulness["faithfulness_percent"],
                "answer_relevance": answer_rel["relevance_percent"],
                "groundedness": grounded["groundedness_percent"],
                "overall": round(overall * 100, 2)
            }
        }


# Singleton
_rag_evaluator = None

def get_rag_evaluator() -> RAGEvaluator:
    """Get or create RAG evaluator instance."""
    global _rag_evaluator
    if _rag_evaluator is None:
        _rag_evaluator = RAGEvaluator()
    return _rag_evaluator

