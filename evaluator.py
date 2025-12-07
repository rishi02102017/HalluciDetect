"""Main evaluation pipeline that combines all evaluation methods."""
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from llm_client import LLMClient
from fact_checker import FactChecker
from semantic_similarity import SemanticSimilarityChecker
from rule_based_scorer import RuleBasedScorer
from models import EvaluationResult, EvaluationBatch
from config import Config

class HallucinationEvaluator:
    """Main evaluator that orchestrates all evaluation methods."""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.fact_checker = FactChecker()
        self.semantic_checker = SemanticSimilarityChecker()
        self.rule_scorer = RuleBasedScorer()
        self.hallucination_threshold = Config.HALLUCINATION_SCORE_THRESHOLD
    
    def evaluate(
        self,
        prompt: str,
        model_name: str = None,
        prompt_version: str = "v1",
        reference_text: Optional[str] = None,
        generate_output: bool = True
    ) -> EvaluationResult:
        """
        Evaluate a single LLM output for hallucinations.
        
        Args:
            prompt: Input prompt
            model_name: LLM model to use (if generate_output=True)
            prompt_version: Version identifier for the prompt
            reference_text: Optional reference/ground truth text
            generate_output: Whether to generate output or use provided reference_text
            
        Returns:
            EvaluationResult object
        """
        # Generate LLM output if needed
        if generate_output:
            if not model_name:
                model_name = Config.DEFAULT_LLM_MODEL
            llm_output = self.llm_client.generate(prompt, model=model_name)
        else:
            llm_output = reference_text or ""
        
        # Run all evaluation methods
        fact_check_result = self.fact_checker.check_facts(llm_output, reference_text)
        semantic_result = self.semantic_checker.compute_similarity(
            llm_output,
            reference_text or prompt
        ) if reference_text else {"similarity_score": 0.5, "is_similar": False}
        rule_based_result = self.rule_scorer.score(llm_output, reference_text)
        
        # Calculate overall hallucination score
        # Higher score = more likely hallucination
        overall_score = self._calculate_overall_score(
            fact_check_result,
            semantic_result,
            rule_based_result
        )
        
        # Determine if it's a hallucination
        is_hallucination = overall_score >= self.hallucination_threshold
        
        # Create result object
        result = EvaluationResult(
            id=str(uuid.uuid4()),
            prompt=prompt,
            llm_output=llm_output,
            model_name=model_name or Config.DEFAULT_LLM_MODEL,
            prompt_version=prompt_version,
            timestamp=datetime.now(),
            semantic_similarity_score=semantic_result.get("similarity_score", 0.0),
            fact_check_score=fact_check_result.get("score", 0.0),
            rule_based_score=rule_based_result.get("overall_score", 0.0),
            overall_hallucination_score=overall_score,
            fact_check_details=fact_check_result,
            semantic_similarity_details=semantic_result,
            rule_based_details=rule_based_result,
            is_hallucination=is_hallucination,
            confidence=1.0 - overall_score,  # Confidence that it's NOT a hallucination
            evaluation_metadata={
                "threshold": self.hallucination_threshold,
                "reference_provided": reference_text is not None
            }
        )
        
        return result
    
    def evaluate_batch(
        self,
        prompt_template: str,
        test_cases: List[Dict[str, str]],
        model_name: str = None,
        prompt_version: str = "v1"
    ) -> EvaluationBatch:
        """
        Evaluate multiple test cases in a batch.
        
        Args:
            prompt_template: Template for prompts (use {key} for placeholders)
            test_cases: List of dicts with test case data
            model_name: LLM model to use
            prompt_version: Version identifier
            
        Returns:
            EvaluationBatch object
        """
        batch_id = str(uuid.uuid4())
        results = []
        
        for test_case in test_cases:
            # Format prompt with test case data
            prompt = prompt_template.format(**test_case)
            reference_text = test_case.get("reference", None)
            
            result = self.evaluate(
                prompt=prompt,
                model_name=model_name,
                prompt_version=prompt_version,
                reference_text=reference_text,
                generate_output=True
            )
            results.append(result)
        
        # Calculate aggregate statistics
        total = len(results)
        hallucination_count = sum(1 for r in results if r.is_hallucination)
        hallucination_rate = hallucination_count / total if total > 0 else 0.0
        
        # Calculate average scores
        avg_scores = {
            "semantic_similarity": sum(r.semantic_similarity_score for r in results) / total if total > 0 else 0.0,
            "fact_check": sum(r.fact_check_score for r in results) / total if total > 0 else 0.0,
            "rule_based": sum(r.rule_based_score for r in results) / total if total > 0 else 0.0,
            "overall_hallucination": sum(r.overall_hallucination_score for r in results) / total if total > 0 else 0.0,
        }
        
        batch = EvaluationBatch(
            batch_id=batch_id,
            prompt_template=prompt_template,
            model_name=model_name or Config.DEFAULT_LLM_MODEL,
            prompt_version=prompt_version,
            test_cases=test_cases,
            created_at=datetime.now(),
            results=results,
            total_evaluations=total,
            hallucination_count=hallucination_count,
            hallucination_rate=hallucination_rate,
            average_scores=avg_scores
        )
        
        return batch
    
    def _calculate_overall_score(
        self,
        fact_check_result: Dict[str, Any],
        semantic_result: Dict[str, Any],
        rule_based_result: Dict[str, Any]
    ) -> float:
        """
        Calculate overall hallucination score from individual scores.
        
        Higher score = more likely hallucination
        """
        # Invert fact_check_score (higher fact_check = less hallucination)
        fact_score = 1.0 - fact_check_result.get("score", 0.5)
        
        # Invert semantic similarity (lower similarity = more hallucination)
        semantic_score = 1.0 - semantic_result.get("similarity_score", 0.5)
        
        # Rule-based score is already in hallucination direction
        rule_score = rule_based_result.get("overall_score", 0.5)
        
        # Weighted average
        weights = {
            "fact_check": 0.4,
            "semantic": 0.3,
            "rule_based": 0.3
        }
        
        overall = (
            fact_score * weights["fact_check"] +
            semantic_score * weights["semantic"] +
            rule_score * weights["rule_based"]
        )
        
        return max(0.0, min(1.0, overall))

