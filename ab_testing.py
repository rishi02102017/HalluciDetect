"""
A/B Prompt Testing Module - Compare prompt versions with statistical analysis.
Phase 2 Feature: A/B Prompt Testing
"""
import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import uuid


@dataclass
class PromptVariant:
    """Represents a prompt variant in an A/B test."""
    id: str
    name: str
    prompt_template: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Results storage
    scores: List[float] = field(default_factory=list)
    latencies: List[float] = field(default_factory=list)
    hallucination_rates: List[float] = field(default_factory=list)
    
    def add_result(self, score: float, latency: float = 0.0, hallucination_rate: float = 0.0):
        """Add an evaluation result to this variant."""
        self.scores.append(score)
        self.latencies.append(latency)
        self.hallucination_rates.append(hallucination_rate)
    
    @property
    def sample_size(self) -> int:
        return len(self.scores)
    
    @property
    def mean_score(self) -> float:
        return statistics.mean(self.scores) if self.scores else 0.0
    
    @property
    def std_score(self) -> float:
        return statistics.stdev(self.scores) if len(self.scores) > 1 else 0.0
    
    @property
    def mean_latency(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0.0
    
    @property
    def mean_hallucination_rate(self) -> float:
        return statistics.mean(self.hallucination_rates) if self.hallucination_rates else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "prompt_template": self.prompt_template,
            "created_at": self.created_at.isoformat(),
            "sample_size": self.sample_size,
            "mean_score": round(self.mean_score, 4),
            "std_score": round(self.std_score, 4),
            "mean_latency": round(self.mean_latency, 3),
            "mean_hallucination_rate": round(self.mean_hallucination_rate, 4),
            "scores": self.scores,
            "latencies": self.latencies,
            "hallucination_rates": self.hallucination_rates
        }


@dataclass
class ABTest:
    """Represents an A/B test between prompt variants."""
    id: str
    name: str
    description: str
    variant_a: PromptVariant
    variant_b: PromptVariant
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "active"  # active, completed, cancelled
    
    # Optional: model and reference text for testing
    model_name: Optional[str] = None
    reference_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "variant_a": self.variant_a.to_dict(),
            "variant_b": self.variant_b.to_dict(),
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "model_name": self.model_name,
            "reference_text": self.reference_text
        }


class StatisticalTests:
    """Statistical significance testing for A/B experiments."""
    
    @staticmethod
    def welchs_t_test(
        sample1: List[float], 
        sample2: List[float]
    ) -> Dict[str, Any]:
        """
        Welch's t-test for comparing two samples with potentially unequal variances.
        
        Returns:
            Dict with t-statistic, p-value, and significance determination
        """
        n1, n2 = len(sample1), len(sample2)
        
        if n1 < 2 or n2 < 2:
            return {
                "t_statistic": None,
                "p_value": None,
                "significant": False,
                "error": "Insufficient samples (need at least 2 per group)"
            }
        
        mean1, mean2 = statistics.mean(sample1), statistics.mean(sample2)
        var1, var2 = statistics.variance(sample1), statistics.variance(sample2)
        
        # Standard error
        se = math.sqrt(var1/n1 + var2/n2)
        
        if se == 0:
            return {
                "t_statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "error": "Zero variance in samples"
            }
        
        # t-statistic
        t_stat = (mean1 - mean2) / se
        
        # Degrees of freedom (Welch-Satterthwaite)
        df_num = (var1/n1 + var2/n2) ** 2
        df_denom = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
        df = df_num / df_denom if df_denom > 0 else 1
        
        # Approximate p-value using normal distribution for large df
        # (proper implementation would use t-distribution)
        p_value = 2 * (1 - StatisticalTests._normal_cdf(abs(t_stat)))
        
        return {
            "t_statistic": round(t_stat, 4),
            "degrees_of_freedom": round(df, 2),
            "p_value": round(p_value, 4),
            "significant_95": p_value < 0.05,
            "significant_99": p_value < 0.01,
            "mean_difference": round(mean1 - mean2, 4),
            "sample_sizes": {"n1": n1, "n2": n2}
        }
    
    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Approximate normal CDF using error function approximation."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    @staticmethod
    def effect_size(sample1: List[float], sample2: List[float]) -> Dict[str, Any]:
        """
        Calculate Cohen's d effect size.
        
        Interpretation:
        - |d| < 0.2: negligible
        - 0.2 <= |d| < 0.5: small
        - 0.5 <= |d| < 0.8: medium
        - |d| >= 0.8: large
        """
        if len(sample1) < 2 or len(sample2) < 2:
            return {"cohens_d": None, "interpretation": "insufficient data"}
        
        mean1, mean2 = statistics.mean(sample1), statistics.mean(sample2)
        var1, var2 = statistics.variance(sample1), statistics.variance(sample2)
        n1, n2 = len(sample1), len(sample2)
        
        # Pooled standard deviation
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_std = math.sqrt(pooled_var)
        
        if pooled_std == 0:
            return {"cohens_d": 0.0, "interpretation": "no variance"}
        
        d = (mean1 - mean2) / pooled_std
        
        # Interpretation
        abs_d = abs(d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return {
            "cohens_d": round(d, 4),
            "abs_effect": round(abs_d, 4),
            "interpretation": interpretation
        }
    
    @staticmethod
    def confidence_interval(
        sample: List[float], 
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """Calculate confidence interval for sample mean."""
        if len(sample) < 2:
            mean = sample[0] if sample else 0
            return {"mean": mean, "lower": mean, "upper": mean, "margin": 0}
        
        n = len(sample)
        mean = statistics.mean(sample)
        std = statistics.stdev(sample)
        se = std / math.sqrt(n)
        
        # Z-score for confidence level (approximate)
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(confidence, 1.96)
        
        margin = z * se
        
        return {
            "mean": round(mean, 4),
            "lower": round(mean - margin, 4),
            "upper": round(mean + margin, 4),
            "margin": round(margin, 4),
            "confidence": confidence
        }


class ABTestManager:
    """Manages A/B testing experiments."""
    
    def __init__(self):
        self.tests: Dict[str, ABTest] = {}
        self.stats = StatisticalTests()
    
    def create_test(
        self,
        name: str,
        description: str,
        prompt_a: str,
        prompt_b: str,
        variant_a_name: str = "Control",
        variant_b_name: str = "Variant",
        model_name: Optional[str] = None,
        reference_text: Optional[str] = None
    ) -> ABTest:
        """Create a new A/B test."""
        test_id = str(uuid.uuid4())[:8]
        
        variant_a = PromptVariant(
            id=f"{test_id}-A",
            name=variant_a_name,
            prompt_template=prompt_a
        )
        
        variant_b = PromptVariant(
            id=f"{test_id}-B",
            name=variant_b_name,
            prompt_template=prompt_b
        )
        
        test = ABTest(
            id=test_id,
            name=name,
            description=description,
            variant_a=variant_a,
            variant_b=variant_b,
            model_name=model_name,
            reference_text=reference_text
        )
        
        self.tests[test_id] = test
        return test
    
    def add_result(
        self,
        test_id: str,
        variant: str,  # "A" or "B"
        score: float,
        latency: float = 0.0,
        hallucination_rate: float = 0.0
    ) -> bool:
        """Add an evaluation result to a variant."""
        if test_id not in self.tests:
            return False
        
        test = self.tests[test_id]
        
        if variant.upper() == "A":
            test.variant_a.add_result(score, latency, hallucination_rate)
        elif variant.upper() == "B":
            test.variant_b.add_result(score, latency, hallucination_rate)
        else:
            return False
        
        return True
    
    def analyze_test(self, test_id: str) -> Dict[str, Any]:
        """Perform statistical analysis on an A/B test."""
        if test_id not in self.tests:
            return {"error": "Test not found"}
        
        test = self.tests[test_id]
        scores_a = test.variant_a.scores
        scores_b = test.variant_b.scores
        
        # Basic stats
        analysis = {
            "test_id": test_id,
            "test_name": test.name,
            "status": test.status,
            "variant_a": {
                "name": test.variant_a.name,
                "sample_size": test.variant_a.sample_size,
                "mean_score": round(test.variant_a.mean_score, 4),
                "std_score": round(test.variant_a.std_score, 4),
                "mean_latency": round(test.variant_a.mean_latency, 3),
                "mean_hallucination_rate": round(test.variant_a.mean_hallucination_rate, 4),
                "confidence_interval": self.stats.confidence_interval(scores_a) if scores_a else None
            },
            "variant_b": {
                "name": test.variant_b.name,
                "sample_size": test.variant_b.sample_size,
                "mean_score": round(test.variant_b.mean_score, 4),
                "std_score": round(test.variant_b.std_score, 4),
                "mean_latency": round(test.variant_b.mean_latency, 3),
                "mean_hallucination_rate": round(test.variant_b.mean_hallucination_rate, 4),
                "confidence_interval": self.stats.confidence_interval(scores_b) if scores_b else None
            }
        }
        
        # Statistical tests (only if both have sufficient data)
        if len(scores_a) >= 2 and len(scores_b) >= 2:
            analysis["t_test"] = self.stats.welchs_t_test(scores_a, scores_b)
            analysis["effect_size"] = self.stats.effect_size(scores_a, scores_b)
            
            # Winner determination
            if analysis["t_test"]["significant_95"]:
                winner = "A" if test.variant_a.mean_score > test.variant_b.mean_score else "B"
                analysis["winner"] = {
                    "variant": winner,
                    "name": test.variant_a.name if winner == "A" else test.variant_b.name,
                    "confidence": "95%",
                    "improvement": abs(analysis["t_test"]["mean_difference"])
                }
            else:
                analysis["winner"] = {
                    "variant": None,
                    "message": "No statistically significant difference detected",
                    "recommendation": f"Need more samples (currently A:{len(scores_a)}, B:{len(scores_b)})"
                }
        else:
            analysis["winner"] = {
                "variant": None,
                "message": "Insufficient data for statistical analysis",
                "recommendation": "Collect at least 2 samples per variant"
            }
        
        return analysis
    
    def get_test(self, test_id: str) -> Optional[ABTest]:
        """Get a test by ID."""
        return self.tests.get(test_id)
    
    def list_tests(self) -> List[Dict[str, Any]]:
        """List all tests with summary info."""
        return [
            {
                "id": t.id,
                "name": t.name,
                "status": t.status,
                "created_at": t.created_at.isoformat(),
                "sample_sizes": {
                    "A": t.variant_a.sample_size,
                    "B": t.variant_b.sample_size
                }
            }
            for t in self.tests.values()
        ]
    
    def complete_test(self, test_id: str) -> bool:
        """Mark a test as completed."""
        if test_id in self.tests:
            self.tests[test_id].status = "completed"
            return True
        return False


# Singleton instance
_ab_manager = None

def get_ab_manager() -> ABTestManager:
    """Get or create A/B test manager instance."""
    global _ab_manager
    if _ab_manager is None:
        _ab_manager = ABTestManager()
    return _ab_manager

