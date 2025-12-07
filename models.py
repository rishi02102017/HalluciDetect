"""Data models for evaluation results."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses_json import dataclass_json, config
from marshmallow import fields

@dataclass_json
@dataclass
class EvaluationResult:
    """Result of a single LLM output evaluation."""
    id: str
    prompt: str
    llm_output: str
    model_name: str
    prompt_version: str
    timestamp: datetime = field(
        default_factory=datetime.now,
        metadata=config(
            encoder=datetime.isoformat,
            decoder=datetime.fromisoformat,
            mm_field=fields.DateTime(format='iso')
        )
    )
    
    # Evaluation scores
    semantic_similarity_score: float = 0.0
    fact_check_score: float = 0.0
    rule_based_score: float = 0.0
    overall_hallucination_score: float = 0.0
    
    # Detailed results
    fact_check_details: Dict[str, Any] = field(default_factory=dict)
    semantic_similarity_details: Dict[str, Any] = field(default_factory=dict)
    rule_based_details: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    is_hallucination: bool = False
    confidence: float = 0.0
    evaluation_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass_json
@dataclass
class EvaluationBatch:
    """Batch of evaluations for a specific prompt/model combination."""
    batch_id: str
    prompt_template: str
    model_name: str
    prompt_version: str
    test_cases: List[Dict[str, str]]
    created_at: datetime = field(
        default_factory=datetime.now,
        metadata=config(
            encoder=datetime.isoformat,
            decoder=datetime.fromisoformat,
            mm_field=fields.DateTime(format='iso')
        )
    )
    results: List[EvaluationResult] = field(default_factory=list)
    
    # Aggregate statistics
    total_evaluations: int = 0
    hallucination_count: int = 0
    hallucination_rate: float = 0.0
    average_scores: Dict[str, float] = field(default_factory=dict)

