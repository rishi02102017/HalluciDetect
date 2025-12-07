"""Database models and storage for evaluation results."""
import json
import numpy as np
from sqlalchemy import create_engine, Column, String, Float, Boolean, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from typing import List, Optional, Dict, Any
from config import Config
from models import EvaluationResult, EvaluationBatch

Base = declarative_base()

def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.bool_, np.integer)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

class EvaluationResultDB(Base):
    """Database model for evaluation results."""
    __tablename__ = "evaluation_results"
    
    id = Column(String, primary_key=True)
    prompt = Column(Text, nullable=False)
    llm_output = Column(Text, nullable=False)
    model_name = Column(String, nullable=False)
    prompt_version = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    
    semantic_similarity_score = Column(Float, default=0.0)
    fact_check_score = Column(Float, default=0.0)
    rule_based_score = Column(Float, default=0.0)
    overall_hallucination_score = Column(Float, default=0.0)
    
    fact_check_details = Column(JSON)
    semantic_similarity_details = Column(JSON)
    rule_based_details = Column(JSON)
    
    is_hallucination = Column(Boolean, default=False)
    confidence = Column(Float, default=0.0)
    evaluation_metadata = Column(JSON)

class EvaluationBatchDB(Base):
    """Database model for evaluation batches."""
    __tablename__ = "evaluation_batches"
    
    batch_id = Column(String, primary_key=True)
    prompt_template = Column(Text, nullable=False)
    model_name = Column(String, nullable=False)
    prompt_version = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    
    total_evaluations = Column(Float, default=0)
    hallucination_count = Column(Float, default=0)
    hallucination_rate = Column(Float, default=0.0)
    average_scores = Column(JSON)

class Database:
    """Database interface for storing and retrieving evaluation results."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or Config.DATABASE_URL
        self.engine = create_engine(self.database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def save_result(self, result: EvaluationResult) -> None:
        """Save an evaluation result to the database."""
        session = self.SessionLocal()
        try:
            db_result = EvaluationResultDB(
                id=result.id,
                prompt=result.prompt,
                llm_output=result.llm_output,
                model_name=result.model_name,
                prompt_version=result.prompt_version,
                timestamp=result.timestamp,
                semantic_similarity_score=float(result.semantic_similarity_score),
                fact_check_score=float(result.fact_check_score),
                rule_based_score=float(result.rule_based_score),
                overall_hallucination_score=float(result.overall_hallucination_score),
                fact_check_details=convert_to_serializable(result.fact_check_details),
                semantic_similarity_details=convert_to_serializable(result.semantic_similarity_details),
                rule_based_details=convert_to_serializable(result.rule_based_details),
                is_hallucination=bool(result.is_hallucination),
                confidence=float(result.confidence),
                evaluation_metadata=convert_to_serializable(result.evaluation_metadata)
            )
            session.add(db_result)
            session.commit()
        finally:
            session.close()
    
    def save_batch(self, batch: EvaluationBatch) -> None:
        """Save an evaluation batch to the database."""
        session = self.SessionLocal()
        try:
            # Save batch metadata
            db_batch = EvaluationBatchDB(
                batch_id=batch.batch_id,
                prompt_template=batch.prompt_template,
                model_name=batch.model_name,
                prompt_version=batch.prompt_version,
                created_at=batch.created_at,
                total_evaluations=batch.total_evaluations,
                hallucination_count=batch.hallucination_count,
                hallucination_rate=batch.hallucination_rate,
                average_scores=batch.average_scores
            )
            session.add(db_batch)
            
            # Save all results in the batch
            for result in batch.results:
                self.save_result(result)
            
            session.commit()
        finally:
            session.close()
    
    def get_results(
        self,
        model_name: Optional[str] = None,
        prompt_version: Optional[str] = None,
        limit: int = 100
    ) -> List[EvaluationResult]:
        """Retrieve evaluation results with optional filters."""
        session = self.SessionLocal()
        try:
            query = session.query(EvaluationResultDB)
            
            if model_name:
                query = query.filter(EvaluationResultDB.model_name == model_name)
            if prompt_version:
                query = query.filter(EvaluationResultDB.prompt_version == prompt_version)
            
            query = query.order_by(EvaluationResultDB.timestamp.desc()).limit(limit)
            db_results = query.all()
            
            results = []
            for db_result in db_results:
                result = EvaluationResult(
                    id=db_result.id,
                    prompt=db_result.prompt,
                    llm_output=db_result.llm_output,
                    model_name=db_result.model_name,
                    prompt_version=db_result.prompt_version,
                    timestamp=db_result.timestamp,
                    semantic_similarity_score=db_result.semantic_similarity_score,
                    fact_check_score=db_result.fact_check_score,
                    rule_based_score=db_result.rule_based_score,
                    overall_hallucination_score=db_result.overall_hallucination_score,
                    fact_check_details=db_result.fact_check_details or {},
                    semantic_similarity_details=db_result.semantic_similarity_details or {},
                    rule_based_details=db_result.rule_based_details or {},
                    is_hallucination=db_result.is_hallucination,
                    confidence=db_result.confidence,
                    evaluation_metadata=db_result.evaluation_metadata or {}
                )
                results.append(result)
            
            return results
        finally:
            session.close()
    
    def get_batches(
        self,
        model_name: Optional[str] = None,
        prompt_version: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Retrieve evaluation batches with optional filters."""
        session = self.SessionLocal()
        try:
            query = session.query(EvaluationBatchDB)
            
            if model_name:
                query = query.filter(EvaluationBatchDB.model_name == model_name)
            if prompt_version:
                query = query.filter(EvaluationBatchDB.prompt_version == prompt_version)
            
            query = query.order_by(EvaluationBatchDB.created_at.desc()).limit(limit)
            db_batches = query.all()
            
            batches = []
            for db_batch in db_batches:
                batches.append({
                    "batch_id": db_batch.batch_id,
                    "prompt_template": db_batch.prompt_template,
                    "model_name": db_batch.model_name,
                    "prompt_version": db_batch.prompt_version,
                    "created_at": db_batch.created_at.isoformat(),
                    "total_evaluations": db_batch.total_evaluations,
                    "hallucination_count": db_batch.hallucination_count,
                    "hallucination_rate": db_batch.hallucination_rate,
                    "average_scores": db_batch.average_scores or {}
                })
            
            return batches
        finally:
            session.close()
    
    def get_trends(
        self,
        model_name: Optional[str] = None,
        prompt_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get hallucination rate trends over time."""
        batches = self.get_batches(model_name=model_name, prompt_version=prompt_version, limit=1000)
        
        # Group by prompt_version and model_name
        trends = {}
        for batch in batches:
            key = f"{batch['model_name']}_{batch['prompt_version']}"
            if key not in trends:
                trends[key] = {
                    "model_name": batch["model_name"],
                    "prompt_version": batch["prompt_version"],
                    "data_points": []
                }
            
            trends[key]["data_points"].append({
                "timestamp": batch["created_at"],
                "hallucination_rate": batch["hallucination_rate"],
                "total_evaluations": batch["total_evaluations"]
            })
        
        # Sort data points by timestamp
        for key in trends:
            trends[key]["data_points"].sort(key=lambda x: x["timestamp"])
        
        return trends

