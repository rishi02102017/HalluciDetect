"""Database models and storage for evaluation results."""
import json
import numpy as np
import uuid
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import create_engine, Column, String, Float, Boolean, DateTime, Text, JSON, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from config import Config
from models import EvaluationResult, EvaluationBatch
from flask_login import UserMixin

Base = declarative_base()


class User(Base, UserMixin):
    """User model for authentication."""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(120), unique=True, nullable=False, index=True)
    username = Column(String(80), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Email verification
    email_verified = Column(Boolean, default=False)
    email_verification_token = Column(String(100), nullable=True)
    email_verification_sent_at = Column(DateTime, nullable=True)
    
    # Password reset
    password_reset_token = Column(String(100), nullable=True)
    password_reset_expires = Column(DateTime, nullable=True)
    
    # Relationships
    evaluations = relationship("EvaluationResultDB", back_populates="user", lazy="dynamic")
    templates = relationship("PromptTemplate", back_populates="user", lazy="dynamic")
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def generate_reset_token(self):
        """Generate a password reset token."""
        import secrets
        self.password_reset_token = secrets.token_urlsafe(32)
        self.password_reset_expires = datetime.utcnow() + timedelta(hours=1)
        return self.password_reset_token
    
    def verify_reset_token(self, token):
        """Verify a password reset token."""
        if not self.password_reset_token or not self.password_reset_expires:
            return False
        if self.password_reset_token != token:
            return False
        if datetime.utcnow() > self.password_reset_expires:
            return False
        return True
    
    def clear_reset_token(self):
        """Clear the password reset token."""
        self.password_reset_token = None
        self.password_reset_expires = None
    
    def generate_verification_token(self):
        """Generate an email verification token."""
        import secrets
        self.email_verification_token = secrets.token_urlsafe(32)
        self.email_verification_sent_at = datetime.utcnow()
        return self.email_verification_token
    
    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "username": self.username,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "email_verified": self.email_verified
        }


class PromptTemplate(Base):
    """Prompt template model for saving reusable templates."""
    __tablename__ = "prompt_templates"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=True)  # Null for system templates
    name = Column(String(100), nullable=False)
    description = Column(Text)
    category = Column(String(50), default="custom")  # qa, summarization, code, custom
    prompt_template = Column(Text, nullable=False)
    reference_template = Column(Text)  # Optional reference text template
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="templates")
    
    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "prompt_template": self.prompt_template,
            "reference_template": self.reference_template,
            "is_public": self.is_public,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

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
    user_id = Column(String, ForeignKey("users.id"), nullable=True)  # Null for anonymous evaluations
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
    
    # Relationships
    user = relationship("User", back_populates="evaluations")

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
        
        # Configure engine with connection pooling for PostgreSQL
        if self.database_url.startswith("postgresql://"):
            self.engine = create_engine(
                self.database_url,
                pool_size=Config.DB_POOL_SIZE,
                max_overflow=Config.DB_MAX_OVERFLOW,
                pool_recycle=Config.DB_POOL_RECYCLE,
                pool_pre_ping=True  # Verify connections before use
            )
        else:
            # SQLite doesn't support connection pooling
            self.engine = create_engine(self.database_url)
        
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self._run_migrations()
    
    def is_postgresql(self) -> bool:
        """Check if using PostgreSQL."""
        return self.database_url.startswith("postgresql://")
    
    def _run_migrations(self):
        """Run database migrations for new columns."""
        from sqlalchemy import text
        session = self.SessionLocal()
        try:
            # Check and add new user columns for password reset and email verification
            columns_to_add = [
                ("users", "email_verified", "BOOLEAN DEFAULT 0"),
                ("users", "email_verification_token", "VARCHAR(100)"),
                ("users", "email_verification_sent_at", "DATETIME"),
                ("users", "password_reset_token", "VARCHAR(100)"),
                ("users", "password_reset_expires", "DATETIME"),
            ]
            
            for table, column, column_type in columns_to_add:
                try:
                    # Try to add the column - will fail if it already exists
                    session.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}"))
                    session.commit()
                except Exception:
                    # Column already exists, ignore
                    session.rollback()
        finally:
            session.close()
    
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
    
    # ============ User Methods ============
    
    def create_user(self, email: str, username: str, password: str) -> Optional[User]:
        """Create a new user."""
        session = self.SessionLocal()
        try:
            # Check if user already exists
            existing = session.query(User).filter(
                (User.email == email) | (User.username == username)
            ).first()
            if existing:
                print(f"User already exists: {email} / {username}")
                return None
            
            user = User(email=email, username=username)
            user.set_password(password)
            session.add(user)
            session.commit()
            session.refresh(user)
            return user
        except Exception as e:
            session.rollback()
            print(f"Error creating user: {e}")
            return None
        finally:
            session.close()
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        session = self.SessionLocal()
        try:
            return session.query(User).filter(User.id == user_id).first()
        finally:
            session.close()
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        session = self.SessionLocal()
        try:
            return session.query(User).filter(User.email == email).first()
        finally:
            session.close()
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        session = self.SessionLocal()
        try:
            return session.query(User).filter(User.username == username).first()
        finally:
            session.close()
    
    def authenticate_user(self, email_or_username: str, password: str) -> Optional[User]:
        """Authenticate user by email/username and password."""
        session = self.SessionLocal()
        try:
            user = session.query(User).filter(
                (User.email == email_or_username) | (User.username == email_or_username)
            ).first()
            if user and user.check_password(password):
                return user
            return None
        finally:
            session.close()
    
    def update_user_password(self, user_id: str, new_password: str) -> bool:
        """Update user's password."""
        session = self.SessionLocal()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                user.set_password(new_password)
                user.clear_reset_token()
                session.commit()
                return True
            return False
        finally:
            session.close()
    
    def create_password_reset_token(self, email: str) -> Optional[str]:
        """Create a password reset token for a user."""
        session = self.SessionLocal()
        try:
            user = session.query(User).filter(User.email == email).first()
            if user:
                token = user.generate_reset_token()
                session.commit()
                return token
            return None
        finally:
            session.close()
    
    def verify_password_reset_token(self, token: str) -> Optional[User]:
        """Verify a password reset token and return the user."""
        session = self.SessionLocal()
        try:
            user = session.query(User).filter(User.password_reset_token == token).first()
            if user and user.verify_reset_token(token):
                return user
            return None
        finally:
            session.close()
    
    def reset_password_with_token(self, token: str, new_password: str) -> bool:
        """Reset password using a valid token."""
        session = self.SessionLocal()
        try:
            user = session.query(User).filter(User.password_reset_token == token).first()
            if user and user.verify_reset_token(token):
                user.set_password(new_password)
                user.clear_reset_token()
                session.commit()
                return True
            return False
        finally:
            session.close()
    
    def create_email_verification_token(self, user_id: str) -> Optional[str]:
        """Create an email verification token for a user."""
        session = self.SessionLocal()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                token = user.generate_verification_token()
                session.commit()
                return token
            return None
        finally:
            session.close()
    
    def verify_email_token(self, token: str) -> bool:
        """Verify an email using the token."""
        session = self.SessionLocal()
        try:
            user = session.query(User).filter(User.email_verification_token == token).first()
            if user:
                user.email_verified = True
                user.email_verification_token = None
                session.commit()
                return True
            return False
        finally:
            session.close()
    
    # ============ Template Methods ============
    
    def create_template(self, name: str, prompt_template: str, user_id: Optional[str] = None,
                       description: str = "", category: str = "custom",
                       reference_template: str = None, is_public: bool = False) -> PromptTemplate:
        """Create a new prompt template."""
        session = self.SessionLocal()
        try:
            template = PromptTemplate(
                name=name,
                user_id=user_id,
                description=description,
                category=category,
                prompt_template=prompt_template,
                reference_template=reference_template,
                is_public=is_public
            )
            session.add(template)
            session.commit()
            session.refresh(template)
            return template
        finally:
            session.close()
    
    def get_templates(self, user_id: Optional[str] = None, category: Optional[str] = None,
                     include_public: bool = True) -> List[Dict[str, Any]]:
        """Get prompt templates."""
        session = self.SessionLocal()
        try:
            query = session.query(PromptTemplate)
            
            if user_id:
                if include_public:
                    query = query.filter(
                        (PromptTemplate.user_id == user_id) | 
                        (PromptTemplate.is_public == True) |
                        (PromptTemplate.user_id == None)  # System templates
                    )
                else:
                    query = query.filter(PromptTemplate.user_id == user_id)
            else:
                # Only show public and system templates for anonymous users
                query = query.filter(
                    (PromptTemplate.is_public == True) | (PromptTemplate.user_id == None)
                )
            
            if category:
                query = query.filter(PromptTemplate.category == category)
            
            templates = query.order_by(PromptTemplate.created_at.desc()).all()
            return [t.to_dict() for t in templates]
        finally:
            session.close()
    
    def get_template_by_id(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get a template by ID."""
        session = self.SessionLocal()
        try:
            template = session.query(PromptTemplate).filter(PromptTemplate.id == template_id).first()
            return template.to_dict() if template else None
        finally:
            session.close()
    
    def delete_template(self, template_id: str, user_id: str) -> bool:
        """Delete a template (only if owned by user)."""
        session = self.SessionLocal()
        try:
            template = session.query(PromptTemplate).filter(
                PromptTemplate.id == template_id,
                PromptTemplate.user_id == user_id
            ).first()
            if template:
                session.delete(template)
                session.commit()
                return True
            return False
        finally:
            session.close()
    
    def save_result_with_user(self, result: EvaluationResult, user_id: Optional[str] = None) -> None:
        """Save an evaluation result with user association."""
        session = self.SessionLocal()
        try:
            db_result = EvaluationResultDB(
                id=result.id,
                user_id=user_id,
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
    
    def get_user_results(self, user_id: str, limit: int = 100) -> List[EvaluationResult]:
        """Get evaluation results for a specific user."""
        session = self.SessionLocal()
        try:
            query = session.query(EvaluationResultDB).filter(
                EvaluationResultDB.user_id == user_id
            ).order_by(EvaluationResultDB.timestamp.desc()).limit(limit)
            
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
    
    def init_default_templates(self):
        """Initialize default system templates."""
        session = self.SessionLocal()
        try:
            # Check if templates already exist
            existing = session.query(PromptTemplate).filter(PromptTemplate.user_id == None).count()
            if existing > 0:
                return
            
            default_templates = [
                {
                    "name": "Q&A Accuracy Test",
                    "description": "Test question-answering accuracy with a reference answer",
                    "category": "qa",
                    "prompt_template": "Question: {question}\n\nPlease provide a detailed answer.",
                    "reference_template": "{reference_answer}",
                },
                {
                    "name": "Summarization Evaluation",
                    "description": "Evaluate text summarization quality",
                    "category": "summarization",
                    "prompt_template": "Please summarize the following text:\n\n{text}",
                    "reference_template": "{reference_summary}",
                },
                {
                    "name": "Fact Verification",
                    "description": "Verify factual claims in LLM responses",
                    "category": "fact_check",
                    "prompt_template": "Please provide information about: {topic}",
                    "reference_template": "{verified_facts}",
                },
                {
                    "name": "Code Generation Test",
                    "description": "Test code generation accuracy",
                    "category": "code",
                    "prompt_template": "Write a {language} function that {task_description}",
                    "reference_template": "{reference_code}",
                },
                {
                    "name": "Translation Quality",
                    "description": "Evaluate translation accuracy",
                    "category": "translation",
                    "prompt_template": "Translate the following text from {source_lang} to {target_lang}:\n\n{text}",
                    "reference_template": "{reference_translation}",
                },
            ]
            
            for t in default_templates:
                template = PromptTemplate(
                    name=t["name"],
                    description=t["description"],
                    category=t["category"],
                    prompt_template=t["prompt_template"],
                    reference_template=t.get("reference_template"),
                    is_public=True,
                    user_id=None  # System template
                )
                session.add(template)
            
            session.commit()
        finally:
            session.close()

