"""Configuration settings for the LLM Evaluation system."""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration."""
    
    # API Keys
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    FACTCHECK_API_KEY = os.getenv("FACTCHECK_API_KEY", "")
    FACTCHECK_API_URL = os.getenv("FACTCHECK_API_URL", "https://api.factcheck.org/v1/check")
    
    # Database - PostgreSQL for production, SQLite for local development
    # Render provides DATABASE_URL with postgres:// prefix, convert to postgresql://
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./evaluation_results.db")
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    
    # Connection Pooling settings
    DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
    DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))
    
    # Model defaults
    DEFAULT_LLM_MODEL = "gpt-4o-mini"
    DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Evaluation thresholds
    SEMANTIC_SIMILARITY_THRESHOLD = 0.7
    FACT_CHECK_CONFIDENCE_THRESHOLD = 0.8
    HALLUCINATION_SCORE_THRESHOLD = 0.5
    
    # Flask settings
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    
    # JWT Settings
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", SECRET_KEY)
    JWT_ACCESS_TOKEN_EXPIRES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES", "3600"))  # 1 hour
    JWT_REFRESH_TOKEN_EXPIRES = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRES", "604800"))  # 7 days
    
    # Knowledge Base
    KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "./knowledge_base")
