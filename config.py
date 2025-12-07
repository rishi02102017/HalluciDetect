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
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./evaluation_results.db")
    
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

