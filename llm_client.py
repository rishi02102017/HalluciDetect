"""LLM client for generating outputs from various models."""
import os
from typing import Optional, Dict, Any, List
from openai import OpenAI
from anthropic import Anthropic

class LLMClient:
    """Client for interacting with different LLM providers."""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.openrouter_client = None
        
        # Initialize OpenRouter client (priority - supports multiple models)
        if os.getenv("OPENROUTER_API_KEY"):
            self.openrouter_client = OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1"
            )
        
        # Initialize OpenAI client if API key is available
        if os.getenv("OPENAI_API_KEY"):
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize Anthropic client if API key is available
        if os.getenv("ANTHROPIC_API_KEY"):
            self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def generate(
        self,
        prompt: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """
        Generate text from an LLM.
        
        Args:
            prompt: Input prompt
            model: Model name (e.g., 'gpt-4o-mini', 'claude-3-haiku-20240307')
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text
        """
        # Try OpenRouter first (supports all models via unified API)
        if self.openrouter_client:
            # Map common model names to OpenRouter model IDs
            openrouter_model = self._map_to_openrouter_model(model)
            
            response = self.openrouter_client.chat.completions.create(
                model=openrouter_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        
        # Direct OpenAI API
        elif model.startswith("gpt") or model.startswith("o1"):
            if not self.openai_client:
                raise ValueError("OpenAI API key not configured")
            
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        
        # Direct Anthropic API
        elif model.startswith("claude"):
            if not self.anthropic_client:
                raise ValueError("Anthropic API key not configured")
            
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.content[0].text
        
        else:
            raise ValueError(f"Unsupported model: {model}. No API client configured.")
    
    def _map_to_openrouter_model(self, model: str) -> str:
        """Map common model names to OpenRouter model IDs."""
        model_mapping = {
            # OpenAI models
            "gpt-4o": "openai/gpt-4o",
            "gpt-4o-mini": "openai/gpt-4o-mini",
            "gpt-4-turbo": "openai/gpt-4-turbo",
            "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
            # Anthropic models
            "claude-3-5-sonnet-20241022": "anthropic/claude-3.5-sonnet",
            "claude-3-opus-20240229": "anthropic/claude-3-opus",
            "claude-3-sonnet-20240229": "anthropic/claude-3-sonnet",
            "claude-3-haiku-20240307": "anthropic/claude-3-haiku",
            # Free/cheap models for testing
            "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct:free",
            "gemma-2-9b": "google/gemma-2-9b-it:free",
            "mistral-7b": "mistralai/mistral-7b-instruct:free",
        }
        return model_mapping.get(model, model)
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """List available models by provider."""
        models = {
            "openai": [],
            "anthropic": [],
            "free": []
        }
        
        # OpenRouter gives access to all models
        if self.openrouter_client:
            models["openai"] = [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-3.5-turbo"
            ]
            models["anthropic"] = [
                "claude-3-5-sonnet-20241022",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307"
            ]
            models["free"] = [
                "llama-3.1-8b",
                "gemma-2-9b",
                "mistral-7b"
            ]
        else:
            # Direct API access
            if self.openai_client:
                models["openai"] = [
                    "gpt-4o",
                    "gpt-4o-mini",
                    "gpt-4-turbo",
                    "gpt-3.5-turbo"
                ]
            
            if self.anthropic_client:
                models["anthropic"] = [
                    "claude-3-5-sonnet-20241022",
                    "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307"
                ]
        
        return models

