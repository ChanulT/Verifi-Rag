"""
Gemini LLM Client for Chat Service.

Alternative to OpenAI, provides same interface.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

import google.generativeai as genai

from app.config import settings_manager
from app.graph.state import QueryType
from app.llm.openai_client import (
    QueryAnalysisResult,
    ValidationResult,
    parse_query_analysis,
    parse_validation,
    QUERY_ANALYSIS_PROMPT,
    RAG_GENERATION_PROMPT,
    VALIDATION_PROMPT,
)

logger = logging.getLogger(__name__)

__all__ = ["GeminiClient", "get_gemini_client"]


class GeminiClient:
    """
    Gemini client for RAG operations.

    Same interface as OpenAIClient for easy swapping.
    """

    def __init__(
            self,
            api_key: str,
            model: str = "gemini-1.5-flash",
            temperature: float = 0.1,
            max_tokens: int = 2048,
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: Gemini API key
            model: Model to use
            temperature: Generation temperature
            max_tokens: Maximum tokens in response
        """
        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.info(f"Gemini client initialized: model={model}")

    async def analyze_query(
            self,
            query: str,
            chat_history: List[Dict[str, str]] = None,
    ) -> QueryAnalysisResult:
        """Analyze user query to understand intent."""
        history_str = ""
        if chat_history:
            history_str = "\n".join([
                f"{msg['role'].title()}: {msg['content'][:200]}"
                for msg in chat_history[-4:]
            ])

        prompt = QUERY_ANALYSIS_PROMPT.format(
            query=query,
            history=history_str or "None"
        )

        try:
            response = await self.model.generate_content_async(prompt)
            result = parse_query_analysis(response.text)
            return result

        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return QueryAnalysisResult(
                query_type=QueryType.FACTUAL,
                search_query=query,
                needs_retrieval=True,
                is_followup=False,
            )

    async def generate_answer(
            self,
            query: str,
            context: str,
            chat_history: List[Dict[str, str]] = None,
    ) -> Tuple[str, int]:
        """Generate answer with citations."""
        history_str = ""
        if chat_history:
            history_str = "\n".join([
                f"{msg['role'].title()}: {msg['content'][:300]}"
                for msg in chat_history[-6:]
            ])

        prompt = RAG_GENERATION_PROMPT.format(
            context=context,
            history=history_str or "None",
            query=query,
        )

        try:
            response = await self.model.generate_content_async(prompt)
            answer = response.text
            # Gemini doesn't provide token count easily
            tokens_used = len(answer.split()) * 2  # Rough estimate

            return answer, tokens_used

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            raise

    async def generate_no_context_response(
            self,
            query: str,
            reason: str = "no relevant documents found"
    ) -> str:
        """Generate response when no relevant context is found."""
        prompt = f"""The user asked: "{query}"

However, {reason}.

Generate a helpful response that:
1. Acknowledges you don't have the specific information
2. Suggests what documents might help (e.g., blood test reports)
3. Offers to help if they can provide more context

Keep the response brief and helpful."""

        try:
            response = await self.model.generate_content_async(prompt)
            return response.text

        except Exception as e:
            logger.error(f"No-context response generation failed: {e}")
            return f"I don't have enough information to answer your question about '{query}'."

    async def validate_answer(
            self,
            answer: str,
            context: str,
    ) -> ValidationResult:
        """Validate that answer is grounded in context."""
        prompt = VALIDATION_PROMPT.format(
            context=context,
            answer=answer,
        )

        try:
            response = await self.model.generate_content_async(prompt)
            result = parse_validation(response.text)
            return result

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return ValidationResult(
                is_grounded=True,
                confidence=0.5,
                issues=["validation_failed"],
            )

    async def health_check(self) -> bool:
        """Check if Gemini API is accessible."""
        try:
            response = await self.model.generate_content_async("test")
            return True
        except Exception as e:
            logger.error(f"Gemini health check failed: {e}")
            return False


# =============================================================================
# Singleton Factory
# =============================================================================

_gemini_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    """Get or create Gemini client singleton."""
    global _gemini_client

    if _gemini_client is None:
        settings = settings_manager.current

        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not configured")

        _gemini_client = GeminiClient(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            temperature=settings.openai_temperature,  # Reuse temp setting
            max_tokens=settings.openai_max_tokens,
        )

    return _gemini_client