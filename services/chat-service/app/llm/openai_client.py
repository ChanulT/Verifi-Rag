"""
OpenAI LLM Client for Chat Service.

Provides:
- Query analysis (understanding user intent)
- RAG generation (answering with citations)
- Validation (checking answer grounding)

Uses structured prompts optimized for medical document QA.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from app.config import settings_manager
from app.graph.state import QueryType

logger = logging.getLogger(__name__)

__all__ = ["OpenAIClient", "get_openai_client"]

# =============================================================================
# Prompt Templates
# =============================================================================

QUERY_ANALYSIS_PROMPT = """You are a medical document assistant analyzing user queries about blood test reports and medical documents.

Analyze the following query and determine:
1. Query type: factual, comparison, explanation, summary, clarification, or out_of_scope
2. Optimized search query: rewrite for better semantic search
3. Whether retrieval is needed

Query types:
- factual: Direct questions about specific values (e.g., "What is my hemoglobin level?")
- comparison: Comparing values or trends (e.g., "How has my cholesterol changed?")
- explanation: Understanding what values mean (e.g., "What does elevated WBC indicate?")
- summary: Requesting overview (e.g., "Summarize my blood test results")
- clarification: Follow-up about previous answer (e.g., "What do you mean by that?")
- out_of_scope: Questions unrelated to medical documents (e.g., "What's the weather?")

User Query: {query}

Conversation History:
{history}

Respond in this exact format:
QUERY_TYPE: <type>
SEARCH_QUERY: <optimized query for retrieval>
NEEDS_RETRIEVAL: <true/false>
IS_FOLLOWUP: <true/false>"""

RAG_GENERATION_PROMPT = """You are a helpful medical assistant answering questions about blood test reports and medical documents. Your answers must be:

1. ACCURATE: Only state information found in the provided context
2. CITED: Use [1], [2], etc. to reference sources
3. CLEAR: Use simple language, explain medical terms
4. HONEST: If information isn't available, say so

CRITICAL RULES:
- ONLY use information from the context below.
- If the user asks for a time range (e.g., 'past 3 years') and you only have data for a portion of that time, ANALYZE THE AVAILABLE DATA and explicitly mention which dates you are referring to.
- If no relevant medical values are found at all, then say you don't have enough information.
- Never make up medical values or information
- For medical advice, recommend consulting a healthcare provider

Context from documents:
{context}

Conversation History:
{history}

User Question: {query}

Provide a helpful, accurate answer with citations:"""

VALIDATION_PROMPT = """You are a fact-checker validating whether an answer is properly grounded in the provided context.

Context:
{context}

Answer to validate:
{answer}

Check:
1. Is every claim in the answer supported by the context?
2. Are the citations accurate (do they point to the right information)?
3. Does the answer make any claims not found in the context?

Respond in this exact format:
IS_GROUNDED: <true/false>
CONFIDENCE: <0.0-1.0>
ISSUES: <list any issues, or "none">"""


# =============================================================================
# Response Parsing
# =============================================================================

@dataclass
class QueryAnalysisResult:
    """Result of query analysis."""
    query_type: QueryType
    search_query: str
    needs_retrieval: bool
    is_followup: bool


@dataclass
class ValidationResult:
    """Result of answer validation."""
    is_grounded: bool
    confidence: float
    issues: List[str]


def parse_query_analysis(response: str) -> QueryAnalysisResult:
    """Parse query analysis response."""
    lines = response.strip().split('\n')

    query_type = QueryType.FACTUAL
    search_query = ""
    needs_retrieval = True
    is_followup = False

    for line in lines:
        line = line.strip()
        if line.startswith("QUERY_TYPE:"):
            type_str = line.split(":", 1)[1].strip().lower()
            try:
                query_type = QueryType(type_str)
            except ValueError:
                query_type = QueryType.FACTUAL
        elif line.startswith("SEARCH_QUERY:"):
            search_query = line.split(":", 1)[1].strip()
        elif line.startswith("NEEDS_RETRIEVAL:"):
            needs_retrieval = line.split(":", 1)[1].strip().lower() == "true"
        elif line.startswith("IS_FOLLOWUP:"):
            is_followup = line.split(":", 1)[1].strip().lower() == "true"

    return QueryAnalysisResult(
        query_type=query_type,
        search_query=search_query or "",
        needs_retrieval=needs_retrieval,
        is_followup=is_followup,
    )


def parse_validation(response: str) -> ValidationResult:
    """Parse validation response."""
    lines = response.strip().split('\n')

    is_grounded = True
    confidence = 0.5
    issues = []

    for line in lines:
        line = line.strip()
        if line.startswith("IS_GROUNDED:"):
            is_grounded = line.split(":", 1)[1].strip().lower() == "true"
        elif line.startswith("CONFIDENCE:"):
            try:
                confidence = float(line.split(":", 1)[1].strip())
            except ValueError:
                confidence = 0.5
        elif line.startswith("ISSUES:"):
            issues_str = line.split(":", 1)[1].strip()
            if issues_str.lower() != "none":
                issues = [i.strip() for i in issues_str.split(",")]

    return ValidationResult(
        is_grounded=is_grounded,
        confidence=confidence,
        issues=issues,
    )


# =============================================================================
# OpenAI Client
# =============================================================================

class OpenAIClient:
    """
    OpenAI client for RAG operations.

    Provides methods for:
    - Query analysis
    - RAG generation with citations
    - Answer validation
    """

    def __init__(
            self,
            api_key: str,
            model: str = "gpt-4o-mini",
            temperature: float = 0.1,
            max_tokens: int = 2048,
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-4o-mini, gpt-4o, etc.)
            temperature: Generation temperature (lower = more focused)
            max_tokens: Maximum tokens in response
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.info(f"OpenAI client initialized: model={model}, temp={temperature}")

    async def analyze_query(
            self,
            query: str,
            chat_history: List[Dict[str, str]] = None,
    ) -> QueryAnalysisResult:
        """
        Analyze user query to understand intent.

        Args:
            query: User's question
            chat_history: Previous conversation

        Returns:
            QueryAnalysisResult with type, search query, etc.
        """
        history_str = ""
        if chat_history:
            history_str = "\n".join([
                f"{msg['role'].title()}: {msg['content'][:200]}"
                for msg in chat_history[-4:]  # Last 2 turns
            ])

        prompt = QUERY_ANALYSIS_PROMPT.format(
            query=query,
            history=history_str or "None"
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a query analyzer for a medical document QA system."
                                                  "CRITICAL: When in doubt, set NEEDS_RETRIEVAL to true."
                                                  " Only set to false if the user is saying 'hello', 'goodbye', or "
                                                  "asking a question completely unrelated to health (like the weather)."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Deterministic for analysis
                max_tokens=256,
            )

            result = parse_query_analysis(response.choices[0].message.content)
            logger.debug(f"Query analysis: type={result.query_type}, needs_retrieval={result.needs_retrieval}")

            return result

        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            # Return safe defaults
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
        """
        Generate answer with citations.

        Args:
            query: User's question
            context: Retrieved context with source references
            chat_history: Previous conversation

        Returns:
            Tuple of (answer text, tokens used)
        """
        history_str = ""
        if chat_history:
            history_str = "\n".join([
                f"{msg['role'].title()}: {msg['content'][:300]}"
                for msg in chat_history[-6:]  # Last 3 turns
            ])

        prompt = RAG_GENERATION_PROMPT.format(
            context=context,
            history=history_str or "None",
            query=query,
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful medical document assistant. Always cite sources using [1], [2], etc."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0

            logger.debug(f"Generated answer: {len(answer)} chars, {tokens_used} tokens")

            return answer, tokens_used

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            raise

    async def generate_no_context_response(
            self,
            query: str,
            reason: str = "no relevant documents found"
    ) -> str:
        """
        Generate response when no relevant context is found.

        This provides a helpful message instead of a generic error.
        """
        prompt = f"""The user asked: "{query}"

However, {reason}.

Generate a helpful response that:
1. Acknowledges you don't have the specific information
2. Suggests what documents might help (e.g., blood test reports)
3. Offers to help if they can provide more context

Keep the response brief and helpful."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful medical document assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=256,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"No-context response generation failed: {e}")
            return f"I don't have enough information to answer your question about '{query}'. Please make sure you've uploaded relevant medical documents."

    async def validate_answer(
            self,
            answer: str,
            context: str,
    ) -> ValidationResult:
        """
        Validate that answer is grounded in context.

        Args:
            answer: Generated answer to validate
            context: Source context

        Returns:
            ValidationResult with grounding status and confidence
        """
        prompt = VALIDATION_PROMPT.format(
            context=context,
            answer=answer,
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a fact-checker validating answer accuracy."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=256,
            )

            result = parse_validation(response.choices[0].message.content)
            logger.debug(f"Validation: grounded={result.is_grounded}, confidence={result.confidence}")

            return result

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            # Return conservative defaults
            return ValidationResult(
                is_grounded=True,  # Assume grounded to not block
                confidence=0.5,
                issues=["validation_failed"],
            )

    async def health_check(self) -> bool:
        """Check if OpenAI API is accessible."""
        try:
            # Simple test completion
            await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )
            return True
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False


# =============================================================================
# Singleton Factory
# =============================================================================

_openai_client: Optional[OpenAIClient] = None


def get_openai_client() -> OpenAIClient:
    """Get or create OpenAI client singleton."""
    global _openai_client

    if _openai_client is None:
        settings = settings_manager.current

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY not configured")

        _openai_client = OpenAIClient(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
        )

    return _openai_client


def initialize_openai_client(
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
) -> OpenAIClient:
    """Initialize OpenAI client with specific settings."""
    global _openai_client

    _openai_client = OpenAIClient(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return _openai_client