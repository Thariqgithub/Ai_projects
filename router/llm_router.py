from typing import Optional, AsyncGenerator

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    raise ImportError(
        "Could not import 'google.genai'. Run fix_env.bat or:\n"
        "  pip uninstall google-generativeai -y && pip install 'google-genai>=1.5.0'"
    )
from groq import Groq
from openai import AsyncOpenAI  # used for async Groq streaming (OpenAI-compat endpoint)

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Single shared Gemini client
_gemini_client = genai.Client(api_key=settings.GOOGLE_API_KEY)


class LLMRouter:
    """
    Routes LLM calls to Groq (llama-4-scout) or Gemini (fallback).

    Providers:
        groq   — meta-llama/llama-4-scout-17b-16e-instruct via Groq LPU (default)
        gemini — gemini-2.0-flash via google-genai SDK (fallback / vision tasks)
    """

    def __init__(self, default_model: str = settings.DEFAULT_LLM):
        self.default_model = default_model
        self._groq: Optional[Groq] = None
        self._groq_async: Optional[AsyncOpenAI] = None

    # ------------------------------------------------------------------ #
    # Lazy clients                                                         #
    # ------------------------------------------------------------------ #

    @property
    def groq_client(self) -> Groq:
        if self._groq is None:
            self._groq = Groq(api_key=settings.GROQ_API_KEY)
        return self._groq

    @property
    def groq_async_client(self) -> AsyncOpenAI:
        """Groq exposes an OpenAI-compatible async endpoint."""
        if self._groq_async is None:
            self._groq_async = AsyncOpenAI(
                api_key=settings.GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
            )
        return self._groq_async

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """
        Synchronous generation.
        model: 'groq' | 'gemini'  (defaults to DEFAULT_LLM env var)
        """
        selected = model or self._auto_route(prompt)
        logger.info(f"LLM routing to: {selected}")

        if selected == "gemini":
            return self._call_gemini(prompt, system_prompt, temperature, max_tokens)
        else:
            return self._call_groq(prompt, system_prompt, temperature, max_tokens)

    async def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """Async token streaming generator for SSE responses."""
        selected = model or self._auto_route(prompt)
        logger.info(f"Streaming from: {selected}")

        if selected == "gemini":
            async for chunk in self._stream_gemini(prompt, system_prompt, temperature, max_tokens):
                yield chunk
        else:
            async for chunk in self._stream_groq(prompt, system_prompt, temperature, max_tokens):
                yield chunk

    # ------------------------------------------------------------------ #
    # Groq — llama-4-scout                                                 #
    # ------------------------------------------------------------------ #

    def _groq_messages(self, prompt: str, system_prompt: Optional[str]) -> list[dict]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _call_groq(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        resp = self.groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=self._groq_messages(prompt, system_prompt),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    async def _stream_groq(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ):
        stream = await self.groq_async_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=self._groq_messages(prompt, system_prompt),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ------------------------------------------------------------------ #
    # Gemini — gemini-2.0-flash (new google-genai SDK)                    #
    # ------------------------------------------------------------------ #

    def _call_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt or "You are a helpful assistant.",
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        response = _gemini_client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config=config,
        )
        return response.text

    async def _stream_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ):
        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt or "You are a helpful assistant.",
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        # new SDK stream: iterate over chunks from generate_content_stream
        for chunk in _gemini_client.models.generate_content_stream(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config=config,
        ):
            if chunk.text:
                yield chunk.text

    # ------------------------------------------------------------------ #
    # Auto-routing heuristics                                              #
    # ------------------------------------------------------------------ #

    def _auto_route(self, prompt: str) -> str:
        """
        Always route to Groq (llama-4-scout) — free, fast, no quota issues.
        Gemini generation disabled to avoid free-tier quota errors.
        """
        return "groq"