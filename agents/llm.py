"""LLM initialization for TripWeaver agents."""

from __future__ import annotations

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

_MODEL = "gpt-4o-mini"


def get_llm(temperature: float = 0) -> ChatOpenAI:
    """Return a ChatOpenAI instance for general agent use."""
    return ChatOpenAI(model=_MODEL, temperature=temperature)


def get_router_llm() -> ChatOpenAI:
    """Return a ChatOpenAI instance for structured intent classification."""
    return ChatOpenAI(model=_MODEL, temperature=0)
