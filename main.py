"""Minimal FastAPI wrapper around the TripWeaver LangGraph agent graph."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from fastapi import FastAPI
from pydantic import BaseModel

from agents.entity import AgentState
from agents.graph import graph

app = FastAPI(title="TripWeaver")

# In-memory session store: session_id -> AgentState
# NOTE: this is process-local and resets on restart. Fine for now,
# swap for a real store (Redis/DB) if session persistence matters later.
_SESSIONS: dict[str, AgentState] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str


class ChatResponse(BaseModel):
    response: str


def _get_session_state(session_id: str) -> AgentState:
    """Fetch existing session state or create a fresh one."""
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = AgentState()
    return _SESSIONS[session_id]


def _last_ai_message_content(state: AgentState) -> str:
    """Return the content of the most recent AIMessage in state.messages."""
    for msg in reversed(state.messages):
        if isinstance(msg, AIMessage):
            return msg.content
    return ""


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Run one turn of the conversation through the compiled LangGraph graph."""
    current_state = _get_session_state(request.session_id)

    # Append the new user message to the session's existing state
    updated_state = current_state.model_copy(
        update={
            "messages": current_state.messages + [HumanMessage(content=request.message)]
        }
    )

    result = await graph.ainvoke(updated_state)

    # graph.ainvoke returns a dict (LangGraph's output shape); rehydrate into AgentState
    new_state = AgentState(**result)
    _SESSIONS[request.session_id] = new_state

    return ChatResponse(response=_last_ai_message_content(new_state))


@app.get("/health")
async def health() -> dict:
    """Basic liveness check."""
    return {"status": "ok"}
