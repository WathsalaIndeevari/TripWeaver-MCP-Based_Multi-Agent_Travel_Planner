"""FastAPI backend for TripWeaver: non-streaming and SSE-streaming chat endpoints."""

from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from pydantic import BaseModel

from agents.entity import AgentState
from agents.graph import graph

app = FastAPI(title="TripWeaver")

# In-memory session store: session_id -> AgentState
_SESSIONS: dict[str, AgentState] = {}

_ACTIVITY_LABELS = {
    "router": "Understanding your request...",
    "hotel": "Searching hotel suggestions...",
    "flight": "Searching flight options...",
    "general": "Thinking...",
}


class ChatRequest(BaseModel):
    message: str
    session_id: str


class ChatResponse(BaseModel):
    response: str


def _get_session_state(session_id: str) -> AgentState:
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = AgentState()
    return _SESSIONS[session_id]


def _last_ai_message_content(state: AgentState) -> str:
    for msg in reversed(state.messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Non-streaming fallback: run one turn end-to-end and return the final text."""
    current_state = _get_session_state(request.session_id)
    updated_state = current_state.model_copy(
        update={"messages": current_state.messages + [HumanMessage(content=request.message)]}
    )
    result = await graph.ainvoke(updated_state)
    new_state = AgentState(**result)
    _SESSIONS[request.session_id] = new_state
    return ChatResponse(response=_last_ai_message_content(new_state))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Streaming endpoint: emits activity, token, and final SSE events as the graph runs."""
    current_state = _get_session_state(request.session_id)
    updated_state = current_state.model_copy(
        update={"messages": current_state.messages + [HumanMessage(content=request.message)]}
    )

    async def event_generator():
        seen_nodes: set[str] = set()
        final_state_dict: dict | None = None

        async for event in graph.astream_events(updated_state, version="v2"):
            kind = event.get("event")
            node_name = event.get("metadata", {}).get("langgraph_node")

            if kind == "on_chain_start" and node_name in _ACTIVITY_LABELS and node_name not in seen_nodes:
                seen_nodes.add(node_name)
                yield _sse({"type": "activity", "label": _ACTIVITY_LABELS[node_name]})

            elif kind == "on_chat_model_stream":
                if node_name in ("general", "hotel", "flight"):
                    chunk = event.get("data", {}).get("chunk")
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        yield _sse({"type": "token", "content": chunk.content})

            elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                final_state_dict = event.get("data", {}).get("output")

        if final_state_dict is not None:
            new_state = AgentState(**final_state_dict)
            _SESSIONS[request.session_id] = new_state
            yield _sse({"type": "final", "content": _last_ai_message_content(new_state)})
        else:
            yield _sse({"type": "final", "content": "Sorry, something went wrong generating a response."})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}