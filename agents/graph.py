from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from agents.entity import AgentState, Intent
from agents.nodes import router_node, general_qa_node, hotel_node, flight_node


def _route_from_intent(state: AgentState) -> str:
    """Map the router's classified intent to the next node name."""
    if state.current_intent == Intent.HOTEL:
        return "hotel"
    if state.current_intent == Intent.FLIGHT:
        return "flight"
    return "general"


builder = StateGraph(AgentState)

builder.add_node("router", router_node)
builder.add_node("general", general_qa_node)
builder.add_node("hotel", hotel_node)
builder.add_node("flight", flight_node)

builder.set_entry_point("router")

builder.add_conditional_edges(
    "router",
    _route_from_intent,
    {
        "general": "general",
        "hotel": "hotel",
        "flight": "flight",
    },
)

builder.add_edge("general", END)
builder.add_edge("hotel", END)
builder.add_edge("flight", END)

graph = builder.compile()


if __name__ == "__main__":
    import asyncio

    async def _manual_test() -> None:
        result = await graph.ainvoke(
            AgentState(messages=[HumanMessage(content="find me a hotel in Tokyo")])
        )
        for msg in result["messages"]:
            print(f"[{msg.type}] {msg.content}")

    asyncio.run(_manual_test())
