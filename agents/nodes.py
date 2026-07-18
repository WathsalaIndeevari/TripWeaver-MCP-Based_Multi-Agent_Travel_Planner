from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage

from agents.entity import (
    AgentState,
    Booking,
    BookingItemType,
    BookingStatus,
    ToolCallStatus,
)
from agents.llm import get_llm, get_router_llm
from agents.prompts import (
    FLIGHT_AGENT_SYSTEM_PROMPT,
    GENERAL_QA_SYSTEM_PROMPT,
    HOTEL_AGENT_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    RouterOutput,
)
from agents.tools import get_mcp_tools, get_tool_by_name


def _parse_mcp_result(result: Any) -> Any:
    """Unwrap langchain-mcp-adapters' raw MCP content blocks into plain Python data.

    Tool results come back as a list of content blocks like
    [{"type": "text", "text": "<json string>"}, ...] -- sometimes one block
    per item (e.g. one per hotel), sometimes a single block containing a
    JSON array. This decodes each block independently and never
    string-concatenates raw JSON fragments.
    """
    if (
        isinstance(result, list)
        and result
        and isinstance(result[0], dict)
        and "text" in result[0]
    ):
        parsed_items = []
        for block in result:
            if block.get("type") != "text":
                continue
            try:
                decoded = json.loads(block["text"])
            except json.JSONDecodeError:
                decoded = block["text"]
            # A block can itself decode to a list (already an array) or a single object
            if isinstance(decoded, list):
                parsed_items.extend(decoded)
            else:
                parsed_items.append(decoded)

        if len(parsed_items) == 1:
            return parsed_items[0]
        return parsed_items

    return result


def _format_hotel_results(hotels: list[dict]) -> str:
    """Turn a list of hotel dicts into a readable bullet list for the chat message."""
    if not hotels:
        return "I couldn't find any hotels matching that search."
    lines = [
        f"- {h['name']} ({h['location']}) — ${h['price_per_night']}/night, {h['rating']}★"
        for h in hotels
    ]
    return "Here's what I found:\n" + "\n".join(lines)


def _format_flight_results(flights: list[dict]) -> str:
    """Turn a list of flight dicts into a readable bullet list for the chat message."""
    if not flights:
        return "I couldn't find any flights matching that search."
    lines = [
        f"- {f['airline']} {f['flight_number']}: {f['origin']} → {f['destination']}, ${f['price']}"
        for f in flights
    ]
    return "Here's what I found:\n" + "\n".join(lines)


# 1. Router


async def router_node(state: AgentState) -> dict:
    router_llm = get_router_llm().with_structured_output(RouterOutput)
    result: RouterOutput = await router_llm.ainvoke(
        [SystemMessage(content=ROUTER_SYSTEM_PROMPT), *state.messages]
    )
    extracted = {
        k: v for k, v in result.model_dump().items() if k != "intent" and v is not None
    }
    return {
        "current_intent": result.intent,
        "extracted_details": {**state.extracted_details, **extracted},
    }


# 2. General QA


async def general_qa_node(state: AgentState) -> dict:
    """Answer general travel questions conversationally. No tool use."""
    llm = get_llm()
    response = await llm.ainvoke(
        [SystemMessage(content=GENERAL_QA_SYSTEM_PROMPT), *state.messages]
    )
    return {"messages": [AIMessage(content=response.content)]}


# 3. Hotel missing-field check (plain Python, no LLM)


def check_hotel_missing_fields(state: AgentState) -> list[str]:
    """Return required-but-missing hotel fields based on extracted_details.

    Rule-based only -- no LLM calls. If the user appears to be booking
    (action == "book" or a hotel_id is already present) a hotel_id is
    required. Otherwise, treat it as a search and require a location.
    """
    details = state.extracted_details or {}
    is_booking = details.get("action") == "book" or details.get("hotel_id") is not None

    if is_booking:
        return [] if details.get("hotel_id") else ["hotel_id"]

    return [] if details.get("location") else ["location"]


# 4. Hotel node
def _hotel_clarifying_question(missing_fields: list[str]) -> str:
    """Build a clarifying question naming the missing hotel field."""
    if "location" in missing_fields:
        return "Sure -- which city or area would you like to search hotels in?"
    return "Which hotel would you like to book? Please give me the hotel ID."


async def _invoke_hotel_agent(state: AgentState):
    """Bind MCP hotel tools to the LLM and get a response for the current turn."""
    tools = await get_mcp_tools()
    llm = get_llm().bind_tools(tools)
    context = SystemMessage(
        content=f"{HOTEL_AGENT_SYSTEM_PROMPT}\n\nPrior hotel search results: {state.hotel_results}"
    )
    return await llm.ainvoke([context, *state.messages]), tools


async def _execute_hotel_tool_call(ai_response, tools: list, state: AgentState) -> dict:
    """Run the single tool call the hotel agent requested, updating state safely."""
    call = ai_response.tool_calls[0]
    tool = get_tool_by_name(tools, call["name"])

    try:
        raw_result = await tool.ainvoke(call["args"])
        result = _parse_mcp_result(raw_result)
    except Exception as e:  # noqa: BLE001 -- tool failures must never crash the graph
        return {
            "last_tool_status": ToolCallStatus.FAILED,
            "last_tool_error": str(e),
            "messages": [
                AIMessage(
                    content="I couldn't complete that hotel request right now, "
                    "please try again shortly."
                )
            ],
        }

    if call["name"] == "book_hotel":
        booking = Booking(
            confirmation_id=str(uuid.uuid4())[:8],
            item_type=BookingItemType.HOTEL,
            item_id=call["args"].get("hotel_id", ""),
            total_cost=result.get("total_cost", 0) if isinstance(result, dict) else 0,
            booking_status=BookingStatus.CONFIRMED,
        )
        return {
            "last_tool_status": ToolCallStatus.SUCCEEDED,
            "bookings": state.bookings + [booking],
            "messages": [
                AIMessage(content=f"Booked! Confirmation ID: {booking.confirmation_id}")
            ],
        }

    # search_hotels / list_hotels
    return {
        "last_tool_status": ToolCallStatus.SUCCEEDED,
        "hotel_results": result,
        "messages": [AIMessage(content=_format_hotel_results(result))],
    }


async def hotel_node(state: AgentState) -> dict:
    """Handle a hotel search/booking turn, asking for missing info first if needed."""
    missing = check_hotel_missing_fields(state)
    if missing:
        return {
            "missing_fields": missing,
            "messages": [AIMessage(content=_hotel_clarifying_question(missing))],
        }

    ai_response, tools = await _invoke_hotel_agent(state)
    if not ai_response.tool_calls:
        return {
            "missing_fields": [],
            "messages": [AIMessage(content=ai_response.content)],
        }

    return {
        "missing_fields": [],
        **await _execute_hotel_tool_call(ai_response, tools, state),
    }


# 5. Flight missing-field check + flight node (same pattern as hotel)


def check_flight_missing_fields(state: AgentState) -> list[str]:
    """Return required-but-missing flight fields based on extracted_details.

    Rule-based only -- no LLM calls. Booking requires flight_id; searching
    requires both origin and destination.
    """
    details = state.extracted_details or {}
    is_booking = details.get("action") == "book" or details.get("flight_id") is not None

    if is_booking:
        return [] if details.get("flight_id") else ["flight_id"]

    missing = []
    if not details.get("origin"):
        missing.append("origin")
    if not details.get("destination"):
        missing.append("destination")
    return missing


def _flight_clarifying_question(missing_fields: list[str]) -> str:
    """Build a clarifying question naming the missing flight field(s)."""
    if "flight_id" in missing_fields:
        return "Which flight would you like to book? Please give me the flight ID."
    if "origin" in missing_fields and "destination" in missing_fields:
        return "Where are you flying from, and where to?"
    if "origin" in missing_fields:
        return "Which city are you flying from?"
    return "Which city are you flying to?"


async def _invoke_flight_agent(state: AgentState):
    """Bind MCP flight tools to the LLM and get a response for the current turn."""
    tools = await get_mcp_tools()
    llm = get_llm().bind_tools(tools)
    context = SystemMessage(
        content=f"{FLIGHT_AGENT_SYSTEM_PROMPT}\n\nPrior flight search results: {state.flight_results}"
    )
    return await llm.ainvoke([context, *state.messages]), tools


async def _execute_flight_tool_call(
    ai_response, tools: list, state: AgentState
) -> dict:
    """Run the single tool call the flight agent requested, updating state safely."""
    call = ai_response.tool_calls[0]
    tool = get_tool_by_name(tools, call["name"])

    try:
        raw_result = await tool.ainvoke(call["args"])
        result = _parse_mcp_result(raw_result)
    except Exception as e:  # noqa: BLE001 -- tool failures must never crash the graph
        return {
            "last_tool_status": ToolCallStatus.FAILED,
            "last_tool_error": str(e),
            "messages": [
                AIMessage(
                    content="I couldn't complete that flight request right now, "
                    "please try again shortly."
                )
            ],
        }

    if call["name"] == "book_flight":
        booking = Booking(
            confirmation_id=str(uuid.uuid4())[:8],
            item_type=BookingItemType.FLIGHT,
            item_id=call["args"].get("flight_id", ""),
            total_cost=result.get("total_cost", 0) if isinstance(result, dict) else 0,
            booking_status=BookingStatus.CONFIRMED,
        )
        return {
            "last_tool_status": ToolCallStatus.SUCCEEDED,
            "bookings": state.bookings + [booking],
            "messages": [
                AIMessage(content=f"Booked! Confirmation ID: {booking.confirmation_id}")
            ],
        }

    # search_flights / list_flights
    return {
        "last_tool_status": ToolCallStatus.SUCCEEDED,
        "flight_results": result,
        "messages": [AIMessage(content=_format_flight_results(result))],
    }


async def flight_node(state: AgentState) -> dict:
    """Handle a flight search/booking turn, asking for missing info first if needed."""
    missing = check_flight_missing_fields(state)
    if missing:
        return {
            "missing_fields": missing,
            "messages": [AIMessage(content=_flight_clarifying_question(missing))],
        }

    ai_response, tools = await _invoke_flight_agent(state)
    if not ai_response.tool_calls:
        return {
            "missing_fields": [],
            "messages": [AIMessage(content=ai_response.content)],
        }

    return {
        "missing_fields": [],
        **await _execute_flight_tool_call(ai_response, tools, state),
    }
