from __future__ import annotations

import json
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from agents.entity import AgentState, Intent, Hotel, Flight, Booking, ToolCallStatus
from agents.prompts import (
    RouterOutput,
    ROUTER_SYSTEM_PROMPT,
    GENERAL_QA_SYSTEM_PROMPT,
    HOTEL_AGENT_SYSTEM_PROMPT,
    FLIGHT_AGENT_SYSTEM_PROMPT,
)
from agents.llm import get_llm, get_router_llm
from agents.tools import get_mcp_tools, get_tool_by_name


async def router_node(state: AgentState) -> dict:
    """Classify the user's latest message into an intent using structured output."""
    llm = get_router_llm().with_structured_output(RouterOutput)
    prompt = [SystemMessage(content=ROUTER_SYSTEM_PROMPT)] + state.messages
    result = await llm.ainvoke(prompt)
    return {"current_intent": result.intent}


async def general_qa_node(state: AgentState) -> dict:
    """Answer general travel questions conversationally without tools."""
    llm = get_llm()
    prompt = [SystemMessage(content=GENERAL_QA_SYSTEM_PROMPT)] + state.messages
    response = await llm.ainvoke(prompt)
    return {"messages": [AIMessage(content=response.content)]}


def check_hotel_missing_fields(state: AgentState) -> list[str]:
    """Check if required fields are missing in state.extracted_details for hotels."""
    last_msg = ""
    if state.messages:
        last_msg = str(state.messages[-1].content).lower()

    is_booking = "book" in last_msg or "reserve" in last_msg
    if is_booking:
        # Check if booking relative to prior results (e.g., "book the second one")
        is_relative = any(w in last_msg for w in ["second", "first", "third", "last", "it", "option"])
        if is_relative and state.hotel_results:
            return []
        if not state.extracted_details.get("hotel_id"):
            return ["hotel_id"]
    else:
        # Require location if searching
        is_search = any(w in last_msg for w in ["search", "find", "show", "look", "hotel", "room"])
        if is_search and not state.extracted_details.get("location"):
            return ["location"]
    return []


async def hotel_node(state: AgentState) -> dict:
    """Process hotel queries by performing validation and executing tools if needed."""
    missing = check_hotel_missing_fields(state)
    if missing:
        return _handle_missing_fields("hotel", missing)

    llm = get_llm()
    tools = await get_mcp_tools()
    llm_with_tools = llm.bind_tools(tools)

    context = f"{HOTEL_AGENT_SYSTEM_PROMPT}\n\nPrior hotel search results in state: {state.hotel_results}"
    prompt = [SystemMessage(content=context)] + state.messages
    response = await llm_with_tools.ainvoke(prompt)

    if not response.tool_calls:
        return {"messages": [response]}

    return await _execute_hotel_tool(response, tools, prompt, state, llm)


def check_flight_missing_fields(state: AgentState) -> list[str]:
    """Check if required fields are missing in state.extracted_details for flights."""
    last_msg = ""
    if state.messages:
        last_msg = str(state.messages[-1].content).lower()

    is_booking = "book" in last_msg or "reserve" in last_msg
    if is_booking:
        is_relative = any(w in last_msg for w in ["second", "first", "third", "last", "it", "option"])
        if is_relative and state.flight_results:
            return []
        if not state.extracted_details.get("flight_id"):
            return ["flight_id"]
    else:
        is_search = any(w in last_msg for w in ["search", "find", "show", "look", "flight", "fly", "to", "from"])
        if is_search:
            missing = []
            if not state.extracted_details.get("origin"):
                missing.append("origin")
            if not state.extracted_details.get("destination"):
                missing.append("destination")
            return missing
    return []


async def flight_node(state: AgentState) -> dict:
    """Process flight queries by performing validation and executing tools if needed."""
    missing = check_flight_missing_fields(state)
    if missing:
        return _handle_missing_fields("flight", missing)

    llm = get_llm()
    tools = await get_mcp_tools()
    llm_with_tools = llm.bind_tools(tools)

    context = f"{FLIGHT_AGENT_SYSTEM_PROMPT}\n\nPrior flight search results in state: {state.flight_results}"
    prompt = [SystemMessage(content=context)] + state.messages
    response = await llm_with_tools.ainvoke(prompt)

    if not response.tool_calls:
        return {"messages": [response]}

    return await _execute_flight_tool(response, tools, prompt, state, llm)


def _handle_missing_fields(agent_type: str, missing: list[str]) -> dict:
    """Generate the partial state update and clarifying question for missing fields."""
    if agent_type == "hotel":
        field = missing[0]
        question = (
            "Please specify the location you would like to search for hotels in."
            if field == "location"
            else "Please provide the hotel ID you want to book."
        )
    else:
        if "origin" in missing and "destination" in missing:
            question = "Please specify both the origin and destination for your flight search."
        elif "origin" in missing:
            question = "Please specify the origin city or airport for your flight."
        elif "destination" in missing:
            question = "Please specify the destination city or airport for your flight."
        else:
            question = "Please provide the flight ID you want to book."
    return {
        "missing_fields": missing,
        "messages": [AIMessage(content=question)],
    }


def _parse_tool_result(result: any) -> list[dict] | dict:
    """Parse MCP content blocks or raw outputs from tool invocations."""
    if isinstance(result, list):
        parsed = []
        for item in result:
            if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                try:
                    data = json.loads(item["text"])
                    if isinstance(data, list):
                        parsed.extend(data)
                    else:
                        parsed.append(data)
                except json.JSONDecodeError:
                    parsed.append(item["text"])
            else:
                parsed.append(item)
        return parsed
    elif isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return result
    return result


async def _execute_hotel_tool(
    response: AIMessage,
    tools: list,
    prompt: list,
    state: AgentState,
    llm,
) -> dict:
    """Execute the hotel tool call and return the updated state dict."""
    tool_call = response.tool_calls[0]
    tool = get_tool_by_name(tools, tool_call["name"])
    if not tool:
        return {
            "last_tool_status": ToolCallStatus.FAILED,
            "last_tool_error": f"Tool {tool_call['name']} not found",
            "messages": [
                response,
                AIMessage(content="I couldn't complete that hotel request right now, please try again shortly."),
            ],
        }

    try:
        tool_result = await tool.ainvoke(tool_call["args"])
        parsed_result = _parse_tool_result(tool_result)
        tool_message = ToolMessage(
            content=json.dumps(parsed_result) if isinstance(parsed_result, (list, dict)) else str(parsed_result),
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
        )

        updates = {
            "last_tool_status": ToolCallStatus.SUCCEEDED,
            "last_tool_error": None,
        }

        if tool_call["name"] in ("search_hotels", "list_hotels"):
            updates["hotel_results"] = [Hotel(**item) for item in parsed_result]
        elif tool_call["name"] == "book_hotel":
            booking_data = parsed_result[0] if isinstance(parsed_result, list) else parsed_result
            updates["bookings"] = state.bookings + [Booking(**booking_data)]

        final_prompt = prompt + [response, tool_message]
        final_response = await llm.ainvoke(final_prompt)
        updates["messages"] = [response, tool_message, final_response]
        return updates

    except Exception as e:
        return {
            "last_tool_status": ToolCallStatus.FAILED,
            "last_tool_error": str(e),
            "messages": [
                response,
                AIMessage(content="I couldn't complete that hotel request right now, please try again shortly."),
            ],
        }


async def _execute_flight_tool(
    response: AIMessage,
    tools: list,
    prompt: list,
    state: AgentState,
    llm,
) -> dict:
    """Execute the flight tool call and return the updated state dict."""
    tool_call = response.tool_calls[0]
    tool = get_tool_by_name(tools, tool_call["name"])
    if not tool:
        return {
            "last_tool_status": ToolCallStatus.FAILED,
            "last_tool_error": f"Tool {tool_call['name']} not found",
            "messages": [
                response,
                AIMessage(content="I couldn't complete that flight request right now, please try again shortly."),
            ],
        }

    try:
        tool_result = await tool.ainvoke(tool_call["args"])
        parsed_result = _parse_tool_result(tool_result)
        tool_message = ToolMessage(
            content=json.dumps(parsed_result) if isinstance(parsed_result, (list, dict)) else str(parsed_result),
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
        )

        updates = {
            "last_tool_status": ToolCallStatus.SUCCEEDED,
            "last_tool_error": None,
        }

        if tool_call["name"] in ("search_flights", "list_flights"):
            updates["flight_results"] = [Flight(**item) for item in parsed_result]
        elif tool_call["name"] == "book_flight":
            booking_data = parsed_result[0] if isinstance(parsed_result, list) else parsed_result
            updates["bookings"] = state.bookings + [Booking(**booking_data)]

        final_prompt = prompt + [response, tool_message]
        final_response = await llm.ainvoke(final_prompt)
        updates["messages"] = [response, tool_message, final_response]
        return updates

    except Exception as e:
        return {
            "last_tool_status": ToolCallStatus.FAILED,
            "last_tool_error": str(e),
            "messages": [
                response,
                AIMessage(content="I couldn't complete that flight request right now, please try again shortly."),
            ],
        }
