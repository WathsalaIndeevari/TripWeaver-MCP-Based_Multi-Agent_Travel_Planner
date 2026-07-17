from __future__ import annotations

import operator
from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class Intent(str, Enum):
    """Active routing target for the multi-agent workflow."""

    GENERAL = "general"
    HOTEL = "hotel"
    FLIGHT = "flight"


class BookingItemType(str, Enum):
    """Discriminator for the type of booked travel item."""

    HOTEL = "hotel"
    FLIGHT = "flight"


class BookingStatus(str, Enum):
    """Lifecycle status of a reservation."""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"


class ToolCallStatus(str, Enum):
    """Lifecycle status of an MCP tool invocation."""

    INVOKED = "invoked"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Hotel(BaseModel):
    """Hotel listing with pricing and availability."""

    id: str
    name: str
    location: str
    price_per_night: float = Field(ge=0)
    rating: float = Field(ge=0, le=5)
    availability: bool = True


class Flight(BaseModel):
    """Flight option with schedule and fare details."""

    id: str
    flight_number: str
    airline: str
    origin: str
    destination: str
    departure_time: datetime
    arrival_time: datetime
    price: float = Field(ge=0)


class Booking(BaseModel):
    """Reservation record for a hotel or flight."""

    confirmation_id: str
    item_type: BookingItemType
    item_id: str
    total_cost: float = Field(ge=0)
    booking_status: BookingStatus = BookingStatus.PENDING


class AgentState(BaseModel):
    """Pydantic graph state shared across TripWeaver LangGraph nodes."""

    messages: Annotated[list[BaseMessage], operator.add] = Field(default_factory=list)
    current_intent: Intent | str = Intent.GENERAL
    extracted_details: dict[str, Any] = Field(default_factory=dict)

    # Fields still required before a tool call can be made (code-checked per intent)
    missing_fields: list[str] = Field(default_factory=list)

    # Outcome of the most recent MCP tool invocation, for graceful degradation
    last_tool_status: ToolCallStatus | None = None
    last_tool_error: str | None = None

    # Search results — persist across turns so "book the second one" works later
    hotel_results: list[Hotel] = Field(default_factory=list)
    flight_results: list[Flight] = Field(default_factory=list)

    # Confirmed bookings this session
    bookings: list[Booking] = Field(default_factory=list)