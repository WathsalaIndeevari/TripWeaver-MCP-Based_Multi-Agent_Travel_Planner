"""TripWeaver MCP server exposing hotel and flight search and booking tools."""

from __future__ import annotations

import uuid

from mcp.server.fastmcp import FastMCP

from agents.entity import Booking, BookingItemType
from mcp_server.mock_data import (
    MOCK_FLIGHTS,
    MOCK_HOTELS,
    get_flights_by_route,
    get_hotels_by_location,
)

mcp = FastMCP("TripWeaver Travel")

BOOKINGS: list[Booking] = []


def _short_confirmation_id() -> str:
    return uuid.uuid4().hex[:8]


@mcp.tool()
def list_hotels() -> list[dict]:
    """Return every hotel in the catalog with pricing, rating, and availability."""
    return [hotel.model_dump(mode="json") for hotel in MOCK_HOTELS]


@mcp.tool()
def search_hotels(location: str) -> list[dict]:
    """Search hotels by location using a case-insensitive partial match on city or country."""
    return [hotel.model_dump(mode="json") for hotel in get_hotels_by_location(location)]


@mcp.tool()
def book_hotel(hotel_id: str) -> dict:
    """Book a hotel by ID and return the reservation details including confirmation_id."""
    hotel = next((item for item in MOCK_HOTELS if item.id == hotel_id), None)
    if hotel is None:
        raise ValueError(f"Hotel not found: {hotel_id!r}")
    if not hotel.availability:
        raise ValueError(f"Hotel {hotel_id!r} is not available for booking")

    booking = Booking(
        confirmation_id=_short_confirmation_id(),
        item_type=BookingItemType.HOTEL,
        item_id=hotel.id,
        total_cost=hotel.price_per_night,
    )
    BOOKINGS.append(booking)
    return booking.model_dump(mode="json")


@mcp.tool()
def list_flights() -> list[dict]:
    """Return every flight in the catalog with schedule, airline, and fare details."""
    return [flight.model_dump(mode="json") for flight in MOCK_FLIGHTS]


@mcp.tool()
def search_flights(origin: str, destination: str) -> list[dict]:
    """Search flights by origin and destination using case-insensitive partial matches."""
    return [
        flight.model_dump(mode="json")
        for flight in get_flights_by_route(origin, destination)
    ]


@mcp.tool()
def book_flight(flight_id: str) -> dict:
    """Book a flight by ID and return the reservation details including confirmation_id."""
    flight = next((item for item in MOCK_FLIGHTS if item.id == flight_id), None)
    if flight is None:
        raise ValueError(f"Flight not found: {flight_id!r}")

    booking = Booking(
        confirmation_id=_short_confirmation_id(),
        item_type=BookingItemType.FLIGHT,
        item_id=flight.id,
        total_cost=flight.price,
    )
    BOOKINGS.append(booking)
    return booking.model_dump(mode="json")


if __name__ == "__main__":
    mcp.run()
