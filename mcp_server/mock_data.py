"""In-memory mock travel inventory for development and testing."""

from __future__ import annotations

from datetime import datetime

from agents.entity import Flight, Hotel

MOCK_HOTELS: list[Hotel] = [
    Hotel(
        id="hotel-001",
        name="Hotel Le Marais",
        location="Paris, France",
        price_per_night=220.0,
        rating=4.5,
        availability=True,
    ),
    Hotel(
        id="hotel-002",
        name="Champs-Élysées Palace",
        location="Paris, France",
        price_per_night=450.0,
        rating=4.8,
        availability=True,
    ),
    Hotel(
        id="hotel-003",
        name="Shinjuku Grand",
        location="Tokyo, Japan",
        price_per_night=180.0,
        rating=4.3,
        availability=True,
    ),
    Hotel(
        id="hotel-004",
        name="Asakusa Inn",
        location="Tokyo, Japan",
        price_per_night=95.0,
        rating=3.9,
        availability=False,
    ),
    Hotel(
        id="hotel-005",
        name="The Manhattan Loft",
        location="New York, USA",
        price_per_night=310.0,
        rating=4.6,
        availability=True,
    ),
    Hotel(
        id="hotel-006",
        name="Brooklyn Bridge Hotel",
        location="New York, USA",
        price_per_night=175.0,
        rating=4.1,
        availability=True,
    ),
]

MOCK_FLIGHTS: list[Flight] = [
    Flight(
        id="flight-001",
        flight_number="DL210",
        airline="Delta Air Lines",
        origin="New York, JFK",
        destination="Paris, CDG",
        departure_time=datetime(2026, 8, 15, 18, 30),
        arrival_time=datetime(2026, 8, 16, 8, 45),
        price=685.0,
    ),
    Flight(
        id="flight-002",
        flight_number="NH105",
        airline="All Nippon Airways",
        origin="Los Angeles, LAX",
        destination="Tokyo, NRT",
        departure_time=datetime(2026, 9, 2, 12, 15),
        arrival_time=datetime(2026, 9, 3, 16, 40),
        price=920.0,
    ),
    Flight(
        id="flight-003",
        flight_number="AF007",
        airline="Air France",
        origin="Paris, CDG",
        destination="New York, JFK",
        departure_time=datetime(2026, 8, 20, 10, 0),
        arrival_time=datetime(2026, 8, 20, 12, 30),
        price=612.0,
    ),
    Flight(
        id="flight-004",
        flight_number="UA958",
        airline="United Airlines",
        origin="Chicago, ORD",
        destination="London, LHR",
        departure_time=datetime(2026, 7, 25, 17, 45),
        arrival_time=datetime(2026, 7, 26, 7, 20),
        price=540.0,
    ),
    Flight(
        id="flight-005",
        flight_number="BA005",
        airline="British Airways",
        origin="London, LHR",
        destination="Tokyo, NRT",
        departure_time=datetime(2026, 10, 5, 15, 30),
        arrival_time=datetime(2026, 10, 6, 11, 0),
        price=875.0,
    ),
    Flight(
        id="flight-006",
        flight_number="VS4",
        airline="Virgin Atlantic",
        origin="New York, JFK",
        destination="London, LHR",
        departure_time=datetime(2026, 11, 12, 20, 0),
        arrival_time=datetime(2026, 11, 13, 8, 15),
        price=498.0,
    ),
]


def get_hotels_by_location(location: str) -> list[Hotel]:
    """Return hotels whose location partially matches the query (case-insensitive)."""
    query = location.casefold()
    return [hotel for hotel in MOCK_HOTELS if query in hotel.location.casefold()]


def get_flights_by_route(origin: str, destination: str) -> list[Flight]:
    """Return flights whose origin and destination partially match (case-insensitive)."""
    origin_query = origin.casefold()
    dest_query = destination.casefold()
    return [
        flight
        for flight in MOCK_FLIGHTS
        if origin_query in flight.origin.casefold()
        and dest_query in flight.destination.casefold()
    ]
