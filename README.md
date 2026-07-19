# ✈️ TripWeaver – MCP-Based Multi-Agent Travel Planner

TripWeaver is an AI-powered travel planning assistant built using **FastAPI**, **LangGraph**, **Model Context Protocol (MCP)**, and **Gradio**. The application understands user travel requests, routes them to specialized AI agents, and performs tasks such as searching hotels, searching flights, answering general travel questions, and creating mock bookings. The backend streams responses in real time using Server-Sent Events (SSE), while the Gradio frontend provides an interactive chat interface similar to modern AI assistants.

---

## 🌐 Live Demo

- **Frontend:** https://tripweaver-mcp-based-multi-agent-travel-pltr.onrender.com
- **Backend API:** https://tripweaver-mcp-based-multi-agent-travel-pgt9.onrender.com
- **Health Check:** https://tripweaver-mcp-based-multi-agent-travel-pgt9.onrender.com/health

---

# Features

- 🤖 Multi-agent AI workflow using LangGraph
- 🏨 Hotel search assistant
- ✈️ Flight search assistant
- 🌍 General travel question answering
- 🔄 Real-time streaming responses using Server-Sent Events (SSE)
- 🔌 MCP-based tool integration
- 💬 Modern Gradio chat interface
- ⚡ FastAPI REST API backend

---

# System Architecture

```text
                    +--------------------+
                    |  Gradio Frontend   |
                    |  (Streaming Chat)  |
                    +---------+----------+
                              |
                              | HTTP + SSE
                              |
                    +---------v----------+
                    |   FastAPI Backend  |
                    |     (main.py)      |
                    +---------+----------+
                              |
                              |
                    +---------v----------+
                    | LangGraph Workflow |
                    |                    |
                    |  Router Agent      |
                    |        │           |
                    |   ┌────┴────┐      |
                    |   │         │      |
                    | Hotel   Flight  General
                    | Agent    Agent    Agent
                    +---------+----------+
                              |
                    MCP Client (langchain-mcp-adapters)
                              |
                    +---------v----------+
                    |   MCP Server       |
                    |   (stdio mode)     |
                    +---------+----------+
                              |
                       Travel Tools
```

## Architecture Summary

### FastAPI Backend

The FastAPI backend is responsible for:

- Exposing REST API endpoints
- Streaming AI responses using Server-Sent Events (SSE)
- Managing conversation session state
- Executing the LangGraph workflow
- Communicating with the MCP server

Available endpoints:

- `POST /chat`
- `POST /chat/stream`
- `GET /health`

---

### LangGraph Multi-Agent Workflow

TripWeaver uses **LangGraph** to coordinate multiple specialized AI agents.

Workflow:

```text
User Request
      │
      ▼
Router Agent
      │
      ├────────────► Hotel Agent
      │
      ├────────────► Flight Agent
      │
      └────────────► General Travel Agent
```

The router classifies the user's intent and forwards the request to the most appropriate specialized agent.

---

### MCP Server (stdio)

TripWeaver follows the **Model Context Protocol (MCP)** architecture by separating travel tools into an independent MCP server.

Instead of embedding hotel and flight tool implementations inside the AI agents, the agents invoke tools through **langchain-mcp-adapters**, which communicate with the MCP server over **stdio**.

This design keeps the system modular because:

- Agent code remains clean and focused on reasoning.
- Tool implementations can be modified independently.
- New tools can be added without changing agent logic.
- Tool reuse across different AI workflows becomes easier.

---

### Gradio Frontend

The Gradio frontend provides:

- Interactive chat interface
- Streaming responses
- Conversation history
- Example prompts
- Simple and responsive user experience

---

# Project Structure

```text
TripWeaver/
│
├── agents/
│   ├── entity.py
│   ├── graph.py
│   ├── nodes.py
│   └── tools.py
│
├── mcp_server/
│   ├── server.py
│   └── data/
│
├── frontend.py
├── main.py
├── requirements.txt
├── .env
└── README.md
```

---

# Local Setup

## 1. Clone the repository

```bash
git clone <repository-url>

cd TripWeaver
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv env

env\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv env

source env/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file in the project root.

```env
OPENAI_API_KEY=your_openai_api_key
```

For the frontend (optional):

```env
BACKEND_URL=http://localhost:8000
```

---

## 5. Run the FastAPI Backend

```bash
uvicorn main:app --reload
```

Backend:

```
http://localhost:8000
```

---

## 6. Run the Gradio Frontend

Open another terminal.

```bash
python frontend.py
```

Frontend:

```
http://localhost:7860
```

---

# MCP Server Setup

TripWeaver communicates with an MCP server running in **stdio mode**.

The MCP server (`mcp_server/server.py`) exposes the following tools:

| Tool | Description |
|------|-------------|
| **search_hotels** | Returns available hotels for a destination. |
| **book_hotel** | Creates a mock hotel reservation. |
| **search_flights** | Returns available flights between two cities. |
| **book_flight** | Creates a mock flight reservation. |
| **get_booking** | Retrieves booking information using a confirmation ID. |
| **cancel_booking** | Cancels an existing mock booking. |

The LangGraph agents connect to the MCP server through **langchain-mcp-adapters**, which automatically discovers and invokes these tools. This approach keeps tool logic separate from the agent workflow, making the system easier to maintain and extend.

### Test the MCP Server

Run the MCP server directly:

```bash
python mcp_server/server.py
```

The server starts in stdio mode and waits for requests from the LangGraph client.

---

# Deployment

## Backend Deployment (Render)

The FastAPI backend is deployed on **Render**.

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Required Environment Variables

```text
OPENAI_API_KEY
```

---

## Frontend Deployment (Render)

The Gradio frontend is deployed as a separate Render Web Service.

### Start Command

```bash
python frontend.py
```

### Required Environment Variables

```text
BACKEND_URL=https://tripweaver-mcp-based-multi-agent-travel-pgt9.onrender.com
```

The frontend forwards all user requests to the deployed FastAPI backend using this environment variable.

---

# User Guide

### Example 1 – Search for Hotels

**User**

```
Find me a hotel in Tokyo
```

**Assistant**

- Searches available hotels
- Displays hotel options
- Allows the user to book one of the returned hotels

---

### Example 2 – Search for Flights

**User**

```
Show flights from New York to London
```

**Assistant**

- Searches available flights
- Returns matching flight options
- Allows the user to book a selected flight

---

### Example 3 – Ask a General Travel Question

**User**

```
What's the best time to visit Japan?
```

**Assistant**

Provides general travel recommendations and destination information.

---

# Known Limitations

- Uses **mock travel data** instead of real hotel and airline APIs.
- Booking operations are simulated and do not create real reservations.
- Conversation session state is stored **in memory only**.
- Session history is lost whenever the FastAPI backend restarts.

---

# Technologies Used

- Python
- FastAPI
- LangGraph
- LangChain
- OpenAI API
- Model Context Protocol (MCP)
- langchain-mcp-adapters
- Gradio
- HTTPX
- Pydantic

---

# Future Improvements

- Integrate real hotel and flight APIs
- Persistent database for bookings and conversations
- User authentication
- Multi-user support
- AI-generated travel itineraries
- Payment integration
- Docker and CI/CD deployment

---

## Author

**TripWeaver – MCP-Based Multi-Agent Travel Planner**

Final Assignment of StemLink GENAI Bootcamp