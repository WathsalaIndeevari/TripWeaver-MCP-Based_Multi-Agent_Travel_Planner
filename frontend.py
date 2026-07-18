"""Gradio streaming chat frontend for TripWeaver."""

from __future__ import annotations

import json
import os
import uuid

import gradio as gr
import httpx

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
STREAM_ENDPOINT = f"{BACKEND_URL}/chat/stream"

_FALLBACK_ERROR = "Something went wrong reaching the travel assistant. Please try again."


def _new_session_id() -> str:
    return str(uuid.uuid4())


async def _stream_chat(message: str, history: list, session_id: str):
    """Call the backend's SSE stream and yield progressively updated chat history.

    Gradio's ChatInterface-style generators expect each yield to be the full
    updated `history` list (messages format), so we mutate the last assistant
    turn in place as activity/token/final events arrive.
    """
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "_Understanding your request..._"},
    ]
    yield history

    assembled_text = ""
    saw_final = False

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                STREAM_ENDPOINT,
                json={"message": message, "session_id": session_id},
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    payload_raw = line[len("data: "):].strip()
                    if not payload_raw:
                        continue

                    try:
                        event = json.loads(payload_raw)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type")

                    if event_type == "activity":
                        # Only show the activity label while we don't yet have real content
                        if not assembled_text:
                            history[-1]["content"] = f"_{event.get('label', 'Working...')}_"
                            yield history

                    elif event_type == "token":
                        assembled_text += event.get("content", "")
                        history[-1]["content"] = assembled_text
                        yield history

                    elif event_type == "final":
                        assembled_text = event.get("content", "") or assembled_text
                        history[-1]["content"] = assembled_text or "I didn't get a response, please try again."
                        saw_final = True
                        yield history

    except (httpx.HTTPError, httpx.TimeoutException):
        history[-1]["content"] = _FALLBACK_ERROR
        yield history
        return

    if not saw_final:
        # Stream ended without a proper "final" event -- degrade gracefully rather than
        # leaving a stale activity label on screen.
        if not assembled_text:
            history[-1]["content"] = _FALLBACK_ERROR
        yield history


def _build_demo() -> gr.Blocks:
    with gr.Blocks(title="TripWeaver \u2708\ufe0f") as demo:
        session_id_state = gr.State(_new_session_id)

        gr.Markdown("# TripWeaver \u2708\ufe0f")
        gr.Markdown("Your AI travel assistant \u2014 ask about destinations, search hotels, or find flights.")

        chatbot = gr.Chatbot(height=520, label=None, show_label=False)

        with gr.Row():
            msg_box = gr.Textbox(
                placeholder="Try: \"find me a hotel in Tokyo\" or \"what's the best time to visit Japan?\"",
                show_label=False,
                scale=8,
                container=False,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        gr.Examples(
            examples=[
                "What's the best time of year to visit Japan?",
                "Find me a hotel in Paris",
                "Show me flights from New York to London",
            ],
            inputs=msg_box,
        )

        def _clear_textbox():
            return ""

        send_btn.click(
            _stream_chat,
            inputs=[msg_box, chatbot, session_id_state],
            outputs=chatbot,
        ).then(_clear_textbox, outputs=msg_box)

        msg_box.submit(
            _stream_chat,
            inputs=[msg_box, chatbot, session_id_state],
            outputs=chatbot,
        ).then(_clear_textbox, outputs=msg_box)

    return demo


if __name__ == "__main__":
    theme = gr.themes.Soft(primary_hue="teal", secondary_hue="blue")
    demo = _build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        theme=theme,
    )