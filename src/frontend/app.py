"""Streamlit adapter for the compiled interview graph and optional voice I/O."""

import asyncio
from contextlib import nullcontext
from uuid import uuid4

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langsmith.run_helpers import tracing_context

from src.backend.artifacts import validate_data_artifacts
from src.backend.config import settings
from src.backend.graph import build_graph
from src.backend.voice import (
    VoiceInputError,
    VoiceOutputError,
    synthesize_speech,
    transcribe_audio,
)

validate_data_artifacts(
    settings.profile_path,
    settings.project_path,
    settings.project_bm25_index_path,
)
graph = build_graph()


async def generate_response(
    messages: list[AIMessage | HumanMessage],
    placeholder: st.delta_generator.DeltaGenerator,
    disable_tracing: bool = False,
) -> AIMessage:
    final_message: AIMessage | None = None
    response = ""

    async for mode, update in graph.astream(
        {"messages": messages},
        stream_mode=["custom", "updates"],
        config={"callbacks": []} if disable_tracing else None,
    ):
        if mode == "custom":
            response += str(update)
            placeholder.markdown(response)
            continue

        for node_update in update.values():
            node_messages = node_update.get("messages", [])
            if node_messages:
                final_message = node_messages[-1]

    if final_message is None:
        final_message = AIMessage(content="No response generated.")
    if not response:
        placeholder.markdown(str(final_message.content))

    return final_message


def _render_voice_reply(message: AIMessage, reply: dict) -> None:
    """Render cached session audio or offer an explicit retry after an error."""
    audio = reply.get("audio")
    if audio:
        autoplay = not reply.get("autoplay_attempted", False)
        if autoplay:
            reply["autoplay_attempted"] = True
        st.audio(audio, format="audio/wav", autoplay=autoplay)
        return

    if reply.get("error"):
        st.caption(reply["error"])
        if st.button("Retry voice output", key=f"retry-tts-{message.id}"):
            try:
                reply["audio"] = synthesize_speech(str(message.content))
                reply.pop("error", None)
                reply["autoplay_attempted"] = False
            except VoiceOutputError as exc:
                reply["error"] = str(exc)
            st.rerun()


def _submit_turn(prompt: str, mode: str) -> None:
    history = st.session_state.messages
    messages = [*history, HumanMessage(content=prompt)]

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        trace_context = (
            tracing_context(enabled=False) if mode == "Voice" else nullcontext()
        )
        with trace_context:
            final_message = asyncio.run(
                generate_response(
                    messages,
                    response_placeholder,
                    disable_tracing=mode == "Voice",
                )
            )
        final_message = AIMessage(
            content=str(final_message.content),
            id=uuid4().hex,
        )

        st.session_state.messages = [*messages, final_message]
        if mode == "Voice":
            reply = {"audio": None, "autoplay_attempted": False}
            st.session_state.voice_replies[final_message.id] = reply
            try:
                reply["audio"] = synthesize_speech(str(final_message.content))
            except VoiceOutputError as exc:
                reply["error"] = str(exc)
            _render_voice_reply(final_message, reply)


st.set_page_config(page_title="Interview Me", page_icon="💬")
st.title("Interview Me")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "Text"
if "voice_replies" not in st.session_state:
    st.session_state.voice_replies = {}
if "voice_input_version" not in st.session_state:
    st.session_state.voice_input_version = 0
if "voice_input_error" not in st.session_state:
    st.session_state.voice_input_error = None

mode = st.radio(
    "Chat mode",
    ("Text", "Voice"),
    horizontal=True,
    key="chat_mode",
    help="The selected mode applies to new turns only. Typing remains available in either mode.",
)

if mode == "Voice":
    st.markdown(
        """
        <style>
        [data-testid="stElementContainer"][class*="st-key-voice-input-"] {
            position: fixed !important;
            bottom: calc(56px + 3.6rem + 0.5rem) !important;
            left: max(calc((100vw - 736px) / 2 + 16px), 16px) !important;
            width: min(704px, calc(100vw - 2.5rem)) !important;
            z-index: 100 !important;
        }
        [data-testid="stMainBlockContainer"] {
            padding-bottom: 8rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    role = "assistant" if isinstance(message, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(str(message.content))
        if isinstance(message, AIMessage):
            reply = st.session_state.voice_replies.get(message.id)
            if reply:
                _render_voice_reply(message, reply)

prompt = st.chat_input("Ask interview questions...")
audio_recording = None
if mode == "Voice":
    if st.session_state.voice_input_error:
        st.warning(st.session_state.voice_input_error)
        if st.button("Dismiss and record again", key="retry-recording"):
            st.session_state.voice_input_error = None
    audio_recording = st.audio_input(
        key=f"voice-input-{st.session_state.voice_input_version}",
    )

voice_prompt: str | None = None
if audio_recording is not None:
    audio_bytes = audio_recording.getvalue()
    # Rotate the widget key immediately so its audio is not retained for a later turn.
    st.session_state.voice_input_version += 1
    try:
        st.session_state.pending_voice_prompt = transcribe_audio(audio_bytes)
        st.session_state.voice_input_error = None
    except VoiceInputError as exc:
        st.session_state.voice_input_error = str(exc)
    finally:
        del audio_bytes
        del audio_recording
    if prompt:
        st.session_state.pending_text_prompt = (prompt, mode)
    # Rerun with the rotated widget key to clear the browser recording before submission.
    st.rerun()

voice_prompt = st.session_state.pop("pending_voice_prompt", None)
pending_text_prompt = st.session_state.pop("pending_text_prompt", None)
if pending_text_prompt:
    _submit_turn(*pending_text_prompt)
elif prompt:
    _submit_turn(prompt, mode)
if voice_prompt:
    _submit_turn(voice_prompt, "Voice")
