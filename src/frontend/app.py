"""Streamlit adapter for the compiled interview graph."""

import asyncio

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.backend.graph import build_graph

graph = build_graph()

async def generate_response(
    messages: list[AIMessage | HumanMessage],
    placeholder: st.delta_generator.DeltaGenerator,
) -> AIMessage:
    final_message: AIMessage | None = None
    response = ""

    async for mode, update in graph.astream(
        {"messages": messages},
        stream_mode=["custom", "updates"],
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


st.set_page_config(page_title="Interview Me", page_icon="💬")
st.title("Interview Me")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = "assistant" if isinstance(message, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(str(message.content))

if prompt := st.chat_input("Ask interview questions..."):
    history = st.session_state.messages
    messages = [*history, HumanMessage(content=prompt)]

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        final_message = asyncio.run(generate_response(messages, response_placeholder))

    st.session_state.messages = [*messages, final_message]