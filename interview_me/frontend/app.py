"""Chainlit adapter for the compiled interview graph."""

import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage

from interview_me.backend.graph import graph


@cl.on_message
async def on_message(message: cl.Message) -> None:
    history = cl.user_session.get("messages") or []
    messages = [*history, HumanMessage(content=message.content)]
    final_message: AIMessage | None = None

    async for update in graph.astream(
        {"messages": messages},
        stream_mode="updates",
    ):
        for node_update in update.values():
            node_messages = node_update.get("messages", [])
            if node_messages:
                final_message = node_messages[-1]

    if final_message is None:
        final_message = AIMessage(content="No response generated.")
    await cl.Message(content=str(final_message.content)).send()
    cl.user_session.set("messages", [*messages, final_message])