"""Chainlit adapter for the compiled interview graph."""

import chainlit as cl
from langchain_core.messages import HumanMessage

from interview_me.backend.graph import graph


@cl.on_message
async def on_message(message: cl.Message) -> None:
    result = await graph.ainvoke({"messages": [HumanMessage(content=message.content)]})
    final_message = result["messages"][-1]
    await cl.Message(content=str(final_message.content)).send()