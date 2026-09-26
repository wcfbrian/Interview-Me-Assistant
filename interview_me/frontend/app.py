"""Chainlit adapter for the compiled interview graph."""

import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage

from interview_me.backend.agents.main import stream_main_agent
from interview_me.backend.graph import fixed_greeting, fixed_relevance_response, guard_route


@cl.on_message
async def on_message(message: cl.Message) -> None:
    history = cl.user_session.get("messages") or []
    messages = [*history, HumanMessage(content=message.content)]
    route = guard_route({"messages": messages})
    if route == "greeting":
        content = fixed_greeting({})["messages"][0].content
        await cl.Message(content=str(content)).send()
        cl.user_session.set("messages", [*messages, AIMessage(content=str(content))])
        return
    if route == "irrelevant":
        content = fixed_relevance_response({})["messages"][0].content
        await cl.Message(content=str(content)).send()
        cl.user_session.set("messages", [*messages, AIMessage(content=str(content))])
        return

    response = cl.Message(content="")
    await response.send()
    async for token in stream_main_agent(messages):
        await response.stream_token(token)
    await response.update()
    cl.user_session.set("messages", [*messages, AIMessage(content=response.content)])