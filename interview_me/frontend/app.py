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
        final_message = fixed_greeting({"messages": messages})["messages"][0]
        await cl.Message(content=str(final_message.content)).send()
    elif route == "prohibited":
        final_message = fixed_relevance_response({"messages": messages})["messages"][0]
        await cl.Message(content=str(final_message.content)).send()
    else:
        response = cl.Message(content="")
        await response.send()
        chunks: list[str] = []
        async for chunk in stream_main_agent(messages):
            chunks.append(chunk)
            await response.stream_token(chunk)
        await response.update()
        final_message = AIMessage(content="".join(chunks) or "No response generated.")

    cl.user_session.set("messages", [*messages, final_message])