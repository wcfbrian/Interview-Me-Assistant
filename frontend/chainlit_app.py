"""Chainlit entrypoint for the interview assistant."""

import chainlit as cl

from backend.agents.runtime import create_chat_model
from backend.config import get_settings
from backend.runtime import create_runtime_graph
from backend.services.memory import MainAgentMemory


@cl.password_auth_callback
def authenticate(username: str, password: str):
    settings = get_settings()
    if (
        settings.chainlit_auth_username
        and settings.chainlit_auth_password
        and username == settings.chainlit_auth_username
        and password == settings.chainlit_auth_password
    ):
        return cl.User(identifier=username)
    return None


async def _summarize(messages: list[dict[str, str]]) -> str:
    settings = get_settings()
    model = create_chat_model(settings, settings.main_model)
    prompt = "Summarize this interview context without adding facts:\n" + str(messages)
    result = await model.ainvoke(prompt)
    return result.content


@cl.on_chat_start
async def start() -> None:
    cl.user_session.set("graph", create_runtime_graph())
    cl.user_session.set("memory", MainAgentMemory(_summarize))
    await cl.Message(content="Ready for interview questions.").send()


@cl.on_message
async def handle_message(message: cl.Message) -> None:
    graph = cl.user_session.get("graph")
    memory = cl.user_session.get("memory")
    if graph is None or memory is None:
        await cl.Message(content="I am not able to answer that reliably.").send()
        return

    await cl.Message(content="Thinking...").send()
    query = message.content.strip()
    result = await graph.ainvoke({"query": query, "main_history": memory.messages()})
    response = result.get("response", "I am not able to answer that reliably.")
    await memory.add("user", query)
    await memory.add("assistant", response)
    await cl.Message(content=response).send()