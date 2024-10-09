from contextlib import contextmanager
from typing import Optional, Union, Annotated
from typing_extensions import TypedDict
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import (
    ConfigurableField,
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnablePassthrough,
)
from langserve.client import RemoteRunnable
from langserve.server import add_routes
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
import asyncio
from tests.unit_tests.utils.llms import FakeListLLM
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop()
    try:
        yield loop
    finally:
        loop.close()


class State(TypedDict):
    input: str
    messages: Annotated[list, add_messages]


@pytest.fixture()
def simple_graph() -> StateGraph:
    graph_builder = StateGraph(State)
    llm = RunnablePassthrough()

    def chatbot(state: State):
        return {"messages": [llm.invoke(state["input"])]}

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    return graph_builder.compile()


@pytest.fixture()
def interrupting_graph() -> StateGraph:
    graph_builder = StateGraph(State)
    llm = RunnablePassthrough()

    def chatbot(state: State):
        return {"messages": [llm.invoke(state["input"])]}

    def post_interrupt(state: State):
        return {"messages": [AIMessage(content="Resuming execution!")]}

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("post_interrupt", post_interrupt)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", "post_interrupt")
    graph_builder.add_edge("post_interrupt", END)
    return graph_builder.compile(checkpointer=MemorySaver(), interrupt_after=["chatbot"])


def app(graph: CompiledStateGraph) -> FastAPI:
    """A simple server that wraps a Runnable and exposes it as an API."""
    app = FastAPI()
    add_routes(
        app, graph, include_callback_events=True
    )
    return app


@pytest.fixture()
def simple_app(simple_graph: StateGraph):
    a = app(simple_graph)
    try:
        yield a
    finally:
        del a


@pytest.fixture()
def interrupting_app(interrupting_graph: StateGraph):
    a = app(interrupting_graph)
    try:
        yield a
    finally:
        del a


@contextmanager
def get_sync_remote_runnable(
    server: FastAPI, *, path: Optional[str] = None, raise_server_exceptions: bool = True
) -> RemoteRunnable:
    """Get an async client."""
    url = "http://localhost:9999"
    if path:
        url += path
    remote_runnable_client = RemoteRunnable(url=url)
    sync_client = TestClient(
        app=server, base_url=url, raise_server_exceptions=raise_server_exceptions
    )
    remote_runnable_client.sync_client = sync_client
    try:
        yield remote_runnable_client
    finally:
        sync_client.close()


def test_simple_langgraph(simple_app: FastAPI):
    with get_sync_remote_runnable(simple_app) as remote_runnable:
        response = remote_runnable.invoke(
            {"input": "Hello, world!", "messages": []}
        )
        assert len(response["messages"]) == 1
        assert response["messages"][0].content == "Hello, world!"


def test_interrupting_langgraph(interrupting_app: FastAPI):
    with get_sync_remote_runnable(interrupting_app) as remote_runnable:
        config = {"configurable": {"thread_id": "1"}}
        response = remote_runnable.invoke(
            {"input": "Hello, world!", "messages": []}, config=config
        )
        assert len(response["messages"]) == 1
        assert response["messages"][0].content == "Hello, world!"
        response = remote_runnable.invoke(
            {}, config=config
        )
        assert len(response["messages"]) == 2
        assert response["messages"][1].content == "Resuming execution!"


def test_update_langgraph_state_endpoint(interrupting_app: FastAPI):
    with get_sync_remote_runnable(interrupting_app) as remote_runnable:
        config = {"configurable": {"thread_id": "2"}}
        config = {"configurable": {"thread_id": "1"}}
        response = remote_runnable.invoke(
            {"input": "Hello, world!", "messages": []}, config=config
        )
        assert response["messages"][0].content == "Hello, world!"

        # Add message according to langgraph state reducer
        response = remote_runnable.update_langgraph_state(
            {"messages": [AIMessage(content="modified!")]}, config=config
        )
        assert response is True

        response = remote_runnable.invoke(
            {}, config=config
        )
        assert len(response["messages"]) == 3
        assert response["messages"][0].content == "Hello, world!"
        assert response["messages"][1].content == "modified!"
        assert response["messages"][2].content == "Resuming execution!"
