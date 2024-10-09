from contextlib import contextmanager
from typing import Optional, Union, Annotated
import httpx
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


class NonstandardMessagesState(TypedDict):
    input: str
    weird_named_messages: Annotated[list, add_messages]


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


@pytest.fixture()
def non_langgraph_app() -> FastAPI:
    """A simple server that wraps a Runnable and exposes it as an API."""

    async def add_one_or_passthrough(
        x: Union[int, HumanMessage],
    ) -> Union[int, HumanMessage]:
        """Add one to int or passthrough."""
        if isinstance(x, int):
            return x + 1
        else:
            return x

    runnable_lambda = RunnableLambda(func=add_one_or_passthrough)
    app = FastAPI()
    try:
        add_routes(
            app, runnable_lambda, config_keys=["tags"], include_callback_events=True
        )
        yield app
    finally:
        del app


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


def test_langgraph_update_state_endpoint(interrupting_app: FastAPI):
    with get_sync_remote_runnable(interrupting_app) as remote_runnable:
        config = {"configurable": {"thread_id": "2"}}
        response = remote_runnable.invoke(
            {"input": "Hello, world!", "messages": []}, config=config
        )
        assert response["messages"][0].content == "Hello, world!"

        # Add message according to langgraph state reducer
        response = remote_runnable.langgraph_update_state(
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


def test_langgraph_update_state_fails_non_langgraph(non_langgraph_app: FastAPI):
    with get_sync_remote_runnable(non_langgraph_app) as remote_runnable:
        config = {"configurable": {"thread_id": "3"}}
        with pytest.raises(httpx.HTTPError):
            remote_runnable.langgraph_update_state(
                {"messages": [AIMessage(content="modified!")]}, config=config
            )


def test_langgraph_add_human_message_endpoint(interrupting_app: FastAPI):
    with get_sync_remote_runnable(interrupting_app) as remote_runnable:
        config = {"configurable": {"thread_id": "4"}}
        response = remote_runnable.invoke(
            {"input": "Hello, world!", "messages": []}, config=config
        )
        assert response["messages"][0].content == "Hello, world!"

        # Add message according to langgraph state reducer
        response = remote_runnable.langgraph_add_human_message(
            "added message", config=config
        )
        assert response is True

        response = remote_runnable.invoke(
            {}, config=config
        )
        assert len(response["messages"]) == 3
        assert response["messages"][0].content == "Hello, world!"
        assert response["messages"][1].content == "added message"
        assert response["messages"][2].content == "Resuming execution!"


def test_langgraph_add_human_message_endpoint_custom_message_var():
    graph_builder = StateGraph(NonstandardMessagesState)
    llm = RunnablePassthrough()

    def chatbot(state: NonstandardMessagesState):
        return {"weird_named_messages": [llm.invoke(state["input"])]}

    def post_interrupt(state: NonstandardMessagesState):
        return {"weird_named_messages": [AIMessage(content="Resuming execution!")]}

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("post_interrupt", post_interrupt)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", "post_interrupt")
    graph_builder.add_edge("post_interrupt", END)
    graph = graph_builder.compile(
        checkpointer=MemorySaver(), interrupt_after=["chatbot"])

    a = app(graph)
    with get_sync_remote_runnable(a) as remote_runnable:
        config = {"configurable": {"thread_id": "52"}}
        response = remote_runnable.invoke(
            {"input": "Hello, world!", "weird_named_messages": []}, config=config
        )
        assert response["weird_named_messages"][0].content == "Hello, world!"

        # Add message according to langgraph state reducer
        response = remote_runnable.langgraph_add_human_message(
            "added message", messages_state_var="weird_named_messages", config=config
        )
        assert response is True

        response = remote_runnable.invoke(
            {}, config=config
        )
        assert len(response["weird_named_messages"]) == 3
        assert response["weird_named_messages"][0].content == "Hello, world!"
        assert response["weird_named_messages"][1].content == "added message"
        assert response["weird_named_messages"][2].content == "Resuming execution!"


def test_langgraph_add_human_message_fails_non_langgraph(non_langgraph_app: FastAPI):
    with get_sync_remote_runnable(non_langgraph_app) as remote_runnable:
        config = {"configurable": {"thread_id": "6"}}
        with pytest.raises(httpx.HTTPError):
            remote_runnable.langgraph_add_human_message(
                'should fail', config=config
            )
