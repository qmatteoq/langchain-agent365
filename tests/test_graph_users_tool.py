import asyncio
import logging
from collections import defaultdict
from types import SimpleNamespace

import pytest

import agent as agent_module
from agent import LangChainAgent
from graph_users_tool import GraphUsersTool, _condense_users_response


def build_agent() -> LangChainAgent:
    agent = object.__new__(LangChainAgent)
    agent.logger = logging.getLogger("test-agent")
    agent.auth_options = None
    agent.tool_service = None
    agent.tools = ["existing-tool"]
    agent.mcp_client = None
    agent.mcp_clients = []
    agent.mcp_servers_initialized = True
    agent.chat_model = object()
    agent._conversation_histories = {}
    agent._conversation_locks = defaultdict(asyncio.Lock)
    agent._max_history_messages = 20
    return agent


def build_context() -> SimpleNamespace:
    return SimpleNamespace(
        activity=SimpleNamespace(
            recipient=SimpleNamespace(tenant_id="tenant-1", agentic_app_id="agent-1"),
            from_property=SimpleNamespace(name="Adele", id="user-adele", aad_object_id="aad-1"),
        )
    )


def test_condense_users_response_returns_expected_fields() -> None:
    payload = {
        "value": [
            {
                "id": "1",
                "displayName": "Adele Vance",
                "mail": "adele@example.com",
                "userPrincipalName": "adele@example.com",
                "jobTitle": "Ignored",
            }
        ]
    }

    assert _condense_users_response(payload) == (
        '[{"id": "1", "displayName": "Adele Vance", "mail": "adele@example.com", '
        '"userPrincipalName": "adele@example.com"}]'
    )


@pytest.mark.asyncio
async def test_graph_users_tool_returns_unauthorized_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status_code = 403
        reason_phrase = "Forbidden"
        text = '{"error":"forbidden"}'

        def json(self):
            return {"error": "forbidden"}

        def raise_for_status(self) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, headers: dict[str, str]) -> FakeResponse:
            assert url.endswith("/users")
            assert headers["Authorization"] == "Bearer graph-token"
            return FakeResponse()

    monkeypatch.setattr("graph_users_tool.httpx.AsyncClient", FakeAsyncClient)

    tool = GraphUsersTool(lambda: _async_value("graph-token"))
    result = await tool.get_tenant_users()

    assert "not authorized" in result
    assert "403 Forbidden" in result


@pytest.mark.asyncio
async def test_build_runtime_tools_adds_graph_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = build_agent()
    context = build_context()

    class FakeCredential:
        def get_token(self, scope: str) -> SimpleNamespace:
            assert scope == "https://graph.microsoft.com/.default"
            return SimpleNamespace(token="graph-token")

    monkeypatch.setattr(agent_module, "AzureCliCredential", lambda: FakeCredential())

    runtime_tools = agent._build_runtime_tools(None, None, context)
    assert runtime_tools == ["existing-tool"]

    auth = SimpleNamespace()
    runtime_tools = agent._build_runtime_tools(auth, None, context)

    assert runtime_tools[0] == "existing-tool"
    assert runtime_tools[1].name == "get_tenant_users"


async def _async_value(value: str) -> str:
    return value