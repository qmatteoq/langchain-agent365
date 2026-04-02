import asyncio
import logging
from collections import defaultdict
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import agent as agent_module
from agent import LangChainAgent


def build_agent() -> LangChainAgent:
    agent = object.__new__(LangChainAgent)
    agent.logger = logging.getLogger("test-agent")
    agent.auth_options = None
    agent.tool_service = None
    agent.tools = []
    agent.mcp_client = None
    agent.mcp_clients = []
    agent.mcp_servers_initialized = True
    agent.chat_model = object()
    agent._conversation_histories = {}
    agent._conversation_locks = defaultdict(asyncio.Lock)
    agent._max_history_messages = 20
    return agent


def build_context(conversation_id: str, user_name: str = "Adele") -> SimpleNamespace:
    return SimpleNamespace(
        activity=SimpleNamespace(
            text="",
            channel_id="msteams",
            conversation=SimpleNamespace(id=conversation_id),
            recipient=SimpleNamespace(tenant_id="tenant-1", agentic_app_id="agent-1"),
            from_property=SimpleNamespace(
                name=user_name,
                id=f"user-{user_name.lower()}",
                aad_object_id="aad-1",
            ),
        )
    )


@pytest.mark.asyncio
async def test_process_user_message_reuses_prior_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = build_agent()
    payloads: list[dict] = []
    responses = iter([
        "Here is the list of things I can do.",
        "I can send that list to x@y.z.",
    ])

    class FakeRunnable:
        async def ainvoke(self, payload: dict) -> dict:
            payloads.append(payload)
            return {"messages": [AIMessage(content=next(responses))]}

    async def fake_setup_mcp_servers(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("agent.create_agent", lambda **_kwargs: FakeRunnable())
    agent.setup_mcp_servers = fake_setup_mcp_servers

    context = build_context("conversation-1")

    first_reply = await agent.process_user_message("what can you do?", None, None, context)
    second_reply = await agent.process_user_message(
        "can you send this list to mail address x@y.z?",
        None,
        None,
        context,
    )

    assert first_reply == "Here is the list of things I can do."
    assert second_reply == "I can send that list to x@y.z."

    first_messages = payloads[0]["messages"]
    assert len(first_messages) == 1
    assert isinstance(first_messages[0], HumanMessage)
    assert first_messages[0].content == "what can you do?"

    second_messages = payloads[1]["messages"]
    assert [type(message) for message in second_messages] == [
        HumanMessage,
        AIMessage,
        HumanMessage,
    ]
    assert second_messages[0].content == "what can you do?"
    assert second_messages[1].content == "Here is the list of things I can do."
    assert second_messages[2].content == "can you send this list to mail address x@y.z?"


@pytest.mark.asyncio
async def test_process_user_message_isolates_conversation_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = build_agent()
    payloads: list[dict] = []

    class FakeRunnable:
        async def ainvoke(self, payload: dict) -> dict:
            payloads.append(payload)
            return {"messages": [AIMessage(content="ok")]}

    async def fake_setup_mcp_servers(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("agent.create_agent", lambda **_kwargs: FakeRunnable())
    agent.setup_mcp_servers = fake_setup_mcp_servers

    first_context = build_context("conversation-1", user_name="Adele")
    second_context = build_context("conversation-2", user_name="Megan")

    await agent.process_user_message("first thread", None, None, first_context)
    await agent.process_user_message("second thread", None, None, second_context)

    first_messages = payloads[0]["messages"]
    second_messages = payloads[1]["messages"]

    assert [message.content for message in first_messages] == ["first thread"]
    assert [message.content for message in second_messages] == ["second thread"]


def test_get_additional_mcp_server_configs_uses_github_remote_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = build_agent()

    monkeypatch.setenv("ENABLE_GITHUB_REMOTE_MCP", "true")
    monkeypatch.setenv("GITHUB_REMOTE_MCP_TOKEN", "github-token")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(agent_module.Utility, "get_user_agent_header", lambda _name: "test-agent")

    client_config = agent._get_additional_mcp_server_configs()

    assert client_config == {
        agent_module.GITHUB_REMOTE_MCP_SERVER_NAME: {
            "transport": "http",
            "url": agent_module.GITHUB_REMOTE_MCP_DEFAULT_URL,
            "headers": {
                "Authorization": "Bearer github-token",
                "User-Agent": "test-agent",
            },
        }
    }


def test_get_additional_mcp_server_configs_skips_when_token_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = build_agent()

    monkeypatch.setenv("ENABLE_GITHUB_REMOTE_MCP", "true")
    monkeypatch.delenv("GITHUB_REMOTE_MCP_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)

    assert agent._get_additional_mcp_server_configs() == {}


@pytest.mark.asyncio
async def test_load_tools_from_client_config_continues_after_server_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = build_agent()

    class FakeMCPClient:
        def __init__(self, server_config: dict[str, dict[str, str]]) -> None:
            self.server_config = server_config

        async def get_tools(self) -> list[str]:
            server_name = next(iter(self.server_config))
            if server_name == agent_module.GITHUB_REMOTE_MCP_SERVER_NAME:
                raise RuntimeError("GitHub MCP unavailable")
            return [f"tool:{server_name}"]

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(agent_module, "MultiServerMCPClient", FakeMCPClient)

    tools = await agent._load_tools_from_client_config(
        {
            "mail": {
                "transport": "http",
                "url": "https://example.test/mail",
                "headers": {},
            },
            agent_module.GITHUB_REMOTE_MCP_SERVER_NAME: {
                "transport": "http",
                "url": agent_module.GITHUB_REMOTE_MCP_DEFAULT_URL,
                "headers": {},
            },
        }
    )

    assert tools == ["tool:mail"]
    assert len(agent.mcp_clients) == 1
    assert agent.mcp_client is agent.mcp_clients[0]
