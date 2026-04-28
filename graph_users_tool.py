import json
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from langchain_core.tools import tool

GRAPH_SCOPE = "https://graph.microsoft.com/.default"
GRAPH_USERS_URL = "https://graph.microsoft.com/v1.0/users"


def _condense_users_response(payload: dict[str, Any]) -> str:
    users = payload.get("value")
    if not isinstance(users, list):
        return json.dumps(payload)

    condensed = []
    for user in users:
        if not isinstance(user, dict):
            continue
        condensed.append(
            {
                "id": user.get("id"),
                "displayName": user.get("displayName"),
                "mail": user.get("mail"),
                "userPrincipalName": user.get("userPrincipalName"),
            }
        )

    return json.dumps(condensed)


class GraphUsersTool:
    def __init__(self, access_token_provider: Callable[[], Awaitable[str]]) -> None:
        self._access_token_provider = access_token_provider

    async def get_tenant_users(self) -> str:
        try:
            access_token = await self._access_token_provider()
        except Exception as error:
            return f"Error: unable to acquire a Microsoft Graph access token. Details: {error}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                GRAPH_USERS_URL,
                headers={"Authorization": f"Bearer {access_token}"},
            )

        if response.status_code in {401, 403}:
            return (
                "Error: the request to Microsoft Graph was not authorized "
                f"(HTTP {response.status_code} {response.reason_phrase}). "
                "The current token may lack the required permissions. Details: "
                f"{response.text}"
            )

        response.raise_for_status()
        return _condense_users_response(response.json())

    def as_langchain_tool(self):
        @tool(
            "get_tenant_users",
            description=(
                "Get the list of all users in the Microsoft 365 tenant from Microsoft Graph."
            ),
        )
        async def get_tenant_users() -> str:
            return await self.get_tenant_users()

        return get_tenant_users