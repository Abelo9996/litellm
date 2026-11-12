"""
Test for google_endpoints/endpoints.py
"""

import asyncio
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Final

import httpx
import pytest
import responses
import respx
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from pydantic import JsonValue, ValidationError
from starlette.requests import Request
from starlette.responses import Response

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.google_endpoints.endpoints import _google_count_tokens_request, google_count_tokens
from litellm.types.llms.vertex_ai import TokenCountDetailsResponse
from tests._master_key import MASTER_KEY


@pytest.mark.parametrize(
    ("system_key", "schema_key"),
    (("systemInstruction", "parameters"), ("system_instruction", "parametersJsonSchema")),
)
def test_google_count_tokens_translates_system_and_function_declarations(system_key: str, schema_key: str) -> None:
    schema: Final = {"type": "object", "properties": {"query": {"type": "string"}}}
    request_body: Final[dict[str, JsonValue]] = {
        "contents": [{"role": "user", "parts": [{"text": "Look this up"}]}],
        system_key: {"parts": [{"text": "Use the provided tool."}]},
        "tools": [
            {
                "functionDeclarations": [
                    {
                        "name": "lookup",
                        "description": "Find a record",
                        schema_key: schema,
                    }
                ]
            }
        ],
    }
    request: Final = _google_count_tokens_request("claude-sonnet", request_body)

    assert request.system == [{"type": "text", "text": "Use the provided tool."}]
    assert request.tools == [
        {
            "name": "lookup",
            "description": "Find a record",
            "input_schema": schema,
        }
    ]
    assert request.messages == [{"role": "user", "content": "Look this up"}]


@pytest.mark.parametrize(
    ("schema_key", "expected_schema"),
    (
        ("parameters", {"type": "object", "properties": {"query": {"type": "string"}}}),
        ("parametersJsonSchema", {"type": "OBJECT", "properties": {"query": {"type": "STRING"}}}),
    ),
)
def test_google_count_tokens_normalizes_parameters_only(schema_key: str, expected_schema: dict[str, JsonValue]) -> None:
    request: Final = _google_count_tokens_request(
        "claude-sonnet",
        {
            "contents": [{"role": "user", "parts": [{"text": "Look this up"}]}],
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": "lookup",
                            schema_key: {
                                "type": "OBJECT",
                                "properties": {"query": {"type": "STRING"}},
                            },
                        }
                    ]
                }
            ],
        },
    )

    assert request.tools == [{"name": "lookup", "input_schema": expected_schema}]


load_dotenv()


@pytest.mark.asyncio
async def test_proxy_gemini_to_openai_like_model_token_counting():
    """
    Test the token counting endpoint for proxing gemini to openai-like models.
    """
    response: TokenCountDetailsResponse = await google_count_tokens(
        request=Request(
            scope={
                "type": "http",
                "headers": [],
                "parsed_body": (
                    ["contents"],
                    {"contents": [{"parts": [{"text": "Hello, how are you?"}]}]},
                ),
            }
        ),
        model_name="volcengine/foo",
        fastapi_response=Response(),
    )

    assert response.get("totalTokens") > 0


_DECLARATION: Final = {"name": "lookup", "description": "Find a record"}
_SYSTEM: Final = {"systemInstruction": {"parts": [{"text": "Follow these rules"}]}}
_NO_SYSTEM_PARTS: Final = {"systemInstruction": {"parts": []}}
_TOOLS: Final = {"tools": [{"functionDeclarations": [_DECLARATION]}]}
_TURN: Final = {"role": "user", "parts": [{"text": "Count this request"}]}
_CONTENTS: Final = {"contents": [_TURN]}


@pytest.mark.parametrize(
    ("spelled", "canonical"),
    (
        ({"systemInstruction": {"parts": {"text": "Follow these rules"}}}, _SYSTEM),
        ({"system_instruction": {"parts": {"text": "Follow these rules"}}}, _SYSTEM),
        ({"systemInstruction": {"parts": [{"text": "Follow these rules"}, None]}}, _SYSTEM),
        ({"systemInstruction": {"parts": None}}, _NO_SYSTEM_PARTS),
        ({"systemInstruction": {"parts": [{"text": None}, {"text": "Follow these rules"}]}}, _SYSTEM),
        ({"systemInstruction": {"parts": {"text": None}}}, _NO_SYSTEM_PARTS),
        ({"systemInstruction": {"parts": {"text": None, "inlineData": None}}}, _NO_SYSTEM_PARTS),
        ({"systemInstruction": {"parts": {"text": None, "thought": False}}}, _NO_SYSTEM_PARTS),
        ({"systemInstruction": {"parts": [{"thoughtSignature": "c2ln"}, {"text": "Follow these rules"}]}}, _SYSTEM),
        ({"systemInstruction": {"parts": [{}]}}, _NO_SYSTEM_PARTS),
        ({"systemInstruction": {}}, _NO_SYSTEM_PARTS),
        ({"tools": None}, {}),
        ({"tools": [None]}, {}),
        ({"tools": [{"functionDeclarations": None}]}, {}),
        ({"tools": {"functionDeclarations": [_DECLARATION]}}, _TOOLS),
        ({"tools": [{"functionDeclarations": _DECLARATION}]}, _TOOLS),
        ({"tools": {"functionDeclarations": _DECLARATION}}, _TOOLS),
        ({"contents": _TURN}, _CONTENTS),
        ({"contents": [None, _TURN]}, _CONTENTS),
        ({"contents": None}, {"contents": []}),
        ({"contents": [{"role": "user", "parts": {"text": "Count this request"}}]}, _CONTENTS),
        ({"contents": [{"role": "user", "parts": [None, {"text": "Count this request"}]}]}, _CONTENTS),
        ({"contents": [{"role": "user", "parts": None}]}, {"contents": [{"role": "user", "parts": []}]}),
    ),
)
def test_google_count_tokens_counts_every_google_spelling_of_a_repeated_field_like_its_list_form(
    spelled: dict[str, JsonValue], canonical: dict[str, JsonValue]
) -> None:
    contents: Final = [{"role": "user", "parts": [{"text": "Count this request"}]}]

    assert _google_count_tokens_request("counted", {"contents": contents, **spelled}) == _google_count_tokens_request(
        "counted", {"contents": contents, **canonical}
    )


@pytest.mark.parametrize(
    "malformed",
    (
        {"tools": "lookup"},
        {"tools": [{"functionDeclarations": "lookup"}]},
        {"systemInstruction": {"parts": 7}},
        {"contents": 5},
        {"contents": ["Count this request"]},
    ),
)
def test_google_count_tokens_rejects_a_scalar_where_google_expects_an_object_or_list(
    malformed: dict[str, JsonValue],
) -> None:
    with pytest.raises(ValidationError):
        _google_count_tokens_request("counted", {"contents": [_TURN], **malformed})


@pytest.mark.parametrize(
    "parts",
    (
        {"inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}},
        [{"inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}}],
        {"text": None, "inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}},
        [{"text": "Follow these rules"}, {"text": None, "fileData": {"fileUri": "gs://bucket/rules.pdf"}}],
    ),
)
def test_google_count_tokens_rejects_media_only_system_parts(parts: JsonValue) -> None:
    with pytest.raises(ValidationError):
        _google_count_tokens_request(
            "counted",
            {"contents": [{"role": "user", "parts": [{"text": "Hello"}]}], "systemInstruction": {"parts": parts}},
        )


def test_google_count_tokens_keeps_counting_system_text_that_sits_beside_media() -> None:
    text_with_media: Final = {"text": "Follow these rules", "inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}}

    assert _google_count_tokens_request(
        "counted", {**_CONTENTS, "systemInstruction": {"parts": [text_with_media]}}
    ) == _google_count_tokens_request("counted", {**_CONTENTS, **_SYSTEM})


_INTERACTION_ID: Final = "abc123"
_VERTEX_GROUP: Final = "vertex-argon"
_VERTEX_DEEP_RESEARCH_GROUP: Final = "vertex-dr"
_DEEP_RESEARCH_AGENT: Final = "deep-research-preview-04-2026"
_VERTEX_COLLECTION_URL: Final = (
    "https://aiplatform.googleapis.com/v1beta1/projects/deployment-project/locations/global/interactions"
)
_VERTEX_INTERACTION_URL: Final = f"{_VERTEX_COLLECTION_URL}/{_INTERACTION_ID}"
_GEMINI_INTERACTION_URL: Final = f"https://generativelanguage.googleapis.com/v1beta/interactions/{_INTERACTION_ID}"
_INTERACTION: Final = {"id": _INTERACTION_ID, "object": "interaction", "status": "cancelled"}
_BY_ID_CALLS: Final = (
    ("GET", f"/v1beta/interactions/{_INTERACTION_ID}", ""),
    ("DELETE", f"/v1beta/interactions/{_INTERACTION_ID}", ""),
    ("POST", f"/v1beta/interactions/{_INTERACTION_ID}/cancel", "/cancel"),
)
_BY_ID_ROUTES: Final = tuple((method, path) for method, path, _ in _BY_ID_CALLS)


def _vertex_deployment(model_name: str, model: str) -> Mapping[str, object]:
    credentials: Final = {
        "type": "authorized_user",
        "client_id": "deployment-client",
        "client_secret": "deployment-secret",
        "refresh_token": "deployment-refresh-token",
        "quota_project_id": "quota-project",
    }
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": model,
            "vertex_project": "deployment-project",
            "vertex_location": "global",
            "vertex_credentials": json.dumps(credentials),
        },
    }


@pytest.fixture
def interactions_proxy(httpx_transport: None, fake_provider_credentials: None) -> Iterator[TestClient]:
    from litellm.proxy import proxy_server

    router: Final = litellm.Router(
        model_list=[
            _vertex_deployment(_VERTEX_GROUP, "vertex_ai/gemini-4-argon"),
            _vertex_deployment(_VERTEX_DEEP_RESEARCH_GROUP, f"vertex_ai/{_DEEP_RESEARCH_AGENT}"),
        ]
    )
    proxy_server.cleanup_router_config_variables()
    try:
        with (
            pytest.MonkeyPatch.context() as proxy_globals,
            responses.RequestsMock(assert_all_requests_are_fired=False) as google_oauth,
        ):
            proxy_globals.setattr(proxy_server, "llm_router", None)
            config_path: Final = Path(__file__).parents[1] / "test_configs" / "test_config_no_auth.yaml"
            asyncio.run(proxy_server.initialize(config=str(config_path)))
            config_router: Final = proxy_server.llm_router
            proxy_globals.setattr(proxy_server, "llm_router", router)
            google_oauth.post(
                "https://oauth2.googleapis.com/token",
                json={"access_token": "deployment-token", "expires_in": 3600, "token_type": "Bearer"},
            )
            client: Final = TestClient(proxy_server.app)
            try:
                yield client
            finally:
                client.close()
                if config_router is not None:
                    config_router.retire()
    finally:
        router.retire()
        proxy_server.cleanup_router_config_variables()


def _selected(selector: str) -> tuple[Mapping[str, str], Mapping[str, str]]:
    match selector:
        case "query":
            return {"model": _VERTEX_GROUP}, {}
        case "header":
            return {}, {"x-litellm-model": _VERTEX_GROUP}
        case _:
            raise AssertionError(selector)


@pytest.mark.parametrize("selector", ("query", "header"))
@pytest.mark.parametrize(("method", "path", "upstream_suffix"), _BY_ID_CALLS)
def test_interaction_by_id_reaches_the_named_vertex_deployment_with_its_project_and_credentials(
    interactions_proxy: TestClient,
    respx_mock: respx.MockRouter,
    selector: str,
    method: str,
    path: str,
    upstream_suffix: str,
) -> None:
    upstream: Final = respx_mock.request(method, f"{_VERTEX_INTERACTION_URL}{upstream_suffix}").mock(
        return_value=httpx.Response(200, json=_INTERACTION)
    )

    params, headers = _selected(selector)

    response: Final = interactions_proxy.request(method, path, params=params, headers=headers)

    assert response.status_code == 200, response.text
    assert response.json()["id"] == _INTERACTION_ID
    assert upstream.call_count == 1
    assert upstream.calls.last.request.headers["authorization"] == "Bearer deployment-token"


@pytest.mark.parametrize("query", ({}, {"model": ""}), ids=("no_selector", "empty_selector"))
@pytest.mark.parametrize(("method", "path", "upstream_suffix"), _BY_ID_CALLS)
def test_interaction_by_id_without_a_model_still_goes_to_google_ai_studio(
    interactions_proxy: TestClient,
    respx_mock: respx.MockRouter,
    query: Mapping[str, str],
    method: str,
    path: str,
    upstream_suffix: str,
) -> None:
    upstream: Final = respx_mock.request(method, f"{_GEMINI_INTERACTION_URL}{upstream_suffix}").mock(
        return_value=httpx.Response(200, json=_INTERACTION)
    )

    response: Final = interactions_proxy.request(method, path, params=query)

    assert response.status_code == 200, response.text
    assert upstream.call_count == 1
    assert upstream.calls.last.request.headers["x-goog-api-key"] == "unit-test"


@pytest.mark.parametrize(("method", "path"), _BY_ID_ROUTES)
def test_interaction_by_id_naming_an_unknown_model_group_is_refused_before_any_upstream_call(
    interactions_proxy: TestClient, respx_mock: respx.MockRouter, method: str, path: str
) -> None:
    response: Final = interactions_proxy.request(method, path, params={"model": "no-such-group"})

    assert response.status_code == 400, response.text
    assert "no-such-group" in response.text
    assert not respx_mock.calls


@pytest.mark.parametrize("selector", ("query", "header"))
@pytest.mark.parametrize(("method", "path"), _BY_ID_ROUTES)
def test_interaction_by_id_naming_a_model_the_key_cannot_call_is_refused_before_any_upstream_call(
    interactions_proxy: TestClient,
    respx_mock: respx.MockRouter,
    selector: str,
    method: str,
    path: str,
) -> None:
    from litellm.proxy import proxy_server

    user_key: Final = "sk-other-models-only"
    user_api_key_cache: Final = UserApiKeyCache()
    user_api_key_cache.set_cache(
        key=proxy_server.hash_token(user_key),
        value=UserAPIKeyAuth(token=proxy_server.hash_token(user_key), models=["some-other-group"]),
    )
    params, headers = _selected(selector)

    with pytest.MonkeyPatch.context() as key_auth:
        key_auth.setattr(proxy_server, "user_api_key_cache", user_api_key_cache)
        key_auth.setattr(proxy_server, "master_key", MASTER_KEY)
        key_auth.setattr(proxy_server, "prisma_client", "no-db-needed-for-a-cached-key")
        response: Final = interactions_proxy.request(
            method, path, params=params, headers={**headers, "authorization": f"Bearer {user_key}"}
        )

    assert response.status_code == 403, response.text
    assert _VERTEX_GROUP in response.text
    assert not respx_mock.calls


_DEEP_RESEARCH_USAGE: Final = {
    "total_input_tokens": 1000,
    "total_cached_tokens": 200,
    "total_output_tokens": 300,
    "total_thought_tokens": 100,
    "total_tokens": 1400,
    "input_tokens_by_modality": [{"modality": "text", "tokens": 1000}],
    "output_tokens_by_modality": [{"modality": "text", "tokens": 300}],
    "grounding_tool_count": [{"type": "google_search", "count": 3}],
}


def _deep_research_cost_from_the_vertex_row() -> float:
    row: Final = litellm.model_cost[f"vertex_ai/{_DEEP_RESEARCH_AGENT}"]
    return (
        (1000 - 200) * row["input_cost_per_token"]
        + 200 * row["cache_read_input_token_cost"]
        + (300 + 100) * row["output_cost_per_token"]
        + 3 * row["search_context_cost_per_query"]["search_context_size_medium"]
    )


@pytest.mark.parametrize("model_group", (_VERTEX_DEEP_RESEARCH_GROUP, _VERTEX_GROUP))
def test_routed_create_with_an_agent_sends_the_agent_with_the_deployment_credentials_and_bills_the_agent(
    interactions_proxy: TestClient, respx_mock: respx.MockRouter, model_group: str
) -> None:
    upstream: Final = respx_mock.post(_VERTEX_COLLECTION_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": _INTERACTION_ID,
                "object": "interaction",
                "agent": _DEEP_RESEARCH_AGENT,
                "status": "completed",
                "usage": _DEEP_RESEARCH_USAGE,
            },
        )
    )

    response: Final = interactions_proxy.post(
        "/v1beta/interactions",
        json={"model": model_group, "agent": _DEEP_RESEARCH_AGENT, "input": "Compare grid-scale storage options."},
    )

    assert response.status_code == 200, response.text
    assert json.loads(upstream.calls.last.request.content) == {
        "agent": _DEEP_RESEARCH_AGENT,
        "input": "Compare grid-scale storage options.",
    }
    assert upstream.calls.last.request.headers["authorization"] == "Bearer deployment-token"
    assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(
        _deep_research_cost_from_the_vertex_row()
    )


def test_routed_create_without_an_agent_still_sends_the_deployment_model(
    interactions_proxy: TestClient, respx_mock: respx.MockRouter
) -> None:
    upstream: Final = respx_mock.post(_VERTEX_COLLECTION_URL).mock(
        return_value=httpx.Response(200, json={"id": _INTERACTION_ID, "object": "interaction", "status": "completed"})
    )

    response: Final = interactions_proxy.post("/v1beta/interactions", json={"model": _VERTEX_GROUP, "input": "Hi"})

    assert response.status_code == 200, response.text
    assert json.loads(upstream.calls.last.request.content) == {"model": "gemini-4-argon", "input": "Hi"}
    assert upstream.calls.last.request.headers["authorization"] == "Bearer deployment-token"
