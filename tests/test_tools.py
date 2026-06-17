# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026, Tavily AI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import builtins
import inspect
import importlib.metadata
import json
from typing import AsyncIterator

import pytest
from tavily import AsyncTavilyClient

from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugin_api import SerializableSecretStr
from nat.test import ToolTestRunner
from nat.plugins.tavily._client import build_async_client
from nat.plugins.tavily.tools import _HIDDEN_PARAMS
from nat.plugins.tavily.tools import TavilyCrawlInput
from nat.plugins.tavily.tools import TavilyExtractInput
from nat.plugins.tavily.tools import TavilyMapInput
from nat.plugins.tavily.tools import TavilyResearchInput
from nat.plugins.tavily.tools import TavilySearchInput
from nat.plugins.tavily.tools import TavilyToolsGroupConfig
from nat.plugins.tavily.tools import _accumulate_research_stream
from nat.plugins.tavily.tools import _build_input_schema


async def _stream_chunks(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


def _expected_fields(method) -> set[str]:
    sig = inspect.signature(method)
    return {
        name for name, p in sig.parameters.items()
        if name not in _HIDDEN_PARAMS
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }


async def test_package_entry_point_loads_tavily_plugin(monkeypatch):
    dist = importlib.metadata.distribution("nemo-agent-toolkit-tavily")
    plugin_entry_points = {ep.name: ep for ep in dist.entry_points if ep.group == "nat.plugins"}

    assert plugin_entry_points["nat_tavily"].value == "nat.plugins.tavily.register"
    assert not [ep for ep in dist.entry_points if ep.group == "nat.components"]

    module = plugin_entry_points["nat_tavily"].load()
    assert module.__name__ == "nat.plugins.tavily.register"

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    function_names = await ToolTestRunner().test_function_group(config_type=TavilyToolsGroupConfig)
    assert "tavily__search" in function_names


@pytest.mark.parametrize("schema, method, required_field", [
    (TavilySearchInput, AsyncTavilyClient.search, "query"),
    (TavilyExtractInput, AsyncTavilyClient.extract, "urls"),
    (TavilyCrawlInput, AsyncTavilyClient.crawl, "url"),
    (TavilyMapInput, AsyncTavilyClient.map, "url"),
    (TavilyResearchInput, AsyncTavilyClient.research, "input"),
])
def test_schema_mirrors_sdk_signature(schema, method, required_field):
    assert set(schema.model_fields) == _expected_fields(method)
    assert schema.model_fields[required_field].is_required()
    for name, field in schema.model_fields.items():
        if name == required_field:
            continue
        assert not field.is_required(), f"{schema.__name__}.{name} should default"


def test_research_schema_hides_stream_and_timeout():
    """`stream` is locked to True by the wrapper; `timeout` is wrapper-config-only."""
    assert "stream" not in TavilyResearchInput.model_fields
    assert "timeout" not in TavilyResearchInput.model_fields
    assert "self" not in TavilyResearchInput.model_fields


def test_schemas_are_independent():
    """Each tool's schema is its own class; they don't collide."""
    classes = {TavilySearchInput, TavilyExtractInput, TavilyCrawlInput, TavilyMapInput, TavilyResearchInput}
    assert len(classes) == 5


def test_build_input_schema_skips_var_kwargs():
    """**kwargs and timeout are dropped from the LLM-facing surface."""
    schema = _build_input_schema(AsyncTavilyClient.search, "S")
    assert "kwargs" not in schema.model_fields
    assert "timeout" not in schema.model_fields
    assert "self" not in schema.model_fields


def test_build_input_schema_coerces_json_list_for_pep604_union():
    """PEP 604 list unions should get the same JSON-list pre-validation."""

    def method(urls: list[str] | str):
        pass

    schema = _build_input_schema(method, "S")

    value = schema(urls='["https://example.com"]')

    assert value.urls == ["https://example.com"]


@pytest.mark.parametrize("tool, method_name, payload, required_kwargs", [
    ("search", "search", {"query": "weather sf"}, {"query": "weather sf"}),
    ("extract", "extract", {"urls": ["https://example.com"]}, {"urls": ["https://example.com"]}),
    ("crawl", "crawl", {"url": "https://example.com", "limit": 5}, {"url": "https://example.com", "limit": 5}),
    ("map", "map", {"url": "https://example.com"}, {"url": "https://example.com"}),
])
async def test_each_tool_routes_to_correct_sdk_method(monkeypatch, tool, method_name, payload, required_kwargs):
    """Each group tool calls its corresponding AsyncTavilyClient method with the agent kwargs."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    captured: dict = {}

    async def fake(self, **kwargs):
        captured["called"] = method_name
        captured["kwargs"] = kwargs
        return {"ok": True, "tool": method_name}

    async def fake_close(self):
        captured["closed"] = captured.get("closed", 0) + 1

    monkeypatch.setattr(AsyncTavilyClient, method_name, fake)
    monkeypatch.setattr(AsyncTavilyClient, "close", fake_close)

    runner = ToolTestRunner()
    result = await runner.test_function_group_tool(config_type=TavilyToolsGroupConfig,
                                                   function_name=tool,
                                                   input_kwargs=payload)

    assert captured["called"] == method_name
    for k, v in required_kwargs.items():
        assert captured["kwargs"][k] == v
    assert captured["closed"] == 1
    assert result == {"ok": True, "tool": method_name}


async def test_group_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    runner = ToolTestRunner()
    with pytest.raises(ValueError, match="Tavily API key"):
        await runner.test_function_group(config_type=TavilyToolsGroupConfig)


def test_build_async_client_rejects_whitespace_config_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Tavily API key"):
        build_async_client(SerializableSecretStr("   "))


def test_build_async_client_rejects_whitespace_env_api_key(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "   ")

    with pytest.raises(ValueError, match="Tavily API key"):
        build_async_client(None)


async def test_group_exposes_all_five_tools(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    runner = ToolTestRunner()
    names = await runner.test_function_group(config_type=TavilyToolsGroupConfig)

    assert names == {
        "tavily__search",
        "tavily__extract",
        "tavily__crawl",
        "tavily__map",
        "tavily__research",
    }


async def test_core_workflow_builder_invokes_search_without_langchain(monkeypatch):
    """Smoke-test the NAT core path without relying on a framework-specific agent wrapper."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    captured: dict = {}

    async def fake_search(self, **kwargs):
        captured["kwargs"] = kwargs
        return {"answer": "sunny", "results": []}

    monkeypatch.setattr(AsyncTavilyClient, "search", fake_search)

    original_import = builtins.__import__

    def reject_langchain_imports(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "langchain" or name.startswith(("langchain.", "langchain_")):
            raise AssertionError(f"Core Tavily smoke path unexpectedly imported {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_langchain_imports)

    async with WorkflowBuilder() as builder:
        group = await builder.add_function_group("tavily", TavilyToolsGroupConfig(include=["search"]))
        accessible = await group.get_accessible_functions()
        assert set(accessible) == {"tavily__search"}

        search = await builder.get_function("tavily__search")
        result = await search.ainvoke({"query": "weather sf"})

    assert captured["kwargs"] == {"query": "weather sf"}
    assert result == {"answer": "sunny", "results": []}


async def test_research_stream_consumes_final_sse_block_without_trailing_separator():
    payloads = [
        {
            "model": "tavily-research",
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "type": "Planning",
                        "content": "Plan the query",
                    }]
                }
            }],
        },
        {
            "model": "tavily-research",
            "choices": [{
                "delta": {
                    "content": "final report",
                }
            }],
        },
        {
            "model": "tavily-research",
            "choices": [{
                "delta": {
                    "sources": [{
                        "title": "Example",
                        "url": "https://example.com",
                    }]
                }
            }],
        },
    ]
    chunks = [
        f"data: {json.dumps(payloads[0])}\n\n".encode(),
        f"data: {json.dumps(payloads[1])}\n\n".encode(),
        f"data: {json.dumps(payloads[2])}".encode(),
    ]

    result = await _accumulate_research_stream(_stream_chunks(chunks), include_trace=True)

    assert result == {
        "content": "final report",
        "sources": [{
            "title": "Example",
            "url": "https://example.com",
        }],
        "model": "tavily-research",
        "trace": [[{
            "type": "Planning",
            "content": "Plan the query",
        }]],
    }
