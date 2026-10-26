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

import base64
import binascii
import inspect
import json
import logging
import types
import typing
from collections.abc import Sequence as AbcSequence
from typing import Annotated
from typing import Any
from typing import AsyncIterator
from typing import Callable
from typing import Literal
from typing import Union

from pydantic import BaseModel
from pydantic import BeforeValidator
from pydantic import ConfigDict
from pydantic import Field
from pydantic import create_model
from pydantic import field_validator

from nat.plugin_api import Builder
from nat.plugin_api import FunctionGroup
from nat.plugin_api import FunctionGroupBaseConfig
from nat.plugin_api import SerializableSecretStr
from nat.plugin_api import register_function_group
from tavily import AsyncTavilyClient

from ._client import build_async_client_from_key
from ._client import resolve_api_key
from .parse_streaming import _accumulate_research_stream

logger = logging.getLogger(__name__)

# Transport settings stay hidden; Extract/Crawl/Map timeout also controls the API
# operation and is exposed below. Research always streams internally.
_HIDDEN_PARAMS: frozenset[str] = frozenset({"self", "kwargs", "timeout", "stream"})

_LIST_ORIGINS: tuple[Any, ...] = (list, tuple, set, frozenset, AbcSequence)

_RESEARCH_FILE_EXTENSIONS = (".txt", ".md", ".json")


class ResearchFile(BaseModel):
    """One Research file attachment, as documented by the Tavily API."""

    name: str = Field(description="Filename with a .txt, .md or .json extension.")
    data: str = Field(description="Base64-encoded file contents.")
    type: Literal["base64"] = "base64"

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        if not value.lower().endswith(_RESEARCH_FILE_EXTENSIONS):
            raise ValueError("file name must end with .txt, .md or .json")
        return value

    @field_validator("data")
    @classmethod
    def _check_data(cls, value: str) -> str:
        try:
            if base64.b64encode(base64.b64decode(value, validate=True)).decode("ascii") != value:
                raise ValueError("not canonical base64")
        except (binascii.Error, ValueError) as exc:
            raise ValueError("file data must be canonical base64") from exc
        return value


def _coerce_json_list(value: Any) -> Any:
    """Decode a JSON-encoded list string into a Python list.

    LLMs (and some framework tool wrappers) sometimes emit array-typed args as JSON
    strings — e.g. `urls='["https://a", "https://b"]'`. This pre-validator unwraps
    them before pydantic dispatches Union arms; without it, pydantic accepts the
    string under the `str` arm of `Union[List[str], str]` and the SDK then fails
    URL validation downstream.
    """
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                return value
            if isinstance(parsed, list):
                return parsed
    return value


# Documented Research limits: up to 20 include/exclude domains and 5 files.
_Domains = Annotated[list[str], Field(max_length=20), BeforeValidator(_coerce_json_list)]
_ResearchFiles = Annotated[list[ResearchFile], Field(max_length=5), BeforeValidator(_coerce_json_list)]


# Public API parameters currently forwarded through SDK **kwargs rather than named
# in the SDK method signature. Keep them visible in generated agent tool schemas.
_EXTRA_API_PARAMS: dict[str, dict[str, tuple[Any, str]]] = {
    "search": {
        "chunks_per_source": (int, "Chunks per source (1-3) for basic, advanced or fast."),
        "include_image_descriptions": (bool, "Include descriptions for returned images."),
        "safe_search": (bool, "Filter unsafe results; unavailable for fast/ultra-fast."),
        "include_published_date": (bool, "Include estimated published dates (beta)."),
        "filter_by_published_date": (bool, "Remove undated or out-of-window results."),
    },
    "research": {
        "include_domains": (_Domains, "Prefer these domains (up to 20); not a hard filter."),
        "exclude_domains": (_Domains, "Block these domains and their subdomains (up to 20)."),
        "output_length": (Literal["short", "standard", "long"], "Report length."),
        "files": (_ResearchFiles, "Up to 5 .txt/.md/.json files with base64 `data`, used as sources."),
    },
}
_API_TIMEOUT_METHODS = frozenset({"extract", "crawl", "map"})


def _annotation_accepts_list(annotation: Any) -> bool:
    """True if `annotation` is (or contains, in a Union) a list/sequence type."""
    origin = typing.get_origin(annotation)
    if origin in _LIST_ORIGINS:
        return True
    if origin in (Union, types.UnionType):
        return any(_annotation_accepts_list(a) for a in typing.get_args(annotation))
    return False


def _check_output_schema(cls: Any, value: Any) -> Any:
    """The API requires an `output_schema` object with a `properties` object."""
    if value is not None and not (isinstance(value, dict) and isinstance(value.get("properties"), dict)):
        raise ValueError("output_schema must be an object with a `properties` object")
    return value


def _build_input_schema(method: Callable, name: str) -> type[BaseModel]:
    """Generate a pydantic input model from a live SDK method signature.

    List-typed fields are wrapped in a BeforeValidator that JSON-decodes
    string-encoded lists from the LLM.
    """
    sig = inspect.signature(method)
    fields: dict[str, tuple[Any, Any]] = {}
    for param_name, param in sig.parameters.items():
        if param_name in _HIDDEN_PARAMS and not (
            param_name == "timeout" and method.__name__ in _API_TIMEOUT_METHODS
        ):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else Any

        if param.default is inspect.Parameter.empty:
            default = ...
        else:
            default = param.default
            if default is None:
                # SDK uses `T = None` as "use server default" — make Optional so pydantic
                # accepts the None default against a non-Optional Literal/scalar annotation.
                annotation = Union[annotation, None]

        if _annotation_accepts_list(annotation):
            annotation = Annotated[annotation, BeforeValidator(_coerce_json_list)]

        fields[param_name] = (annotation, Field(default=default))

    for param_name, (annotation, description) in _EXTRA_API_PARAMS.get(method.__name__, {}).items():
        if param_name not in fields:
            if typing.get_origin(annotation) is not Annotated and _annotation_accepts_list(annotation):
                annotation = Annotated[annotation, BeforeValidator(_coerce_json_list)]
            fields[param_name] = (Union[annotation, None], Field(default=None, description=description))

    validators = {}
    if method.__name__ == "research":
        validators["_check_output_schema"] = field_validator("output_schema")(_check_output_schema)

    return create_model(
        name,
        __config__=ConfigDict(arbitrary_types_allowed=True),
        __validators__=validators,
        **fields,
    )


# Schemas are a static reflection of the SDK signature
TavilySearchInput: type[BaseModel] = _build_input_schema(AsyncTavilyClient.search, "TavilySearchInput")
TavilyExtractInput: type[BaseModel] = _build_input_schema(AsyncTavilyClient.extract, "TavilyExtractInput")
TavilyCrawlInput: type[BaseModel] = _build_input_schema(AsyncTavilyClient.crawl, "TavilyCrawlInput")
TavilyMapInput: type[BaseModel] = _build_input_schema(AsyncTavilyClient.map, "TavilyMapInput")
TavilyResearchInput: type[BaseModel] = _build_input_schema(AsyncTavilyClient.research, "TavilyResearchInput")


class TavilyResearchStatusInput(BaseModel):
    request_id: str = Field(min_length=1, description="Research task id (`request_id`).")
    include_usage: bool | None = Field(default=None, description="Include credit usage.")

_DESCRIPTIONS: dict[str, str] = {
    "search":
        "Search the web with Tavily for up-to-date information. Returns ranked source documents "
        "(URL, title, content snippet, score) and optionally a synthesized answer.",
    "extract":
        "Extract clean text or markdown content from one or more URLs. Use this when you already "
        "have URLs (e.g. from a prior search) and need their full contents.",
    "crawl":
        "Crawl a website starting from a URL, following links to a configurable depth and breadth. "
        "Returns extracted content from each visited page. Use for bulk content retrieval from a "
        "single domain.",
    "map":
        "Discover and list URLs reachable from a starting URL on a domain, without extracting "
        "page content. Faster than crawl. Use to find specific pages on a known site.",
    "research_status":
        "Check a Tavily research task by `request_id`. Returns `status` pending or in_progress "
        "(not finished; poll again later) or completed or failed with the report, sources or error.",
    "research":
        "Run a deep, multi-source research task with citations. Returns a synthesized report "
        "(markdown or structured per `output_schema`) plus a list of sources. Slow (30-120s) — "
        "use only when the user explicitly asks for research-depth output. For quick fact-finding "
        "use `search`.",
}


class TavilyToolsGroupConfig(FunctionGroupBaseConfig, name="tavily"):
    """Tavily tools group: search, extract, crawl, map, research, research_status.

    All tools share a single AsyncTavilyClient and one API key. Per-call arguments
    come from the agent and are passed straight through to the SDK.
    """

    api_key: SerializableSecretStr = Field(
        default_factory=lambda: SerializableSecretStr(""),
        description="Tavily API key. Falls back to the TAVILY_API_KEY env var.",
    )

    api_base_url: str | None = None
    project_id: str | None = None
    session_id: str | None = None
    human_id: str | None = None

    research_timeout_seconds: float | None = Field(
        default=900.0,
        gt=0,
        description="HTTP timeout for the research SSE stream. None disables the client-side cap.",
    )
    research_include_trace: bool = Field(
        default=False,
        description="If True, include intermediate tool_call/tool_response events from the research "
        "stream in the returned dict (under `trace`). Useful for debugging; noisy for agents.",
    )


@register_function_group(config_type=TavilyToolsGroupConfig)
async def tavily_tools(config: TavilyToolsGroupConfig, _builder: Builder) -> AsyncIterator[FunctionGroup]:
    """Register the `tavily` function group."""
    api_key = resolve_api_key(config.api_key)
    client_options = {
        key: getattr(config, key)
        for key in ("api_base_url", "project_id", "session_id", "human_id")
        if getattr(config, key) is not None
    }

    async def _search(value: TavilySearchInput) -> dict:
        async with build_async_client_from_key(api_key, **client_options) as client:
            return await client.search(**value.model_dump(exclude_none=True))

    async def _extract(value: TavilyExtractInput) -> dict:
        async with build_async_client_from_key(api_key, **client_options) as client:
            return await client.extract(**value.model_dump(exclude_none=True))

    async def _crawl(value: TavilyCrawlInput) -> dict:
        async with build_async_client_from_key(api_key, **client_options) as client:
            return await client.crawl(**value.model_dump(exclude_none=True))

    async def _map(value: TavilyMapInput) -> dict:
        async with build_async_client_from_key(api_key, **client_options) as client:
            return await client.map(**value.model_dump(exclude_none=True))

    async def _research(value: TavilyResearchInput) -> dict:
        # `client.research` is async-def; awaiting it returns an AsyncGenerator[bytes] when
        # stream=True (or a dict when stream=False, which we don't use here).
        async with build_async_client_from_key(api_key, **client_options) as client:
            stream = await client.research(
                stream=True,
                timeout=config.research_timeout_seconds,
                **value.model_dump(exclude_none=True),
            )
            return await _accumulate_research_stream(stream, include_trace=config.research_include_trace)

    async def _research_status(value: TavilyResearchStatusInput) -> dict:
        async with build_async_client_from_key(api_key, **client_options) as client:
            return await client.get_research(**value.model_dump(exclude_none=True))

    group = FunctionGroup(config=config)
    group.add_function("search", _search, input_schema=TavilySearchInput, description=_DESCRIPTIONS["search"])
    group.add_function("extract", _extract, input_schema=TavilyExtractInput, description=_DESCRIPTIONS["extract"])
    group.add_function("crawl", _crawl, input_schema=TavilyCrawlInput, description=_DESCRIPTIONS["crawl"])
    group.add_function("map", _map, input_schema=TavilyMapInput, description=_DESCRIPTIONS["map"])
    group.add_function("research", _research, input_schema=TavilyResearchInput, description=_DESCRIPTIONS["research"])

    group.add_function("research_status", _research_status, input_schema=TavilyResearchStatusInput,
                       description=_DESCRIPTIONS["research_status"])

    yield group
