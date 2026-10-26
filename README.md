# nemo-agent-toolkit-tavily

Tavily integration for [NVIDIA NeMo Agent Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit). Provides web search, URL extraction, site crawling, URL discovery, and deep research tools — framework-agnostic and auto-discovered by NAT at runtime.

## Install

```bash
pip install nemo-agent-toolkit-tavily
# or
uv add nemo-agent-toolkit-tavily
```

NAT auto-discovers the plugin via `importlib.metadata.entry_points()`. No additional NAT configuration required.

## Compatibility

This package supports NVIDIA NeMo Agent Toolkit `>=1.8,<2.0`.

## Configuration

### `TavilyToolsGroupConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `api_key` | `str` (secret) | `""` | Tavily API key. Falls back to `TAVILY_API_KEY` env var. |
| `api_base_url` | `str \| None` | `None` | Optional compatible API proxy URL. |
| `project_id`, `session_id`, `human_id` | `str \| None` | `None` | Optional request tracking headers. SDK defaults apply when omitted. |
| `research_timeout_seconds` | `float \| None` | `900.0` | HTTP timeout for the research SSE stream. `None` disables the client-side cap. |
| `research_include_trace` | `bool` | `False` | Include intermediate tool_call events from the research stream in the output (under `trace`). Useful for debugging. |

### Tool selection

Use `include` and `exclude` on the function group to control which tools reach the agent:

```yaml
function_groups:
  tavily:
    _type: tavily
    include: [search, extract]   # only expose search + extract
```

```yaml
function_groups:
  tavily:
    _type: tavily
    exclude: [crawl, map]        # expose everything except crawl + map
```

## Available Tools

| Tool | Description |
|------|-------------|
| `tavily__search` | Search the web for up-to-date information |
| `tavily__extract` | Extract clean text/markdown from URLs |
| `tavily__crawl` | Crawl a website with configurable depth/breadth |
| `tavily__map` | Discover URLs on a domain without extracting content |
| `tavily__research` | Deep multi-source research with citations (30–120s) |
| `tavily__research_status` | Look up a research task by `request_id` (pending, in progress, completed or failed) |

Schemas expose the installed SDK's named options plus public Search chunk, safe-search,
image-description and published-date controls, and Research domain, output-length and
file options. Published-date metadata is beta. Research `include_domains` is a soft
preference; `exclude_domains` blocks domains. The minimum Tavily SDK is 0.8.5 so the
current named language/domain options and attribution are available.

Extract, Crawl and Map expose their API operation `timeout` (also used by the SDK for
the HTTP request). Search transport timeouts stay hidden; Research uses
`research_timeout_seconds`. Research raises an error if the stream ends before its
`done` event, rather than returning a partial report as complete. Both LF and CRLF
SSE separators are supported.

Research `files` take up to 5 `.txt`, `.md` or `.json` entries with base64 `data`; `include_domains`
and `exclude_domains` accept up to 20 entries; `output_schema` must contain `properties`.
Invalid values are rejected before any request is sent. In addition, Tavily rejects an `output_schema` whose properties lack a `description` or whose `required` array is missing or empty; the tool does not check this locally.

`tavily__research` streams, so it does not use the non-streaming create call. A separate
"submit and poll later" tool is not provided: `tavily-python` 0.8.5 treats only HTTP 200 as
success when creating a non-streaming Research task, while the API documents 201 (a fix is in
tavily-python PR #206, unreleased). `tavily__research_status` only reads tasks (HTTP 200/202),
so it works with 0.8.5.

## Minimal Workflow

```yaml
function_groups:
  tavily:
    _type: tavily

llms:
  my_llm:
    _type: litellm
    model_name: anthropic/claude-sonnet-4-6

workflow:
  _type: react_agent
  llm_name: my_llm
  tool_names:
    - tavily   # auto-expands to tavily__search, __extract, __crawl, __map, __research
```

```bash
export TAVILY_API_KEY=tvly-...
export ANTHROPIC_API_KEY=...

uv run nat run --config_file config.yml --input "What is the weather in San Francisco?"
```

## Bug Routing

- **Integration code bugs** → [this repo's issues](https://github.com/tavily-ai/NeMo-Agent-Toolkit-tavily/issues)
- **`nvidia-nat-core` bugs** → [NAT repo's issues](https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues)

## Development

```bash
git clone https://github.com/tavily-ai/NeMo-Agent-Toolkit-tavily.git
cd NeMo-Agent-Toolkit-tavily
uv sync --extra test
uv run pytest tests/ -v
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
