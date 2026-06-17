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
