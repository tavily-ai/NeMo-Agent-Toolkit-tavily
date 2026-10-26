"""Offline Research stream contract regressions."""
import json

import pytest

from nat.plugins.tavily.parse_streaming import _accumulate_research_stream


async def chunks(parts):
    for part in parts:
        yield part


@pytest.mark.parametrize("parts", [[], [b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n']])
async def test_premature_eof_is_not_a_completed_report(parts):
    with pytest.raises(RuntimeError, match="before.*done"):
        await _accumulate_research_stream(chunks(parts), include_trace=False)


@pytest.mark.parametrize("separator", [b"\n\n", b"\r\n\r\n"])
async def test_fragmented_stream_preserves_report_sources_and_done(separator):
    payload = {"model": "mini", "choices": [{"delta": {
        "content": "a report", "sources": [{"url": "https://example.com"}]
    }}]}
    stream = b"data: " + json.dumps(payload).encode() + separator + b"event: done" + separator
    result = await _accumulate_research_stream(chunks([stream[i:i+1] for i in range(len(stream))]), include_trace=False)
    assert result == {"content": "a report", "sources": [{"url": "https://example.com"}], "model": "mini"}


async def test_error_event_is_not_a_completed_report():
    stream = b'data: {"object":"error","error":"synthetic failure"}\n\n'
    with pytest.raises(RuntimeError, match="synthetic failure"):
        await _accumulate_research_stream(chunks([stream]), include_trace=False)
