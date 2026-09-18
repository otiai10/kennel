"""llama-server integration tests, against a server the developer is running.

Start one and point the tests at it::

    llama-server -hf Qwen/Qwen3-4B-GGUF:Q4_K_M --jinja -c 32768 --port 8080 --host 127.0.0.1
    KENNEL_LLAMA_SERVER=http://127.0.0.1:8080 pytest tests/integration -m llama

They assert structure, token counts and tool traces, never exact model wording.
"""

import os
from pathlib import Path

import pytest

from kennel import Agent, Usage

BASE_URL = os.environ.get("KENNEL_LLAMA_SERVER")

pytestmark = [
    pytest.mark.llama,
    pytest.mark.skipif(
        not BASE_URL, reason="set KENNEL_LLAMA_SERVER=<base_url> to run llama-server integration tests"
    ),
]

FIXTURE = Path(__file__).parent.parent / "fixtures" / "meeting_project"


@pytest.fixture(scope="module")
def provider():
    from kennel.providers.llama_server import LlamaServerProvider

    p = LlamaServerProvider(base_url=BASE_URL)
    p.check_availability()
    return p


@pytest.fixture
def agent(provider) -> Agent:
    return Agent(FIXTURE, provider=provider, tools=["glob", "grep", "read"])


def test_the_server_declares_what_it_is_running(provider):
    assert provider.info.name == "llama-server"
    assert provider.info.mode == "local"  # the default base_url is loopback
    assert provider.info.model  # whatever /v1/models reported
    assert provider.info.context_window_tokens and provider.info.context_window_tokens > 4096


async def test_simple_respond(provider):
    session = await provider.create_session(instructions="Answer with one word.", tools=[], invoke=None)
    text = await session.respond("Say hello.")
    assert isinstance(text, str) and text.strip()
    await session.close()


async def test_streaming_yields_answer_text(agent: Agent):
    session = agent.new_session()
    deltas: list[str] = []
    result = await session.run("Reply with the single word: ready.", on_delta=deltas.append)
    assert deltas and "".join(deltas) == result.text
    await session.close()


async def test_a_file_question_goes_through_the_tools(agent: Agent):
    session = agent.new_session()
    result = await session.run("Read the file notes/todo.md and list what is on it.")
    assert result.tool_calls, "the model answered a file question without reading anything"
    assert {"glob", "grep", "read"} & {call.name for call in result.tool_calls}
    assert result.text.strip()
    await session.close()


async def test_usage_is_measured_not_estimated(agent: Agent):
    """Principle 4: this is Kennel's first provider that really counts tokens."""
    session = agent.new_session()
    result = await session.run("Reply with the single word: counted.")
    assert isinstance(result.usage, Usage)
    assert result.usage.input_tokens and result.usage.output_tokens
    usage = session.context_usage()
    assert usage.estimated is False
    assert usage.used_tokens == result.usage.input_tokens + result.usage.output_tokens
    assert "estimated" not in usage.summary()
    await session.close()


async def test_guided_generation_returns_the_schema_shape(agent: Agent):
    schema = {
        "type": "object",
        "properties": {"topic": {"type": "string"}, "urgent": {"type": "boolean"}},
        "required": ["topic", "urgent"],
    }
    session = agent.new_session()
    result = await session.run("Invent one meeting topic and say whether it is urgent.", schema=schema)
    assert set(result.structured_output or {}) >= {"topic", "urgent"}
    assert isinstance(result.structured_output["urgent"], bool)
    await session.close()
