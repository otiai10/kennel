"""llama-cpp integration tests: a GGUF loaded into this process, no server.

Needs the extra and a model file::

    uv sync --extra dev --extra llama-cpp
    KENNEL_LLAMA_CPP_MODEL=~/models/Qwen3-4B-Q4_K_M.gguf pytest tests/integration -m llama

They assert structure, token counts and tool traces, never exact model wording.
The module-scoped provider loads the model once for the whole file.
"""

import os
from pathlib import Path

import pytest

from kennel import Agent, Usage

MODEL = os.environ.get("KENNEL_LLAMA_CPP_MODEL")

pytestmark = [
    pytest.mark.llama,
    pytest.mark.skipif(
        not MODEL, reason="set KENNEL_LLAMA_CPP_MODEL=<gguf path> to run llama-cpp integration tests"
    ),
]

FIXTURE = Path(__file__).parent.parent / "fixtures" / "meeting_project"


@pytest.fixture(scope="module")
def provider():
    from kennel.providers.llama_cpp import LlamaCppProvider

    p = LlamaCppProvider(model_path=MODEL, n_ctx=8192)
    p.check_availability()
    return p


@pytest.fixture
def agent(provider) -> Agent:
    return Agent(FIXTURE, provider=provider, tools=["glob", "grep", "read"])


def test_the_provider_declares_what_it_loaded(provider):
    assert provider.info.name == "llama-cpp"
    assert provider.info.mode == "local"  # in-process inference opens no port
    assert provider.info.model.endswith(".gguf")
    assert provider.info.context_window_tokens == 8192


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
    """The model's own <tool_call> format, parsed by the transport."""
    session = agent.new_session()
    result = await session.run("Read the file notes/todo.md and list what is on it.")
    assert result.tool_calls, "the model answered a file question without reading anything"
    assert {"glob", "grep", "read"} & {call.name for call in result.tool_calls}
    assert result.text.strip()
    await session.close()


async def test_no_markup_reaches_the_answer(agent: Agent):
    """Neither the thinking scratchpad nor a tool call may show up as text."""
    session = agent.new_session()
    result = await session.run("Read notes/todo.md and say in one sentence what it is about.")
    for markup in ("<think>", "</think>", "<tool_call>", "</tool_call>"):
        assert markup not in result.text
    await session.close()


async def test_usage_is_measured_not_estimated(agent: Agent):
    """Principle 4: the tokenizer counts, so context_usage() is not an estimate."""
    session = agent.new_session()
    result = await session.run("Reply with the single word: counted.")
    assert isinstance(result.usage, Usage)
    assert result.usage.input_tokens and result.usage.output_tokens
    usage = session.context_usage()
    assert usage.estimated is False
    assert usage.used_tokens == result.usage.input_tokens + result.usage.output_tokens
    assert usage.window_tokens == 8192
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


async def test_two_sessions_share_one_loaded_model(provider):
    first = await provider.create_session(instructions="Answer with one word.", tools=[], invoke=None)
    second = await provider.create_session(instructions="Answer with one word.", tools=[], invoke=None)
    assert (await first.respond("Say one.")).strip()
    assert (await second.respond("Say two.")).strip()
    await first.close()
    await second.close()
