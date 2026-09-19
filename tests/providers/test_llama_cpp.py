"""LlamaCppProvider against a fake ``llama_cpp`` module (issue #37). No model needed.

The fake replays the shapes measured on llama-cpp-python 0.3.35 with Qwen3-4B-GGUF
Q4_K_M: ``create_completion(stream=True)`` yields ``{"choices": [{"text": ...}]}`` and
no usage at all, tool calls arrive as raw ``<tool_call>{json}</tool_call>`` text, a
thinking model writes ``<think>`` inline, and a full window raises
``ValueError("Requested tokens (N) exceed context window of M")``.

It is injected into ``sys.modules`` (both ``llama_cpp`` and its
``llama_chat_format`` submodule, plus the attribute on the parent), so these tests run
whether or not the real package is installed -- the provider imports it lazily, inside
the functions, so ``sys.modules`` is what it resolves. Real-model coverage lives in
``tests/integration/test_llama_cpp_model.py`` behind the ``llama`` marker.

The fake tokenizer counts one token per whitespace-separated word, which is what makes
the reported usage checkable without a model.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from kennel import (
    Agent,
    ConfigurationError,
    ContextLimitError,
    ModelUnavailableError,
    ProviderError,
    TurnCancelledError,
    Usage,
)
from kennel.providers.llama_cpp import (
    LlamaCppProvider,
    _CompletionParser,
    llama_cpp_doctor_checks,
)
from kennel.providers.registry import create, names

# -- the fake llama_cpp module ------------------------------------------------


@dataclass
class Script:
    """One scripted completion: the text pieces ``create_completion`` streams."""

    pieces: list[str] = field(default_factory=lambda: ["ok"])
    finish_reason: str = "stop"
    error: BaseException | None = None  # raised instead of streaming
    delay: float = 0.0  # seconds before each piece, so a turn can be interrupted

    @property
    def raw(self) -> str:
        return "".join(self.pieces)


@dataclass
class Fake:
    """What the fake module did, and what it should do next."""

    scripts: list[Script] = field(default_factory=lambda: [Script()])
    chat_template: str | None = "{{ messages }}"
    loads: list[dict[str, Any]] = field(default_factory=list)
    renders: list[dict[str, Any]] = field(default_factory=list)
    completions: list[dict[str, Any]] = field(default_factory=list)
    grammars: list[str] = field(default_factory=list)
    formatter_args: dict[str, Any] = field(default_factory=dict)

    def answering(self, *raws: str) -> None:
        """Script one completion per raw output text, streamed in one piece each."""
        self.scripts = [Script([raw]) for raw in raws]


@dataclass
class Rendered:
    """Stands in for ``llama_chat_format.ChatFormatterResponse``."""

    prompt: str
    stop: list[str] = field(default_factory=lambda: ["<|im_end|>"])
    stopping_criteria: Any = None


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> Fake:
    state = Fake()

    class FakeLlama:
        def __init__(self, **kwargs: Any) -> None:
            state.loads.append(kwargs)
            self.metadata = {"tokenizer.chat_template": state.chat_template} if state.chat_template else {}
            self.closed = False

        def token_eos(self) -> int:
            return 151645

        def token_bos(self) -> int:
            return -1  # Qwen3 renders no bos, so the provider must tolerate this

        def detokenize(self, tokens: list[int], special: bool = False) -> bytes:
            assert special, "special tokens must be asked for by name"
            return b"<|im_end|>"

        def tokenize(self, text: bytes, add_bos: bool = True, special: bool = False) -> list[int]:
            assert add_bos is False and special is True  # the template already has them
            return list(range(len(text.decode("utf-8").split())))

        def create_completion(self, tokens: list[int], **kwargs: Any) -> Any:
            state.completions.append({"tokens": tokens, **kwargs})
            script = state.scripts.pop(0) if len(state.scripts) > 1 else state.scripts[0]
            if script.error is not None:
                raise script.error

            def stream() -> Any:
                for piece in script.pieces:
                    if script.delay:
                        time.sleep(script.delay)
                    yield {"choices": [{"text": piece, "finish_reason": None, "index": 0}]}
                yield {"choices": [{"text": "", "finish_reason": script.finish_reason, "index": 0}]}

            return stream()

        def close(self) -> None:
            self.closed = True

    class FakeFormatter:
        def __init__(self, **kwargs: Any) -> None:
            state.formatter_args = kwargs

        def __call__(self, *, messages: list[dict[str, Any]], tools: Any = None, **kwargs: Any) -> Rendered:
            # A real chat template is Jinja and reads message.content directly, so a
            # missing key is a render error, not a None.
            assert all("content" in message for message in messages), messages
            prompt = "PROMPT " + " ".join(str(message["content"]) for message in messages)
            state.renders.append({"messages": messages, "tools": tools, "kwargs": kwargs, "prompt": prompt})
            return Rendered(prompt)

    class FakeGrammar:
        @classmethod
        def from_json_schema(cls, json_schema: str, verbose: bool = True) -> str:
            assert verbose is False  # a grammar dump must never reach stdout
            state.grammars.append(json_schema)
            return f"grammar:{json_schema}"

    module = types.ModuleType("llama_cpp")
    module.__version__ = "0.3.35+fake"  # type: ignore[attr-defined]
    module.Llama = FakeLlama  # type: ignore[attr-defined]
    module.LlamaGrammar = FakeGrammar  # type: ignore[attr-defined]
    chat_format = types.ModuleType("llama_cpp.llama_chat_format")
    chat_format.Jinja2ChatFormatter = FakeFormatter  # type: ignore[attr-defined]
    module.llama_chat_format = chat_format  # type: ignore[attr-defined]
    # Both the submodule entry and the parent attribute, so `from llama_cpp.llama_chat_format
    # import ...` resolves to the fake even when the real package is installed.
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_chat_format", chat_format)
    return state


def gguf(tmp_path: Path) -> Path:
    model = tmp_path / "fake-model.gguf"
    model.write_bytes(b"GGUF" + bytes(2048))
    return model


def provider_for(tmp_path: Path, **options: Any) -> LlamaCppProvider:
    options.setdefault("n_ctx", 4096)
    return LlamaCppProvider(model_path=str(gguf(tmp_path)), **options)


# -- AC-1: what the raw completion turns into --------------------------------

RAW_TOOL_CALL = (
    "Let me look. "
    "<think>I should read the file before answering.</think>"
    "Reading it now."
    '<tool_call>\n{"name": "read", "arguments": {"path": "notes/todo.md"}}\n</tool_call>'
)

CHUNKINGS = {
    "one chunk": [RAW_TOOL_CALL],
    "one character at a time": list(RAW_TOOL_CALL),
    "split inside every tag": [
        "Let me look. <thi",
        "nk>I should read the file",
        " before answering.</think",
        ">Reading it now.<tool",
        '_call>\n{"name": "re',
        'ad", "arguments": {"path": "notes/todo.md"}}',
        "\n</tool_",
        "call>",
    ],
}


@pytest.mark.parametrize("chunking", list(CHUNKINGS), ids=list(CHUNKINGS))
async def test_text_thinking_and_tool_calls_are_separated(
    fake: Fake, tmp_path: Path, meeting_ws: Path, chunking: str
):
    """However the model's output is chunked, the same three things come out of it."""
    pieces = CHUNKINGS[chunking]
    assert "".join(pieces) == RAW_TOOL_CALL
    fake.scripts = [Script(pieces, finish_reason="stop"), Script(["Two items."])]

    agent = Agent(meeting_ws, provider=provider_for(tmp_path), tools=["read"])
    session = agent.new_session()
    deltas: list[str] = []
    result = await session.run("what is on the list?", on_delta=deltas.append)

    assert "".join(deltas) == "Let me look. Reading it now.Two items."
    assert result.text == "".join(deltas)
    assert "<" not in result.text and "think" not in result.text
    assert [(call.name, call.status) for call in result.tool_calls] == [("read", "ok")]
    await session.close()


async def test_the_tool_call_arguments_reach_the_tool(fake: Fake, tmp_path: Path, meeting_ws: Path):
    fake.scripts = [Script([RAW_TOOL_CALL]), Script(["Done."])]
    agent = Agent(meeting_ws, provider=provider_for(tmp_path), tools=["read"])
    result = await agent.run("read it")

    assert result.tool_calls[0].arguments == {"path": "notes/todo.md"}
    # The round trip is rendered by the model's own template: the second request carries
    # the assistant message with its tool_calls and the tool result that answered it.
    request = fake.renders[1]["messages"]
    assert request[-2]["tool_calls"][0]["function"] == {
        "name": "read",
        "arguments": '{"path": "notes/todo.md"}',
    }
    assert request[-2]["tool_calls"][0]["id"] == "call_0"
    assert request[-1]["role"] == "tool" and request[-1]["tool_call_id"] == "call_0"


async def test_a_wordless_tool_call_still_renders_a_content_key(
    fake: Fake, tmp_path: Path, meeting_ws: Path
):
    """Qwen3's template reads message.content directly: a missing key breaks the render."""
    fake.scripts = [
        Script(['<tool_call>{"name": "read", "arguments": {"path": "notes/todo.md"}}</tool_call>']),
        Script(["Done."]),
    ]
    agent = Agent(meeting_ws, provider=provider_for(tmp_path), tools=["read"])
    await agent.run("read it")

    assert fake.renders[1]["messages"][-2]["content"] == ""  # the assistant said nothing
    assert all("content" in message for message in fake.renders[1]["messages"])


async def test_usage_is_counted_by_the_tokenizer(fake: Fake, tmp_path: Path, meeting_ws: Path):
    """Principle 4: the prompt count is exact, the output count is the tokenizer's own."""
    fake.answering("The answer is four.")
    agent = Agent(meeting_ws, provider=provider_for(tmp_path), tools=[])
    session = agent.new_session()
    result = await session.run("how many?")

    prompt_tokens = len(fake.renders[0]["prompt"].split())
    assert result.usage == Usage(input_tokens=prompt_tokens, output_tokens=4)
    usage = session.context_usage()
    assert usage.estimated is False
    assert usage.used_tokens == prompt_tokens + 4
    assert usage.window_tokens == 4096
    assert "estimated" not in usage.summary()
    await session.close()


# -- the parser's own edge cases ----------------------------------------------


def parse(*pieces: str) -> tuple[str, list[tuple[str | None, str]]]:
    """Feed a completion to the parser and return (answer text, tool calls)."""
    parser = _CompletionParser()
    chunks = [chunk for piece in pieces for chunk in parser.feed(piece)]
    chunks += parser.finish()
    text = "".join(chunk.text for chunk in chunks)
    calls = [(delta.name, delta.arguments) for chunk in chunks for delta in chunk.tool_calls]
    return text, calls


def test_text_outside_the_tags_survives_on_both_sides():
    text, calls = parse("before <tool_call>{\"name\": \"glob\", \"arguments\": {}}</tool_call> after")
    assert text == "before  after"
    assert calls == [("glob", "{}")]


def test_two_tool_calls_get_their_own_index_and_id():
    parser = _CompletionParser()
    chunks = parser.feed(
        '<tool_call>{"name": "read", "arguments": {"path": "a"}}</tool_call>'
        '<tool_call>{"name": "read", "arguments": {"path": "b"}}</tool_call>'
    )
    deltas = [delta for chunk in chunks for delta in chunk.tool_calls]
    assert [(d.index, d.id, d.arguments) for d in deltas] == [
        (0, "call_0", '{"path": "a"}'),
        (1, "call_1", '{"path": "b"}'),
    ]


def test_thinking_is_dropped_even_when_it_is_never_closed():
    assert parse("Answer.<think>still thinking") == ("Answer.", [])


def test_a_tool_call_the_stop_string_truncated_is_still_a_call():
    text, calls = parse('<tool_call>\n{"name": "grep", "arguments": {"pattern": "x"}}')
    assert text == "" and calls == [("grep", '{"pattern": "x"}')]


def test_a_stray_closing_think_tag_never_reaches_the_answer():
    """The template prefills the opening tag when thinking is off; a model may still close it."""
    text, calls = parse("\n\n</think>\n\nThe answer.")
    assert "think" not in text and calls == []
    assert text.strip() == "The answer."  # only the tag is dropped, not the text around it


@pytest.mark.parametrize(
    "payload",
    ['{"name": "read", "arguments": ', '["read"]', '{"arguments": {"path": "a"}}', "not json at all"],
)
def test_an_unreadable_tool_call_keeps_its_name_empty(payload: str):
    """It is forwarded nameless, not guessed at and not turned into answer text."""
    text, calls = parse(f"<tool_call>{payload}</tool_call>")
    assert text == ""
    assert calls == [(None, payload.strip())]


async def test_an_unreadable_tool_call_never_reaches_a_tool(fake: Fake, tmp_path: Path, meeting_ws: Path):
    fake.scripts = [Script(["<tool_call>{oops}</tool_call>"]), Script(["Sorry about that."])]
    agent = Agent(meeting_ws, provider=provider_for(tmp_path), tools=["read"])
    result = await agent.run("read it")

    assert result.tool_calls == []  # nothing reached ToolRunner.invoke
    assert result.text == "Sorry about that."
    assert "tool" in fake.renders[1]["messages"][-1]["content"].lower()  # the model was told


# -- AC-2: availability is answered without loading --------------------------


async def test_a_missing_model_file_is_unavailable(fake: Fake, tmp_path: Path):
    provider = LlamaCppProvider(model_path=str(tmp_path / "absent.gguf"))
    with pytest.raises(ModelUnavailableError) as exc:
        provider.check_availability()
    assert str(tmp_path / "absent.gguf") in str(exc.value)
    assert "download" in str(exc.value)  # Kennel does not fetch it for you
    assert fake.loads == []


async def test_create_session_checks_availability_before_loading(fake: Fake, tmp_path: Path):
    """An SDK caller who never asked still gets the real reason, not a FileNotFoundError."""
    provider = LlamaCppProvider(model_path=str(tmp_path / "absent.gguf"))
    with pytest.raises(ModelUnavailableError):
        await provider.create_session(instructions="", tools=[], invoke=None)
    assert fake.loads == []


def test_a_missing_package_points_at_the_extra(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setitem(sys.modules, "llama_cpp", None)  # makes `import llama_cpp` fail
    provider = LlamaCppProvider(model_path=str(gguf(tmp_path)))
    with pytest.raises(ModelUnavailableError) as exc:
        provider.check_availability()
    assert "kennel[llama-cpp]" in str(exc.value)


def test_a_zero_context_window_is_a_configuration_error(tmp_path: Path):
    with pytest.raises(ConfigurationError):
        LlamaCppProvider(model_path=str(gguf(tmp_path)), n_ctx=0)


# -- AC-3: what the provider declares, and how often it loads ----------------


def test_the_provider_declares_local_and_its_window(tmp_path: Path):
    """Principle 1: in-process inference opens no port, so this needs no I/O to know."""
    provider = provider_for(tmp_path, n_ctx=8192)
    assert provider.info.name == "llama-cpp"
    assert provider.info.mode == "local"
    assert provider.info.context_window_tokens == 8192
    assert provider.info.model == "fake-model.gguf"


async def test_the_model_is_loaded_once_and_shared(fake: Fake, tmp_path: Path):
    provider = provider_for(tmp_path, n_gpu_layers=12, llama_kwargs={"n_threads": 3})
    first = await provider.create_session(instructions="a", tools=[], invoke=None)
    second = await provider.create_session(instructions="b", tools=[], invoke=None)

    assert len(fake.loads) == 1
    assert fake.loads[0]["n_ctx"] == 4096
    assert fake.loads[0]["n_gpu_layers"] == 12
    assert fake.loads[0]["n_threads"] == 3
    assert fake.loads[0]["verbose"] is False  # stdout is a machine-readable contract
    assert fake.formatter_args["bos_token"] == ""  # token_bos() == -1
    assert fake.formatter_args["stop_token_ids"] == [151645]
    await first.close()
    await second.close()
    assert len(fake.loads) == 1  # closing a session does not unload the model


async def test_a_gguf_without_a_chat_template_is_unavailable(fake: Fake, tmp_path: Path):
    fake.chat_template = None
    provider = provider_for(tmp_path)
    with pytest.raises(ModelUnavailableError) as exc:
        await provider.create_session(instructions="", tools=[], invoke=None)
    assert "tokenizer.chat_template" in str(exc.value)


# -- AC-4: interrupting a generation -----------------------------------------


async def test_interrupt_stops_at_the_next_token(fake: Fake, tmp_path: Path, meeting_ws: Path):
    fake.scripts = [Script(["tick "] * 40, delay=0.02), Script(["after"])]
    agent = Agent(meeting_ws, provider=provider_for(tmp_path), tools=[])
    session = agent.new_session()

    task = asyncio.create_task(session.run("count for a while"))
    await asyncio.sleep(0.1)
    started = time.perf_counter()
    session.interrupt()
    with pytest.raises(TurnCancelledError):
        await asyncio.wait_for(task, 2.0)
    assert time.perf_counter() - started < 1.0

    # The worker closed the generator and let go of the lock, so the session still works.
    assert (await session.run("again")).text == "after"
    await session.close()
    await asyncio.sleep(0.1)
    assert not [t for t in threading.enumerate() if t.name == "kennel-llama-cpp"]


# -- AC-5: what is passed to the chat template -------------------------------


async def test_thinking_is_disabled_by_default(fake: Fake, tmp_path: Path, meeting_ws: Path):
    agent = Agent(meeting_ws, provider=provider_for(tmp_path), tools=[])
    await agent.run("hi")
    assert fake.renders[0]["kwargs"] == {"enable_thinking": False}


async def test_empty_chat_template_kwargs_passes_nothing(fake: Fake, tmp_path: Path, meeting_ws: Path):
    agent = Agent(meeting_ws, provider=provider_for(tmp_path, chat_template_kwargs={}), tools=[])
    await agent.run("hi")
    assert fake.renders[0]["kwargs"] == {}


async def test_chat_template_kwargs_replace_the_default(fake: Fake, tmp_path: Path, meeting_ws: Path):
    provider = provider_for(tmp_path, chat_template_kwargs={"enable_thinking": True})
    agent = Agent(meeting_ws, provider=provider, tools=[])
    await agent.run("hi")
    assert fake.renders[0]["kwargs"] == {"enable_thinking": True}


async def test_tools_are_rendered_by_the_template(fake: Fake, tmp_path: Path, meeting_ws: Path):
    fake.answering("no tools needed")
    agent = Agent(meeting_ws, provider=provider_for(tmp_path), tools=["read", "glob"])
    await agent.run("hi")
    assert [tool["function"]["name"] for tool in fake.renders[0]["tools"]] == ["read", "glob"]


async def test_no_tools_renders_none_not_an_empty_list(fake: Fake, tmp_path: Path, meeting_ws: Path):
    agent = Agent(meeting_ws, provider=provider_for(tmp_path), tools=[])
    await agent.run("hi")
    assert fake.renders[0]["tools"] is None  # what a template's `{% if tools %}` expects


async def test_the_completion_is_not_capped_at_the_library_default(
    fake: Fake, tmp_path: Path, meeting_ws: Path
):
    """create_completion defaults to 16 tokens; None means "as much as the window allows"."""
    agent = Agent(meeting_ws, provider=provider_for(tmp_path), tools=[])
    await agent.run("hi")
    assert fake.completions[0]["max_tokens"] is None
    assert fake.completions[0]["stream"] is True
    assert fake.completions[0]["stop"] == ["<|im_end|>"]
    assert fake.completions[0]["grammar"] is None


async def test_guided_generation_becomes_a_json_grammar(fake: Fake, tmp_path: Path, meeting_ws: Path):
    schema = {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}
    fake.answering('{"title": "ok"}')
    agent = Agent(meeting_ws, provider=provider_for(tmp_path), tools=[])
    result = await agent.run("give me a title", schema=schema)

    assert result.structured_output == {"title": "ok"}
    assert json.loads(fake.grammars[0]) == schema
    assert fake.completions[0]["grammar"] == f"grammar:{json.dumps(schema)}"


# -- AC-6: a full window ------------------------------------------------------


async def test_a_full_window_becomes_a_context_limit_error(fake: Fake, tmp_path: Path, meeting_ws: Path):
    fake.scripts = [Script(error=ValueError("Requested tokens (6595) exceed context window of 2048"))]
    agent = Agent(meeting_ws, provider=provider_for(tmp_path), tools=[])
    with pytest.raises(ContextLimitError):
        await agent.run("far too much")


async def test_the_compact_and_retry_path_still_applies(fake: Fake, tmp_path: Path, meeting_ws: Path):
    fake.scripts = [
        Script(["first"]),
        Script(error=ValueError("Requested tokens (9000) exceed context window of 4096")),
        Script(["second"]),
    ]
    agent = Agent(meeting_ws, provider=provider_for(tmp_path), tools=[])
    session = agent.new_session()
    assert (await session.run("one")).text == "first"
    result = await session.run("two")

    assert result.text == "second" and result.compactions == 1
    assert len(fake.loads) == 1  # the fresh provider session reuses the loaded model
    await session.close()


async def test_another_generation_failure_is_a_provider_error(fake: Fake, tmp_path: Path, meeting_ws: Path):
    fake.scripts = [Script(error=RuntimeError("llama_decode returned -1"))]
    agent = Agent(meeting_ws, provider=provider_for(tmp_path), tools=[])
    with pytest.raises(ProviderError) as exc:
        await agent.run("hi")
    assert "llama_decode" in str(exc.value)


# -- the registry and doctor wiring -------------------------------------------


def test_the_provider_is_registered_by_name(tmp_path: Path):
    assert "llama-cpp" in names()
    provider = create("llama-cpp", model_path=str(gguf(tmp_path)), n_ctx=2048)
    assert isinstance(provider, LlamaCppProvider)
    assert provider.info.context_window_tokens == 2048


def test_a_model_path_with_a_tilde_is_expanded():
    assert LlamaCppProvider(model_path="~/models/qwen.gguf").model_path == Path.home() / "models/qwen.gguf"


def test_doctor_checks_report_the_package_and_the_model(fake: Fake, tmp_path: Path):
    checks = llama_cpp_doctor_checks(model_path=str(gguf(tmp_path)), n_ctx=8192)
    assert [check.ok for check in checks] == [True, True]
    assert "0.3.35+fake" in checks[0].detail
    assert "GiB" in checks[1].detail and "8192" in checks[1].detail and "mode: local" in checks[1].detail
    assert fake.loads == []  # doctor must never pay for a load


def test_doctor_checks_explain_a_missing_model(fake: Fake, tmp_path: Path):
    checks = llama_cpp_doctor_checks(model_path=str(tmp_path / "absent.gguf"))
    assert [check.ok for check in checks] == [True, False]
    assert "absent.gguf" in checks[1].detail and "kennel.json" in (checks[1].hint or "")


def test_doctor_checks_explain_a_missing_package(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    checks = llama_cpp_doctor_checks(model_path=str(gguf(tmp_path)))
    assert [check.ok for check in checks] == [False, False]
    assert "kennel[llama-cpp]" in (checks[0].hint or "")


def test_doctor_checks_explain_a_bad_option():
    checks = llama_cpp_doctor_checks(model_path="/tmp/x.gguf", n_ctx=-1)
    assert [check.ok for check in checks] == [False, False]
    assert "n_ctx" in checks[0].detail
