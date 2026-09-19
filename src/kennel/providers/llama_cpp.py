"""LlamaCppProvider: a GGUF model loaded into this process through llama-cpp-python.

The sibling ``providers/llama_server.py`` talks to a server the user runs; this one
owns the model itself, so nothing listens on a port and the strongest reading of
principle 1 is available: ``mode`` is ``local``, unconditionally. Kennel still never
downloads a model -- the user puts a ``.gguf`` somewhere and names it.

The tool loop is the shared one (:mod:`kennel.providers.chatloop`); this module is a
transport plus the one thing llama-cpp-python does not do for us.

Design notes (measured against llama-cpp-python 0.3.35, Qwen3-4B-GGUF Q4_K_M, Metal,
``n_ctx=8192``, 2026-09-19):

* **Kennel renders the prompt.** The library's generic handler passes ``tools`` to the
  chat template but does not parse what comes back, so ``tool_calls`` is only ever
  filled when a function name is forced. Instead the GGUF's own
  ``tokenizer.chat_template`` is rendered with ``Jinja2ChatFormatter`` and the raw
  ``<tool_call>{json}</tool_call>`` the model emits is parsed here
  (:class:`_CompletionParser`). The round trip -- an assistant message with
  ``tool_calls``, then a ``tool`` message -- is rendered by that same template. The
  supported wire format is therefore the Qwen3/Hermes one; a GGUF whose template
  ignores ``tools`` can still chat, but cannot call tools.
* **Thinking is off by default.** Left alone, Qwen3 spends a few hundred tokens in
  ``<think>`` before every tool call. ``create_chat_completion(enable_thinking=False)``
  raises ``TypeError``, but the formatter takes template kwargs, so
  ``{"enable_thinking": False}`` is the default here and prefills an empty think
  block. Any ``<think>`` block that still appears is dropped and never reaches
  ``ChatChunk.text``, the way llama-server's ``reasoning_content`` is.
* **One model, one KV cache.** A ``Llama`` cannot serve two generations at once, so
  the provider holds a :class:`threading.Lock` and every request takes it. The model
  is loaded on the first ``create_session()`` and shared by every session after that;
  ``check_availability()`` and ``kennel doctor`` only look at the import, the file and
  ``n_ctx``, so neither ``Agent()`` nor ``doctor`` pays for a load.
* **Usage is counted, not guessed** -- see :meth:`LlamaCppTransport.stream` and the
  ``usage`` note there.
* Overflowing the window is ``ValueError("Requested tokens (N) exceed context window
  of M")``, which maps to :class:`~kennel.errors.ContextLimitError` so
  ``Session.run``'s compact-and-retry applies unchanged.

Threading: ``create_completion`` is a blocking generator, so every request runs on a
worker thread that hands chunks to the caller's loop through an
:class:`asyncio.Queue` -- the same separation ``providers/llama_server.py`` uses for
blocking sockets. Leaving the iterator (a cancelled turn) sets an event the worker
checks **between tokens**, so ``interrupt()`` stops at the next token boundary; while
the prompt is still being evaluated (10.8 s for a 6,752-token prompt) nothing
interrupts it. The worker holds the lock for the whole generation, so the next
``run()`` starts only once the previous one has really stopped. The model is never
unloaded: it belongs to the provider, not to a session, so ``close()`` on a session
leaves it loaded for the next one.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..diagnostics import Check
from ..errors import (
    ConfigurationError,
    ContextLimitError,
    ModelUnavailableError,
    ProviderError,
)
from ..tools.base import Tool
from .base import ModelProvider, ProviderInfo, ProviderSession, ToolInvoker, Usage
from .chatloop import ChatChunk, ChatLoopSession, ChatMessage, ToolCallDelta

DEFAULT_N_CTX = 8192
DEFAULT_N_GPU_LAYERS = -1  # every layer on the GPU; the library's own default is 0 (CPU)

#: Thinking models are asked not to think. Pass ``chat_template_kwargs={}`` to send nothing.
DEFAULT_CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": False}

INSTALL_HINT = (
    "Install it with: pip install 'kennel[llama-cpp]' (or uv sync --extra llama-cpp). "
    "PyPI ships an sdist only, so this builds llama.cpp from source: about a minute on "
    "Apple Silicon, Xcode Command Line Tools required, cmake is fetched automatically."
)

MODEL_HINT = (
    "Kennel never downloads models. Point providers.llama-cpp.model_path in kennel.json at a "
    "GGUF file you already have, for example one from Qwen/Qwen3-4B-GGUF."
)

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"


# -- the library -------------------------------------------------------------


@dataclass(frozen=True)
class _Lib:
    """The three llama-cpp-python names Kennel uses, imported in one place."""

    Llama: Any
    LlamaGrammar: Any
    Jinja2ChatFormatter: Any


def _import_lib() -> _Lib:
    """Import llama-cpp-python, or say how to get it. Importing loads no model."""
    try:
        import llama_cpp
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    except ImportError as exc:
        raise ModelUnavailableError(f"llama-cpp-python is not installed: {exc}. {INSTALL_HINT}") from exc
    return _Lib(llama_cpp.Llama, llama_cpp.LlamaGrammar, Jinja2ChatFormatter)


def _map_error(exc: BaseException) -> BaseException:
    """Map a generation failure to the error a Kennel consumer should see."""
    if isinstance(exc, ValueError) and "exceed context window" in str(exc):
        return ContextLimitError(f"The request exceeded the llama.cpp context window: {exc}")
    if isinstance(exc, ProviderError):
        return exc
    if isinstance(exc, Exception):
        return ProviderError(f"llama.cpp generation failed: {exc}")
    return exc


# -- parsing the raw completion ----------------------------------------------


#: What a tag found in answer text means: which mode to continue in once it is dropped.
#: ``</think>`` is listed so that a model which emits only a closing tag (the template
#: prefills the opening one when thinking is disabled) does not leak it into the answer.
_TAGS: dict[str, str] = {THINK_OPEN: "think", TOOL_OPEN: "tool", THINK_CLOSE: "text"}


def _held_back(buffer: str) -> int:
    """How much of ``buffer``'s tail could still turn into a tag.

    Text is emitted as it arrives, except for a tail like ``"<to"`` that may be the
    start of ``<tool_call>``: releasing it would leak markup into the answer, so it
    waits for the next chunk.
    """
    for size in range(min(len(buffer), max(len(tag) for tag in _TAGS) - 1), 0, -1):
        tail = buffer[-size:]
        if any(tag.startswith(tail) for tag in _TAGS):
            return size
    return 0


class _CompletionParser:
    """Turns raw completion text into answer text and tool calls, chunk by chunk.

    The model writes one stream that mixes three things: answer text, a ``<think>``
    scratchpad and ``<tool_call>`` JSON. Tags arrive split across chunks (they are
    several tokens long), so this is a small state machine over a buffer rather than a
    regex over the whole completion: ``feed`` may be called with a single character and
    still produces the same result (pinned by the tests).
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._mode = "text"  # "text" | "think" | "tool"
        self._calls = 0

    def feed(self, text: str) -> list[ChatChunk]:
        """Consume the next slice of completion text."""
        self._buffer += text
        chunks: list[ChatChunk] = []
        while True:
            if self._mode == "text":
                index, tag = self._next_tag()
                if tag is None:
                    keep = len(self._buffer) - _held_back(self._buffer)
                    ready, self._buffer = self._buffer[:keep], self._buffer[keep:]
                    if ready:
                        chunks.append(ChatChunk(ready))
                    return chunks
                if index:
                    chunks.append(ChatChunk(self._buffer[:index]))
                self._buffer = self._buffer[index + len(tag) :]
                self._mode = _TAGS[tag]
                continue
            close = THINK_CLOSE if self._mode == "think" else TOOL_CLOSE
            index = self._buffer.find(close)
            if index < 0:
                if self._mode == "think":
                    # The scratchpad is never emitted, so only enough of it is kept to
                    # recognise a closing tag that straddles two chunks.
                    self._buffer = self._buffer[-(len(close) - 1) :]
                return chunks
            if self._mode == "tool":
                chunks.append(self._call(self._buffer[:index]))
            self._buffer = self._buffer[index + len(close) :]
            self._mode = "text"

    def finish(self) -> list[ChatChunk]:
        """Flush what the end of the stream leaves behind."""
        if self._mode == "text":
            ready, self._buffer = self._buffer, ""
            return [ChatChunk(ready)] if ready else []
        if self._mode == "tool" and self._buffer.strip():
            # A stop string can eat the closing tag; the JSON before it is still a request.
            payload, self._buffer = self._buffer, ""
            self._mode = "text"
            return [self._call(payload)]
        self._buffer, self._mode = "", "text"
        return []  # an unfinished <think> block is dropped, like a finished one

    def _next_tag(self) -> tuple[int, str | None]:
        """The earliest tag in the buffer; ``None`` when there is none (yet)."""
        candidates = [(self._buffer.find(tag), tag) for tag in _TAGS]
        found = [(index, tag) for index, tag in candidates if index >= 0]
        return min(found) if found else (len(self._buffer), None)

    def _call(self, payload: str) -> ChatChunk:
        """One ``<tool_call>`` body as a tool-call delta.

        Unreadable JSON is forwarded with no name rather than dropped or turned into
        answer text: the shared loop then tells the model its request was unusable, and
        no tool is invoked (nothing reaches a tool except through ``invoke``).
        """
        index, self._calls = self._calls, self._calls + 1
        delta = ToolCallDelta(index=index, id=f"call_{index}", arguments=payload.strip())
        try:
            parsed = json.loads(payload)
        except ValueError:
            return ChatChunk(tool_calls=[delta])
        if not isinstance(parsed, dict) or not isinstance(parsed.get("name"), str):
            return ChatChunk(tool_calls=[delta])
        arguments = parsed.get("arguments", {})
        delta.name = parsed["name"]
        delta.arguments = arguments if isinstance(arguments, str) else json.dumps(arguments)
        return ChatChunk(tool_calls=[delta])


# -- the transport -----------------------------------------------------------


def _render_message(message: ChatMessage) -> dict[str, Any]:
    """The wire form of one message, with ``content`` always present.

    An HTTP API is handed ``content: null`` or nothing at all and does not mind, but a
    chat template is Jinja: Qwen3's reads ``message.content`` directly, so a message
    that left the key out (an assistant turn that is nothing but tool calls) raises
    ``'dict object' has no attribute 'content'`` mid-render. Verified against
    Qwen3-4B-GGUF.
    """
    payload = message.to_dict()
    payload.setdefault("content", "")
    return payload


class LlamaCppTransport:
    """One :class:`~kennel.providers.chatloop.ChatTransport` request, in this process.

    The ``Llama``, the formatter and the lock all belong to the provider: a transport
    is per session, the model is not.
    """

    def __init__(
        self,
        lib: _Lib,
        llama: Any,
        formatter: Any,
        lock: threading.Lock,
        template_kwargs: dict[str, Any],
    ) -> None:
        self._lib = lib
        self._llama = llama
        self._formatter = formatter
        self._lock = lock
        self._template_kwargs = template_kwargs
        self._prompt_tokens: list[int] = []

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[dict[str, Any]],
        schema: dict[str, Any] | None,
    ) -> AsyncIterator[ChatChunk]:
        """Render, tokenize, generate, and yield chunks on the caller's loop.

        The last chunk carries ``usage``. ``input_tokens`` is exact -- it is the length
        of the token list that was fed in. ``output_tokens`` is the model's own
        tokenizer counting the generated text again, because a streamed completion
        reports no token count and its chunks are not one token each; around a stop
        string that re-count can differ from the number of sampled tokens by a token or
        so. Both numbers are measured by the model's tokenizer, never estimated from
        characters (principle 4).
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[ChatChunk | BaseException | None] = asyncio.Queue()
        leaving = threading.Event()

        def put(item: ChatChunk | BaseException | None) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:  # pragma: no cover - the caller's loop is gone
                pass

        def worker() -> None:
            completion: Iterator[dict[str, Any]] | None = None
            try:
                with self._lock:  # one model, one KV cache: generations are serialised
                    if leaving.is_set():  # cancelled while waiting for the lock
                        return
                    completion = self._start(messages, tools, schema)
                    finish_reason: str | None = None
                    raw: list[str] = []
                    parser = _CompletionParser()
                    for event in completion:
                        if leaving.is_set():
                            return  # between tokens: what interrupt() waits for
                        choice = (event.get("choices") or [{}])[0]
                        text = choice.get("text") or ""
                        raw.append(text)
                        finish_reason = choice.get("finish_reason") or finish_reason
                        for chunk in parser.feed(text):
                            put(chunk)
                    for chunk in parser.finish():
                        put(chunk)
                    put(ChatChunk(finish_reason=finish_reason, usage=self._usage("".join(raw))))
            except BaseException as exc:  # noqa: BLE001 - forwarded to the caller
                if not leaving.is_set():
                    put(_map_error(exc))
            finally:
                if completion is not None:
                    completion.close()  # closed here: this thread owns the generator
                put(None)

        threading.Thread(target=worker, name="kennel-llama-cpp", daemon=True).start()
        try:
            while True:
                item = await queue.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            leaving.set()

    async def close(self) -> None:
        return None  # the model belongs to the provider and outlives every session

    def _start(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[dict[str, Any]],
        schema: dict[str, Any] | None,
    ) -> Iterator[dict[str, Any]]:
        """Render the chat template, tokenize it and open the completion stream."""
        rendered = self._formatter(
            messages=[_render_message(message) for message in messages],
            tools=list(tools) or None,  # None, not []: what a template's `if tools` expects
            **self._template_kwargs,
        )
        self._prompt_tokens = self._tokenize(rendered.prompt)
        grammar = (
            self._lib.LlamaGrammar.from_json_schema(json.dumps(schema), verbose=False)
            if schema is not None
            else None
        )
        return self._llama.create_completion(
            self._prompt_tokens,
            stream=True,
            # None means "as much as the window allows": the library's own default is 16.
            max_tokens=None,
            stop=rendered.stop,
            stopping_criteria=getattr(rendered, "stopping_criteria", None),
            grammar=grammar,
        )

    def _tokenize(self, text: str) -> list[int]:
        return self._llama.tokenize(text.encode("utf-8"), add_bos=False, special=True)

    def _usage(self, raw: str) -> Usage:
        return Usage(input_tokens=len(self._prompt_tokens), output_tokens=len(self._tokenize(raw)))


# -- the provider ------------------------------------------------------------


class LlamaCppProvider(ModelProvider):
    """Run a GGUF model inside this process.

    ``model_path`` is the only required option (``~`` is expanded). ``n_ctx`` is the
    context window and is what :attr:`ProviderInfo.context_window_tokens` reports;
    ``n_gpu_layers`` defaults to every layer on the GPU. ``chat_template_kwargs`` is
    passed to the GGUF's chat template and defaults to ``{"enable_thinking": False}``
    -- pass ``{}`` to send nothing. ``llama_kwargs`` reaches the ``Llama`` constructor
    for anything else (``n_threads``, ``seed``, ...); ``verbose`` stays off, because
    Kennel's stdout is a machine-readable contract.
    """

    def __init__(
        self,
        *,
        model_path: str,
        n_ctx: int = DEFAULT_N_CTX,
        n_gpu_layers: int = DEFAULT_N_GPU_LAYERS,
        chat_template_kwargs: dict[str, Any] | None = None,
        llama_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_path = Path(str(model_path)).expanduser()
        self._n_ctx = int(n_ctx)
        if self._n_ctx <= 0:
            raise ConfigurationError(f"llama-cpp: n_ctx must be positive, got {n_ctx!r}")
        self._n_gpu_layers = int(n_gpu_layers)
        self._template_kwargs = (
            dict(DEFAULT_CHAT_TEMPLATE_KWARGS) if chat_template_kwargs is None else dict(chat_template_kwargs)
        )
        self._llama_kwargs = dict(llama_kwargs) if llama_kwargs else {}
        self._lock = threading.Lock()  # one model, one KV cache: load and generation
        self._lib: _Lib | None = None
        self._llama: Any | None = None
        self._formatter: Any | None = None
        self.info = ProviderInfo(
            name="llama-cpp",
            model=self.model_path.name,
            # In-process inference opens no port and reaches no network (principle 1).
            mode="local",
            context_window_tokens=self._n_ctx,
        )

    def check_availability(self) -> None:
        """Can this model be used here? Answered without loading it (nor a GPU context)."""
        self._lib = _import_lib()
        if not self.model_path.is_file():
            raise ModelUnavailableError(f"No GGUF model at {self.model_path}. {MODEL_HINT}")
        if self._n_ctx <= 0:  # pragma: no cover - the constructor refuses this
            raise ConfigurationError(f"llama-cpp: n_ctx must be positive, got {self._n_ctx!r}")

    async def create_session(
        self, *, instructions: str, tools: Sequence[Tool], invoke: ToolInvoker
    ) -> ProviderSession:
        if self._llama is None:
            # Availability first, so a caller who never asked gets the install hint or the
            # path instead of an ImportError from inside the load.
            self.check_availability()
            # Loading takes seconds on a cold page cache; keep the caller's loop free.
            await asyncio.to_thread(self._load)
        transport = LlamaCppTransport(
            self._lib, self._llama, self._formatter, self._lock, self._template_kwargs
        )
        return ChatLoopSession(transport, instructions=instructions, tools=tools, invoke=invoke)

    def _load(self) -> None:
        """Load the model once, and build the chat formatter from its own metadata."""
        with self._lock:
            if self._llama is not None:  # another session won the race
                return
            lib = self._lib or _import_lib()
            llama = lib.Llama(
                model_path=str(self.model_path),
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False,  # stderr stays clean; stdout is a contract
                **self._llama_kwargs,
            )
            template = (llama.metadata or {}).get("tokenizer.chat_template")
            if not template:
                raise ModelUnavailableError(
                    f"{self.model_path.name} carries no tokenizer.chat_template. Kennel renders the "
                    "model's own template and parses Qwen3-style <tool_call> JSON, so it needs a GGUF "
                    "that ships one."
                )
            self._formatter = lib.Jinja2ChatFormatter(
                template=template,
                eos_token=_special(llama, llama.token_eos()),
                bos_token=_special(llama, llama.token_bos()),
                stop_token_ids=[llama.token_eos()],
            )
            self._lib, self._llama = lib, llama


def _special(llama: Any, token: int) -> str:
    """The text of a special token, or ``""`` when the model has none (``bos`` on Qwen3)."""
    if token is None or token < 0:
        return ""
    return llama.detokenize([token], special=True).decode("utf-8", "replace")


# -- registry hooks (see kennel.providers.registry) --------------------------


def llama_cpp_provider(**options: Any) -> LlamaCppProvider:
    return LlamaCppProvider(**options)


def llama_cpp_doctor_checks(**options: Any) -> list[Check]:
    """The ``kennel doctor`` checks for an in-process model: the package, and the file."""
    try:
        provider = LlamaCppProvider(**options)
    except (ConfigurationError, TypeError) as exc:
        return [
            Check("llama-cpp-python", False, f"llama-cpp options: {exc}", hint="fix providers.llama-cpp in the config"),
            Check("model", False, "skipped: the provider could not be built"),
        ]
    try:
        _import_lib()
    except ModelUnavailableError as exc:
        return [
            Check("llama-cpp-python", False, str(exc), hint=INSTALL_HINT),
            Check("model", False, "skipped: llama_cpp is not importable"),
        ]
    import llama_cpp

    package = Check(
        "llama-cpp-python", True, f"llama_cpp {getattr(llama_cpp, '__version__', 'unknown')} importable"
    )
    if not provider.model_path.is_file():
        return [
            package,
            Check("model", False, f"no GGUF model at {provider.model_path}", hint=MODEL_HINT),
        ]
    size = provider.model_path.stat().st_size / (1024**3)
    return [
        package,
        Check(
            "model",
            True,
            f"{provider.model_path} ({size:.1f} GiB, context window: "
            f"{provider.info.context_window_tokens} tokens, mode: {provider.info.mode})",
        ),
    ]
