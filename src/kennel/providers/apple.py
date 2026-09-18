"""AppleProvider: Apple Foundation Models through ``apple_fm_sdk``.

Design notes (verified against apple_fm_sdk 0.2.0):

* The SDK runs the tool-calling loop itself and invokes ``Tool.call`` on a
  worker thread with its own event loop, so Kennel's ``invoke`` is awaited
  right there and must never depend on the caller's loop.
* ``stream_response`` blocks the loop it runs on between snapshots. Every SDK
  request is therefore run on a dedicated model thread, keeping the caller's
  loop (and the CLI) responsive and cancellable.
* SDK ``Tool`` instances must stay referenced for the session's lifetime or
  the process crashes on the next tool call; the session holds them.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any, Optional, TypeVar

from ..errors import ContextLimitError, ModelUnavailableError, ProviderError
from ..tools.base import Tool, ToolParameter
from .base import ModelProvider, ProviderInfo, ProviderSession, ToolInvoker

T = TypeVar("T")

INSTALL_HINT = (
    "apple_fm_sdk is not installed. Install it with: pip install 'apple-fm-sdk==0.2.0' "
    "(requires macOS 26+, Apple Silicon, Xcode and Apple Intelligence enabled)."
)


def _import_sdk():
    try:
        import apple_fm_sdk as fm
    except ImportError as exc:
        raise ModelUnavailableError(INSTALL_HINT) from exc
    return fm


def _unavailable_message(fm, reason) -> str:
    R = fm.SystemLanguageModelUnavailableReason
    if reason == R.APPLE_INTELLIGENCE_NOT_ENABLED:
        return "Apple Intelligence is not enabled. Turn it on in System Settings > Apple Intelligence & Siri."
    if reason == R.DEVICE_NOT_ELIGIBLE:
        return "This Mac is not eligible for Apple Intelligence (Apple Silicon with macOS 26+ required)."
    if reason == R.MODEL_NOT_READY:
        return "The on-device model is still downloading or preparing. Try again in a few minutes."
    return f"The on-device model is unavailable ({reason})."


def _map_error(fm, exc: BaseException) -> BaseException:
    if isinstance(exc, fm.ExceededContextWindowSizeError):
        return ContextLimitError("The request exceeded the on-device model's context window.")
    if isinstance(exc, fm.GuardrailViolationError):
        return ProviderError("The on-device model's safety guardrails blocked this request.")
    if isinstance(exc, fm.RefusalError):
        return ProviderError("The on-device model declined to answer this request.")
    if isinstance(exc, fm.ConcurrentRequestsError):
        return ProviderError("The model is still busy with a previous request. Try again shortly.")
    if isinstance(exc, fm.RateLimitedError):
        return ProviderError("The model rate-limited this request. Try again shortly.")
    if isinstance(exc, fm.FoundationModelsError):
        return ProviderError(f"Foundation Models error: {exc}")
    return exc


def _apple_schema(schema: dict[str, Any], name: str = "Response") -> dict[str, Any]:
    """Return a copy of a JSON schema in the dialect Foundation Models expects.

    Every object schema needs a ``title``, an ``x-order`` list of its property
    names and ``additionalProperties: false``; nested objects and array items
    included.
    """
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "properties" and isinstance(value, dict):
            out[key] = {
                prop: _apple_schema(sub, _title(prop)) if isinstance(sub, dict) else sub for prop, sub in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            out[key] = _apple_schema(value, _title(name.rstrip("s") or name) + "Item" if name else "Item")
        else:
            out[key] = value
    if out.get("type") == "object" and isinstance(out.get("properties"), dict):
        out.setdefault("title", name)
        out.setdefault("x-order", list(out["properties"].keys()))
        out.setdefault("additionalProperties", False)
    return out


def _title(name: str) -> str:
    return "".join(part.capitalize() for part in name.replace("-", "_").split("_")) or "Object"


def _python_type(param: ToolParameter):
    base = {"string": str, "integer": int, "number": float, "boolean": bool}[param.type]
    return base if param.required else Optional[base]  # noqa: UP045 - the SDK inspects the "Optional" name


def _bridge_tool(fm, tool: Tool, invoke: ToolInvoker):
    from apple_fm_sdk.generation_property import Property
    from apple_fm_sdk.generation_schema import GenerationSchema

    type_name = "".join(part.capitalize() for part in tool.name.replace("-", "_").split("_")) + "Args"
    schema = GenerationSchema(
        type_class=type(type_name, (), {}),
        description=f"Arguments for {tool.name}",
        properties=[Property(p.name, _python_type(p), p.description) for p in tool.parameters],
    )
    tool_name, tool_description = tool.name, tool.description

    class BridgedTool(fm.Tool):
        name = tool_name
        description = tool_description

        @property
        def arguments_schema(self):
            return schema

        async def call(self, args) -> str:
            raw = args.value()
            return await invoke(tool_name, dict(raw) if isinstance(raw, dict) else {})

    return BridgedTool()


async def _on_model_thread(factory: Callable[[], Awaitable[T]]) -> T:
    """Run ``factory()`` on a fresh thread + event loop; cancellation propagates."""
    caller_loop = asyncio.get_running_loop()
    done: asyncio.Future[T] = caller_loop.create_future()
    state: dict[str, Any] = {}

    def deliver(fn: Callable[[], None]) -> None:
        if not done.done():
            fn()

    def worker() -> None:
        loop = asyncio.new_event_loop()
        try:
            task = loop.create_task(factory())
            state["loop"], state["task"] = loop, task
            try:
                result = loop.run_until_complete(task)
            except BaseException as exc:  # noqa: BLE001 - forwarded to the caller
                error = exc  # bind before the except block clears the name
                caller_loop.call_soon_threadsafe(deliver, lambda: done.set_exception(error))
            else:
                caller_loop.call_soon_threadsafe(deliver, lambda: done.set_result(result))
        finally:
            loop.close()

    threading.Thread(target=worker, name="kennel-model", daemon=True).start()
    try:
        return await done
    except asyncio.CancelledError:
        loop, task = state.get("loop"), state.get("task")
        if loop is not None and task is not None and not task.done():
            loop.call_soon_threadsafe(task.cancel)
        raise


class AppleSession(ProviderSession):
    def __init__(self, fm, session, bridged_tools: list[Any], options=None) -> None:
        self._fm = fm
        self._session = session
        self._bridged_tools = bridged_tools  # keep alive: SDK holds raw pointers to these
        self._options = options

    async def respond(self, prompt: str) -> str:
        try:
            return await _on_model_thread(lambda: self._session.respond(prompt, options=self._options))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise _map_error(self._fm, exc) from exc

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        caller_loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None | BaseException] = asyncio.Queue()

        def put(item: str | None | BaseException) -> None:
            caller_loop.call_soon_threadsafe(queue.put_nowait, item)

        async def pump() -> None:
            try:
                async for snapshot in self._session.stream_response(prompt, options=self._options):
                    put(snapshot)
            except BaseException as exc:  # noqa: BLE001 - forwarded to the consumer
                put(exc)
                raise
            else:
                put(None)

        worker = asyncio.ensure_future(_on_model_thread(pump))
        previous = ""
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise _map_error(self._fm, item) from item
                if item.startswith(previous):
                    delta = item[len(previous) :]
                else:  # non-monotonic snapshot: emit the unseen tail
                    common = 0
                    for a, b in zip(previous, item, strict=False):
                        if a != b:
                            break
                        common += 1
                    delta = item[common:]
                previous = item
                if delta:
                    yield delta
        finally:
            if not worker.done():
                worker.cancel()
            try:
                await worker
            except (asyncio.CancelledError, Exception):
                pass

    async def respond_structured(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        try:
            apple_schema = _apple_schema(schema)
            content = await _on_model_thread(
                lambda: self._session.respond(prompt, json_schema=apple_schema, options=self._options)
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise _map_error(self._fm, exc) from exc
        value = content.value()
        return dict(value) if isinstance(value, dict) else {"value": value}


class AppleProvider(ModelProvider):
    # apple_fm_sdk 0.2.0 exposes no token counter (Transcript only has from_dict/to_dict
    # and LanguageModelSession keeps its transcript private), so usage() stays unimplemented
    # and Kennel estimates. The window is the documented ~4k of the on-device model.
    info = ProviderInfo(
        name="apple", model="Apple SystemLanguageModel", mode="local", context_window_tokens=4096
    )

    def __init__(self, *, deterministic: bool = False) -> None:
        """``deterministic=True`` uses greedy sampling: same input, same output (useful for evaluation)."""
        self._fm = None
        self._model = None
        self.deterministic = deterministic

    @property
    def fm(self):
        if self._fm is None:
            self._fm = _import_sdk()
        return self._fm

    @property
    def model(self):
        if self._model is None:
            self._model = self.fm.SystemLanguageModel()
        return self._model

    def check_availability(self) -> None:
        ok, reason = self.model.is_available()
        if not ok:
            raise ModelUnavailableError(_unavailable_message(self.fm, reason))

    async def create_session(self, *, instructions: str, tools: Sequence[Tool], invoke: ToolInvoker) -> ProviderSession:
        fm = self.fm
        bridged = [_bridge_tool(fm, tool, invoke) for tool in tools]
        options = fm.GenerationOptions(sampling=fm.SamplingMode.greedy()) if self.deterministic else None
        try:
            session = fm.LanguageModelSession(instructions=instructions, model=self.model, tools=bridged)
        except Exception as exc:  # noqa: BLE001
            raise _map_error(fm, exc) from exc
        return AppleSession(fm, session, bridged, options)
