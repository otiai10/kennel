"""AC-4: the CLI's `↻` nudge line shows which rule fired.

Kept in its own file (not test_cli_instructions.py) because pytest here runs in
prepend import mode without `tests/__init__.py`, so test file basenames must be
unique across the whole tree.
"""

import io

from kennel.cli.renderer import Renderer
from kennel.events import Event, EventType


def _rendered(data: dict) -> str:
    out = io.StringIO()
    renderer = Renderer(out=out, err=io.StringIO(), color=False)
    renderer.on_event(Event(EventType.MODEL_NUDGED, "s1", data))
    return out.getvalue()


def test_nudge_line_includes_the_rule():
    text = _rendered({"chars": 42, "rule": "file_mention"})
    assert "↻ carrying out the described steps instead of narrating them" in text
    assert "(rule: file_mention)" in text


def test_nudge_line_without_a_rule_omits_the_suffix():
    # Defensive: an older event payload without `rule` must still render cleanly.
    text = _rendered({"chars": 42})
    assert "↻ carrying out the described steps instead of narrating them" in text
    assert "(rule:" not in text
