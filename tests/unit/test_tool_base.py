import pytest

from kennel.errors import ToolArgumentError
from kennel.tools.base import Tool, ToolParameter, ToolResult


class Demo(Tool):
    name = "demo"
    description = "demo"
    parameters = (
        ToolParameter("s", "string", "s"),
        ToolParameter("i", "integer", "i", required=False),
        ToolParameter("n", "number", "n", required=False),
        ToolParameter("b", "boolean", "b", required=False),
    )

    async def execute(self, arguments, context):
        return ToolResult("ok")


def test_validate_coerces_and_drops_unknown():
    out = Demo().validate({"s": 5, "i": "10", "n": "1.5", "b": "true", "zzz": 1})
    assert out == {"s": "5", "i": 10, "n": 1.5, "b": True}
    assert Demo().validate({"s": "x", "i": 3.0, "b": 0}) == {"s": "x", "i": 3, "b": False}


def test_validate_missing_required():
    with pytest.raises(ToolArgumentError, match="missing required argument 's'"):
        Demo().validate({})
    with pytest.raises(ToolArgumentError):
        Demo().validate({"s": None})


@pytest.mark.parametrize("bad", [{"s": "x", "i": "ten"}, {"s": "x", "i": True}, {"s": "x", "b": "maybe"}, {"s": "x", "n": "abc"}, {"s": ["list"]}])
def test_validate_type_errors(bad):
    with pytest.raises(ToolArgumentError):
        Demo().validate(bad)


def test_default_summary():
    assert Demo().summarize({"s": "x"}) == 'Demo {"s": "x"}'
