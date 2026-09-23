"""The fetch tool without a network: URLs, the address check, rules, grants, text (issue #78).

The HTTP side (a loopback server, redirects, TLS, interrupts) is tests/providers/test_fetch_http.py.
"""

import pytest

from kennel import Agent, KennelConfig, MockProvider
from kennel.cli.main import build_agent, build_parser
from kennel.errors import ToolArgumentError, ToolExecutionError
from kennel.permissions import Decision, PermissionKind, PermissionManager
from kennel.providers.mock import Text, ToolCall
from kennel.registry import DEFAULT_TOOLS, builtin_registry
from kennel.tools import fetch
from kennel.tools.fetch import FetchTool, _address_problem, _parse, html_to_text, normalize_host

# -- the URL -------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,host",
    [
        ("https://Docs.Python.org./3/", "docs.python.org"),
        ("https://bücher.example/x", "xn--bcher-kva.example"),
        ("http://[::1]:8080/", "::1"),
        ("http://127.0.0.1/", "127.0.0.1"),
    ],
)
def test_the_host_is_normalised(raw, host):
    assert _parse(raw).host == host


@pytest.mark.parametrize(
    "raw",
    ["ftp://example.com/", "file:///etc/passwd", "example.com/page", "https://user:pw@example.com/", "https://:80/", "http://[::1/"],
)
def test_unusable_urls_are_refused(raw):
    with pytest.raises(ToolArgumentError):
        _parse(raw)


def test_the_fragment_is_never_sent_and_the_query_is():
    assert _parse("https://example.com/a/b?x=1#frag").path == "/a/b?x=1"
    assert _parse("https://example.com").path == "/"


def test_same_origin_and_upgrade_are_followed_the_rest_is_not():
    here = _parse("http://example.com/a")
    assert here.may_follow(_parse("http://example.com:80/b"))
    assert here.may_follow(_parse("https://example.com/b"))  # http -> https, default ports
    assert not here.may_follow(_parse("https://example.com:8443/b"))
    assert not here.may_follow(_parse("http://example.com:8080/b"))
    assert not here.may_follow(_parse("http://www.example.com/b"))
    assert not _parse("https://example.com/a").may_follow(_parse("http://example.com/b"))  # no downgrade


# -- the address check (AC-3) ------------------------------------------------------


@pytest.mark.parametrize(
    "address,category",
    [
        ("127.0.0.1", "loopback"),
        ("::1", "loopback"),
        ("10.1.2.3", "private"),
        ("192.168.0.1", "private"),
        ("172.16.0.1", "private"),
        ("169.254.169.254", "link-local"),
        ("fe80::1", "link-local"),
        ("224.0.0.1", "multicast"),
        ("ff02::1", "multicast"),
        ("240.0.0.1", "reserved"),
        ("0.0.0.0", "unspecified"),
        ("::", "unspecified"),
        ("::ffff:127.0.0.1", "loopback"),  # IPv4-mapped IPv6 is unwrapped
        ("::ffff:10.0.0.1", "private"),
        ("100.64.0.1", "non-global"),  # shared address space
        ("not-an-ip", "unrecognised"),
    ],
)
def test_non_global_addresses_are_named(address, category):
    assert _address_problem(address) == category


@pytest.mark.parametrize("address", ["93.184.216.34", "8.8.8.8", "2606:4700:4700::1111"])
def test_global_addresses_pass(address):
    assert _address_problem(address) is None


@pytest.mark.parametrize("resolved", [["10.0.0.5"], ["93.184.216.34", "10.0.0.5"], ["::ffff:169.254.169.254"]])
async def test_one_bad_address_refuses_before_any_connection(ctx, monkeypatch, resolved):
    connects = []
    monkeypatch.setattr(fetch, "_connect", lambda *a: connects.append(a))
    tool = FetchTool(resolver=lambda host, port: resolved)
    with pytest.raises(ToolExecutionError) as info:
        await tool.execute({"url": "https://intranet.example/admin"}, ctx)
    assert connects == []
    message = str(info.value)
    assert "intranet.example" in message and "address" in message
    for address in resolved:  # the model learns the category, not the internal address
        assert address not in message and address.split(":")[-1] not in message


def test_the_config_cannot_lift_the_address_check():
    """Only an application-built FetchTool(allow_private_addresses=True) reaches internal addresses."""
    tool = builtin_registry().create("fetch", KennelConfig(tools=["fetch"], permissions={"fetch": "allow"}))
    assert isinstance(tool, FetchTool) and tool.allow_private_addresses is False
    assert FetchTool(allow_private_addresses=True).allow_private_addresses is True


# -- what is recorded, what is asked (AC-6) ------------------------------------------


def test_summary_drops_query_fragment_and_user_info():
    tool = FetchTool()
    assert tool.summarize({"url": "https://Docs.python.org/3/x?token=secret#part"}) == "Fetch https://Docs.python.org/3/x"
    assert "pw" not in tool.summarize({"url": "https://me:pw@example.com/?q=1"})
    assert "secret" not in tool.summarize({"url": "ftp://example.com/p?secret"})
    assert tool.summarize({"url": "http://[::1/?secret"}) == "Fetch (an invalid URL)"


def test_the_prompt_shows_the_whole_url(ctx):
    assert FetchTool().permission_details({"url": "https://example.com/p?q=1"}, ctx) == "https://example.com/p?q=1"
    with pytest.raises(ToolArgumentError):  # refused before anyone is asked
        FetchTool().permission_details({"url": "gopher://example.com/"}, ctx)


# -- rules and grants per host (AC-8) --------------------------------------------------


@pytest.mark.parametrize(
    "url,matches",
    [
        ("https://docs.python.org/3/library/", True),
        ("http://DOCS.python.org.:8080/x", True),  # normalised; scheme and port do not matter
        ("https://python.org/", False),
        ("https://evil.example/?x=docs.python.org", False),
        ("https://docs.python.org.evil.example/", False),
        ("not a url", False),
    ],
)
def test_a_rule_matches_the_normalised_host_only(url, matches):
    assert FetchTool().match_rule("*.python.org", {"url": url}) is matches


def test_rules_use_the_host_through_the_permission_manager():
    tool = FetchTool()
    pm = PermissionManager({"fetch": "ask", "fetch(*.python.org)": "allow", "fetch(evil.example)": "deny"})
    decide = lambda url: pm.decision_for("fetch", PermissionKind.WEB, {"url": url}, tool.match_rule)  # noqa: E731
    assert decide("https://docs.python.org/3/") is Decision.ALLOW
    assert decide("https://python.org/") is Decision.ASK
    assert decide("https://EVIL.example./x") is Decision.DENY


def test_the_session_scope_is_the_host_and_never_the_whole_tool():
    tool = FetchTool()
    assert tool.session_scope({"url": "https://Docs.Python.org/a?q"}) == "docs.python.org"
    assert tool.session_scope({"url": "http://[::1/"}) == "http://[::1/"  # not None: a bad URL grants nothing wider


# -- page text (AC-1) -----------------------------------------------------------------


def test_html_is_reduced_to_its_text():
    html = """<html><head><title>T</title><style>p{color:red}</style>
    <script>var secret = 1;</script></head><body><h1>Heading</h1><p>One &amp; <b>two</b></p>
    <noscript>no js</noscript><ul><li>a</li><li>b</li></ul></body></html>"""
    assert html_to_text(html) == "T\nHeading\nOne & two\na\nb"


def test_normalize_host_keeps_ip_literals_compact():
    assert normalize_host("::FFFF:0A00:0001") == "::ffff:a00:1"


# -- enabling it (AC-2) ----------------------------------------------------------------


def cli_agent(ws, *flags):
    return build_agent(build_parser().parse_args([str(ws), "--provider", "mock", *flags]), None)


def test_fetch_is_off_unless_allow_fetch(meeting_ws):
    assert "fetch" not in DEFAULT_TOOLS
    assert "fetch" not in cli_agent(meeting_ws).tools
    assert "fetch" not in cli_agent(meeting_ws, "--allow-web").tools  # web does not bring fetch
    asking = cli_agent(meeting_ws, "--allow-fetch")
    assert "fetch" in asking.tools and "web" not in asking.tools
    assert asking.permissions.policy()["fetch"] is Decision.ASK
    bypass = cli_agent(meeting_ws, "--allow-fetch", "--permission-mode", "bypass")
    assert bypass.permissions.policy()["fetch"] is Decision.ALLOW


async def test_listing_fetch_in_the_sdk_does_not_allow_it(meeting_ws):
    """Like web: tools=["fetch"] without a rule is denied in the default mode."""
    agent = Agent(meeting_ws, provider=MockProvider([[ToolCall("fetch", {"url": "https://example.com/"}), Text("ok")]]), tools=["fetch"])
    result = await agent.run("go")
    assert [c.status for c in result.tool_calls] == ["denied"]
