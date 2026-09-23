"""Web search against real services (issue #61). They leave the machine, so they are opt-in.

    KENNEL_SEARXNG_URL=http://localhost:8888 pytest tests/integration -m search
    KENNEL_BRAVE_TESTS=1 BRAVE_API_KEY=... pytest tests/integration -m search

A SearXNG needs ``json`` in ``search.formats`` of its settings.yml. Brave searches count
against the key's quota. The tests check structure, never particular results.
"""

import os

import pytest

from kennel.search import create

SEARXNG_URL = os.environ.get("KENNEL_SEARXNG_URL")
BRAVE = os.environ.get("KENNEL_BRAVE_TESTS") == "1" and bool(os.environ.get("BRAVE_API_KEY"))

pytestmark = pytest.mark.search


def _check(results) -> None:
    assert 0 < len(results) <= 3
    assert all(r.url.startswith("http") and r.title for r in results)


@pytest.mark.skipif(not SEARXNG_URL, reason="set KENNEL_SEARXNG_URL=<url> to search through a SearXNG you run")
async def test_searxng_live():
    _check(await create("searxng", {"url": SEARXNG_URL}).search("Apple Foundation Models", limit=3))


@pytest.mark.skipif(not BRAVE, reason="set KENNEL_BRAVE_TESTS=1 and BRAVE_API_KEY to spend a Brave search")
async def test_brave_live():
    _check(await create("brave").search("Apple Foundation Models", limit=3))
