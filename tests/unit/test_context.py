import pytest

from kennel.context import HistoryTurn, chunk_text, compact_history, map_reduce, truncate_text


def test_truncate_text():
    assert truncate_text("abc", 10) == ("abc", False)
    text, truncated = truncate_text("x" * 100, 40)
    assert truncated and text.endswith("[output truncated]") and len(text.encode()) <= 40
    text, _ = truncate_text("日本語" * 50, 40)
    assert len(text.encode()) <= 40  # never splits a multibyte character


def test_chunk_text_boundaries_and_overlap():
    assert chunk_text("", 100) == []
    assert chunk_text("short", 100) == ["short"]
    paras = [f"paragraph {i} " + "word " * 20 for i in range(10)]
    text = "\n\n".join(paras)
    chunks = chunk_text(text, max_chars=400, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)
    assert "".join(chunks).count("paragraph 0") >= 1 and chunks[-1].strip().endswith("word")
    for i in range(9):
        assert any(f"paragraph {i}" in c for c in chunks)
    # overlap: tail of chunk n appears at the head of chunk n+1
    assert chunks[1][:20] in chunks[0] or chunks[1].split()[0] in chunks[0]


def test_chunk_text_hard_splits_long_lines():
    chunks = chunk_text("a" * 1000, max_chars=300, overlap=0)
    assert [len(c) for c in chunks] == [300, 300, 300, 100]
    with pytest.raises(ValueError):
        chunk_text("x", 0)


def test_compact_history():
    assert compact_history([]) == ""
    text = compact_history([HistoryTurn("what is  x?", "x is " + "y" * 500)], per_item=40)
    assert text.startswith("Summary of the conversation so far")
    assert "- User asked: what is x?" in text
    assert "..." in text and len(text) < 200
    long = compact_history([HistoryTurn("q" * 100, "a" * 100)] * 50, max_chars=500)
    assert len(long) <= 500 and long.endswith("(truncated)")


async def test_map_reduce_uses_fresh_sessions():
    opened = []

    class FakeSession:
        def __init__(self):
            self.closed = False

        async def respond(self, prompt):
            return f"R({prompt})"

        async def close(self):
            self.closed = True

    async def open_session():
        s = FakeSession()
        opened.append(s)
        return s

    out = await map_reduce(
        open_session,
        ["c1", "c2"],
        map_prompt=lambda i, n, c: f"map{i}/{n}:{c}",
        reduce_prompt=lambda parts: "reduce:" + "|".join(parts),
    )
    assert out == "R(reduce:R(map1/2:c1)|R(map2/2:c2))"
    assert len(opened) == 3 and all(s.closed for s in opened)
    single = await map_reduce(open_session, ["only"], map_prompt=lambda i, n, c: c, reduce_prompt=lambda p: "never")
    assert single == "R(only)"


def test_looks_like_tool_narration():
    from kennel.context import looks_like_tool_narration

    tools = ["glob", "grep", "read"]
    assert looks_like_tool_narration("以下のコマンドを実行します。\n```bash\ngrep x\n```", tools)
    assert looks_like_tool_narration("まず田中さんのメールを検索します。", tools)
    assert looks_like_tool_narration("I'll run grep for the term first.", tools)
    assert looks_like_tool_narration("Let me search the workspace.", tools)
    assert looks_like_tool_narration("I would use the read tool on that file.", tools)
    assert looks_like_tool_narration("Bobからの依頼について、どのファイルを確認すればよいでしょうか？", tools)
    assert looks_like_tool_narration("まず、特定のファイルを探して確認できますか？", tools)
    assert looks_like_tool_narration("Which file should I check?", tools)
    assert looks_like_tool_narration("該当するファイルを特定するために、ファイル検索を行います。", tools)
    assert looks_like_tool_narration("該当するファイルを探して読み取らせてください。", tools)
    assert looks_like_tool_narration("Bob asked for CI config; the file notes/todo.md has more.", tools)
    assert looks_like_tool_narration("Would you like me to search the transcripts?", tools)
    assert not looks_like_tool_narration("The decision was to ship v0.1 on Friday.", tools)
    assert not looks_like_tool_narration("確認しました。決定事項は以下です。", tools)
    assert not looks_like_tool_narration("", tools)
